import gin
import numpy as np


@gin.configurable
class Buffer:
    '''Replay storing a large number of transitions for off-policy learning
    and using n-step returns.'''

    def __init__(
        self, size=int(1e6), num_steps=1, batch_iterations=50, batch_size=100,
        discount_factor=0.99, steps_before_batches=int(1e4),
        steps_between_batches=50
    ):
        self.max_size = size
        self.num_steps = num_steps
        self.batch_iterations = batch_iterations
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.steps_before_batches = steps_before_batches
        self.steps_between_batches = steps_between_batches

    def initialize(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        self.buffers = None
        self.index = 0
        self.size = 0
        self.steps = 0

    def ready(self):
        if self.steps < self.steps_before_batches:
            return False
        return self.steps % self.steps_between_batches == 0

    def store(self, **kwargs):
        if 'terminations' in kwargs:
            continuations = np.float32(1 - kwargs['terminations'])
            kwargs['discounts'] = continuations * self.discount_factor

        # Create the named buffers.
        if self.buffers is None:
            self.num_workers = len(list(kwargs.values())[0])
            self.buffers = {}
            for key, val in kwargs.items():
                shape = (self.max_size,) + np.array(val).shape
                self.buffers[key] = np.full(shape, np.nan, np.float32)

        # Store the new values.
        for key, val in kwargs.items():
            self.buffers[key][self.index] = val

        # Accumulate values for n-step returns.
        if self.num_steps > 1:
            self.accumulate_n_steps(kwargs)

        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.steps += 1

    def accumulate_n_steps(self, kwargs):
        rewards = kwargs['rewards']
        next_observations = kwargs['next_observations']
        discounts = kwargs['discounts']
        masks = np.ones(self.num_workers, np.float32)

        for i in range(min(self.size, self.num_steps - 1)):
            index = (self.index - i - 1) % self.max_size
            masks *= (1 - self.buffers['resets'][index])
            new_rewards = (self.buffers['rewards'][index] +
                           self.buffers['discounts'][index] * rewards)
            self.buffers['rewards'][index] = (
                (1 - masks) * self.buffers['rewards'][index] +
                masks * new_rewards)
            new_discounts = self.buffers['discounts'][index] * discounts
            self.buffers['discounts'][index] = (
                (1 - masks) * self.buffers['discounts'][index] +
                masks * new_discounts)
            self.buffers['next_observations'][index] = (
                (1 - masks)[:, None] *
                self.buffers['next_observations'][index] +
                masks[:, None] * next_observations)

    def get(self, *keys):
        '''Get batches from named buffers.'''

        for _ in range(self.batch_iterations):
            total_size = self.size * self.num_workers
            indices = self.np_random.randint(total_size, size=self.batch_size)
            rows = indices // self.num_workers
            columns = indices % self.num_workers
            yield {k: self.buffers[k][rows, columns] for k in keys}


@gin.configurable
class DictBuffer(Buffer):
    def __init__(
        self, size=int(1e6), num_steps=1, batch_iterations=50, batch_size=100, 
        discount_factor=0.99, steps_before_batches=int(1e4), 
        steps_between_batches=50
    ):
        super(DictBuffer, self).__init__(size, num_steps, batch_iterations,
                                        batch_size, discount_factor, 
                                        steps_before_batches, 
                                        steps_between_batches)

    def store(self, **kwargs):

        if 'terminations' in kwargs:
            continuations = np.float32(1 - kwargs['terminations'])
            kwargs['discounts'] = continuations * self.discount_factor

        # Extract elements in dictionary observation
        if 'observations' in kwargs:
            obs = kwargs['observations']
            kwargs.pop('observations')
            
            # Store keys of observations
            self.observation_keys = obs[0].keys()

            for ob in obs:
                for key, val in ob.items():
                    try:
                        kwargs[key].append(val)
                    except KeyError:
                        kwargs[key] = [val]

        if 'next_observations' in kwargs:
            obs = kwargs['next_observations']
            kwargs.pop('next_observations')
            for ob in obs:
                for key, val in ob.items():
                    try:
                        kwargs["next_"+key].append(val)
                    except KeyError:
                        kwargs["next_"+key] = [val]

        # Create the named buffers.
        if self.buffers is None:
            self.num_workers = len(list(kwargs.values())[0])
            self.buffers = {}
            for key, val in kwargs.items():
                shape = (self.max_size,) + np.array(val).shape
                self.buffers[key] = np.full(shape, np.nan, np.float32)

        # Store the new values.
        for key, val in kwargs.items():
            self.buffers[key][self.index] = val

        # Accumulate values for n-step returns.
        if self.num_steps > 1:
            self.accumulate_n_steps(kwargs)

        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.steps += 1
        

    def get(self, *keys):
        '''Get batches from named buffers.'''
        # if self.goal_selection_strategy == 'future':
        #     pass

        for _ in range(self.batch_iterations):
            total_size = self.size * self.num_workers
            indices = self.np_random.randint(total_size, size=self.batch_size)
            rows = indices // self.num_workers
            columns = indices % self.num_workers

            transitions = {}
            for key in keys:
                if key == 'observations':
                    transitions[key] = {
                        k: self.buffers[k][rows, columns] \
                            for k in self.observation_keys}

                elif key == 'next_observations':
                    transitions[key] = {
                        k: self.buffers["next_" + k][rows, columns] \
                            for k in self.observation_keys}
                else:
                    transitions[key] = self.buffers[key][rows, columns]
            
            # print(self.buffers['observation'].shape)
            # print(transitions['observations']['observation'].shape)
            # print(self.buffers['observation'][rows, columns].shape)
            # print(self.buffers['actions'].shape)
            # print(self.buffers['actions'][rows, columns].shape)
            # print(transitions['actions'].shape)

            # import time
            # time.sleep(100)

            
            yield transitions


@gin.configurable
class HerBuffer(DictBuffer):
    def __init__(
        self, size=int(1e6), num_steps=1, batch_iterations=50, batch_size=100, 
        discount_factor=0.99, steps_before_batches=int(1e4), 
        steps_between_batches=50, goal_selection_strategy='future',
        n_sampled_goal=4, time_horizon=500,
    ):
        super(HerBuffer, self).__init__(size, num_steps, batch_iterations,
                                        batch_size, discount_factor,
                                        steps_before_batches, 
                                        steps_between_batches)
        
        self.goal_selection_strategy = goal_selection_strategy

        self.time_horizon = time_horizon
        self.max_episode_stored = self.size // self.time_horizon
        
    def store_episode(self, episode_batch):

        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        batch_size = batch_sizes[0]
        
        idxs = self._get_storage_index(batch_size)

        for key in self.buffers.keys():
            self.buffers[key][idxs] = episode_batch[key]

        self.n_transitions_stored += batch_size * self.time_horizon

    def _get_storage_index(self, inc):
        inc = inc or 1

        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc) 
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randInt(0, self.size, inc)

        self.current_size = min(self.size, self.current_size + inc)

        if inc == 1:
            idx = idx[0]
            
        