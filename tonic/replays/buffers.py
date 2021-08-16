from collections import deque

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

        kwargs = self.unzip_dict_observations(**kwargs)
        kwargs.pop('infos')

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

            yield transitions
            
    def unzip_dict_observations(self, **kwargs):
        # Extract elements in dictionary observation
        if 'observations' in kwargs:
            obs = kwargs.pop('observations')
            
            # Store keys of observations
            self.observation_keys = obs[0].keys()

            for ob in obs:
                for key, val in ob.items():
                    try:
                        kwargs[key].append(val)
                    except KeyError:
                        kwargs[key] = [val]

        if 'next_observations' in kwargs:
            obs = kwargs.pop('next_observations')
            for ob in obs:
                for key, val in ob.items():
                    try:
                        kwargs["next_"+key].append(val)
                    except KeyError:
                        kwargs["next_"+key] = [val]

        return kwargs


@gin.configurable
class HerBuffer(DictBuffer):
    def __init__(
        self, size=int(1e6), num_steps=1, batch_iterations=40, batch_size=256, 
        discount_factor=0.95, steps_before_batches=int(1e4)-1, 
        steps_between_batches=50, goal_selection_strategy='future',
        n_sampled_goal=4, max_timesteps=50, reward_function=None,
        handle_timeout_termination=True
    ):
        super(HerBuffer, self).__init__(size, num_steps, batch_iterations,
                                        batch_size, discount_factor,
                                        steps_before_batches, 
                                        steps_between_batches)
        
        self.goal_selection_strategy = goal_selection_strategy
        self.n_sampled_goal = n_sampled_goal

        self.max_timesteps = max_timesteps
        self.max_n_episodes_stored = self.max_size // self.max_timesteps

        # compute ratio between HER replays and regular replays
        self.her_ratio = 1 - (1.0 / (1 + self.n_sampled_goal))

        self.handle_timeout_termination = handle_timeout_termination

        self.reward_function = reward_function

    def initialize(self, seed):
        super().initialize(seed)
        self.pos = 0
        self.episode_index = 0
        self.full = False
        self.episode_lengths = np.zeros(self.max_n_episodes_stored, 
                                        dtype=np.int64)

    def set_reward_function(self, reward_function):
        assert reward_function is not None
        self.reward_function = reward_function

    def store(self, **kwargs):

        if self.episode_index == 0 and self.full:
            # Clear info buffer
            self.info_buffer[self.pos] = deque(
                maxlen=self.max_n_episodes_stored)
            
        if 'terminations' in kwargs:
            continuations = np.float32(1 - kwargs['terminations'])
            kwargs['discounts'] = continuations * self.discount_factor

        kwargs = self.unzip_dict_observations(**kwargs)
        infos = kwargs.pop('infos')

        # Create the named buffers.
        if self.buffers is None:
            self.num_workers = len(list(kwargs.values())[0])
            self.buffers = {}
            for key, val in kwargs.items():
                shape = (self.max_n_episodes_stored, self.max_timesteps, ) + \
                    np.array(val).shape
                self.buffers[key] = np.full(shape, np.nan, np.float32)

            self.info_buffer = [deque(maxlen=self.max_timesteps) for _ in \
                range(self.max_n_episodes_stored)]

        # Store the new values.
        for key, val in kwargs.items():
            self.buffers[key][self.pos, self.episode_index] = val

        self.info_buffer[self.pos].append(infos)

        # Accumulate values for n-step returns.
        if self.num_steps > 1:
            self.accumulate_n_steps(kwargs)

        self.episode_index += 1
        self.steps += 1
        
        done = False
        for i in range(self.num_workers):
            if kwargs['resets'][i]:
                done = True
        
        if done or self.episode_index >= self.max_timesteps:
            self.next_episode()
        
    def next_episode(self):

        self.episode_lengths[self.pos] = self.episode_index
        
        self.pos += 1
        if self.pos == self.max_n_episodes_stored:
            self.full = True
            self.pos = 0

        self.episode_index = 0

    @property
    def n_episodes_stored(self):
        if self.full:
            return self.max_n_episodes_stored
        return self.pos

    def get(self, *keys):
        '''Get batches from named buffers.'''

        for _ in range(self.batch_iterations):
            temp_buffers = {}
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.n_episodes_stored]
            
            sampled_transitions = self.sample_transitions()

            transitions = {}
            for key in keys:
                if key == 'observations':
                    transitions[key] = {k: sampled_transitions[k][:, 0]
                                        for k in self.observation_keys}
                elif key == 'next_observations':
                    transitions[key] = {k: sampled_transitions["next_"+k][:, 0]
                                        for k in self.observation_keys}
                else:
                    transitions[key] = sampled_transitions[key][:, 0]
            yield transitions
            

    def sample_transitions(self):

        if self.full:
            episode_indices = (
                np.random.randint(1, self.n_episodes_stored, self.batch_size) + 
                self.pos) % self.n_episodes_stored
        else:
            episode_indices = np.random.randint(0, self.n_episodes_stored, 
                                                self.batch_size)

        her_indices = np.arange(
            self.batch_size)[: int(self.her_ratio * self.batch_size)]

        ep_lengths = self.episode_lengths[episode_indices]

        # Filter episodes with 1 state: no transition available
        if self.goal_selection_strategy == 'future':
            her_indices = her_indices[ep_lengths[her_indices] > 1]
            ep_lengths[her_indices] -= 1

        transitions_indices = np.random.randint(ep_lengths)

        transitions = {key: self.buffers[key][episode_indices, 
                                              transitions_indices].copy()
                       for key in self.buffers.keys()}

        new_goals = self.sample_goals(episode_indices, her_indices, 
                                      transitions_indices)

        transitions["desired_goal"][her_indices] = new_goals

        transitions["info"] = np.array(
            [self.info_buffer[episode_idx][transition_idx]
             for episode_idx, transition_idx in zip(episode_indices, 
                                                    transitions_indices)])

        if len(her_indices) > 0:
            transitions["rewards"][her_indices, 0] = self.reward_function(
                transitions["next_achieved_goal"][her_indices, 0],
                transitions["desired_goal"][her_indices, 0],
                transitions["info"][her_indices, 0],
            )

        return transitions

    def sample_goals(self, episode_indices, her_indices, transitions_indices):
        her_episode_indices = episode_indices[her_indices]

        # replay with k random states from the episodes after current transitions
        if self.goal_selection_strategy == 'future':
            transitions_indices = np.random.randint(
                transitions_indices[her_indices] + 1, 
                self.episode_lengths[her_episode_indices]
            )

        # replay with final state of the episodes
        elif self.goal_selection_strategy == 'final':
            transitions_indices = self.episode_lengths[her_episode_indices] - 1

        # replay with random state of the episodes
        elif self.goal_selection_strategy == 'episode':
            transitions_indices = np.random.randint(
                self.episode_lengths[her_episode_indices])
        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy}" +
                             "for sampling goals not supported!")

        return self.buffers["achieved_goal"][her_episode_indices, 
                                             transitions_indices]