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
        super(DictBuffer, self).__init__(
            size, num_steps, batch_iterations, batch_size, discount_factor,
            steps_before_batches, steps_between_batches)

    def _unpack_dict_observations(self, kwargs):
        # Unpack observations
        observations = kwargs.pop('observations')
        for key in observations.keys():
            kwargs[key] = observations[key]

        # Store observation keys
        self.observation_keys = observations.keys()

        # Unpack next_observations
        next_observations = kwargs.pop('next_observations')

        for key in next_observations.keys():
            kwargs['next_'+key] = next_observations[key]

        return kwargs

    def store(self, **kwargs):

        if 'terminations' in kwargs:
            continuations = np.float32(1 - kwargs['terminations'])
            kwargs['discounts'] = continuations * self.discount_factor

        kwargs = self._unpack_dict_observations(kwargs)

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
        for _ in range(self.batch_iterations):
            total_size = self.size * self.num_workers
            indices = self.np_random.randint(total_size, size=self.batch_size)
            rows = indices // self.num_workers
            columns = indices % self.num_workers

            transitions = {}

            for key in keys:
                # Zip dictionary observations
                if key == 'observations':
                    transitions[key] = {
                        k: self.buffers[k][rows, columns]
                        for k in self.observation_keys}

                # Zip dictionary observations
                elif key == 'next_observations':
                    transitions[key] = {
                        k: self.buffers["next_"+k][rows, columns]
                        for k in self.observation_keys}
                else:
                    transitions[key] = self.buffers[key][rows, columns]

            yield transitions


@gin.configurable
class HerBuffer(DictBuffer):
    def __init__(
        self, size=int(1e6), num_steps=1, batch_iterations=40, batch_size=256,
        discount_factor=0.98, steps_before_batches=int(1e4),
        steps_between_batches=50, goal_selection_strategy='future',
        replay_k=4, reward_function=None
    ):
        super(HerBuffer, self).__init__(size, num_steps, batch_iterations,
                                        batch_size, discount_factor,
                                        steps_before_batches,
                                        steps_between_batches)

        self.goal_selection_strategy = goal_selection_strategy
        self.replay_k = replay_k

        self.reward_function = reward_function

        # Ratio between HER replays and regular replays
        self.her_ratio = 1 - (1.0 / (1 + self.replay_k))

    def initialize(self, seed):
        super().initialize(seed)
        self.full = False

    def set_reward_function(self, reward_function):
        assert reward_function is not None
        self.reward_function = reward_function

    def store(self, **kwargs):

        if 'terminations' in kwargs:
            continuations = np.float32(1 - kwargs['terminations'])
            kwargs['discounts'] = continuations * self.discount_factor

        # Unpack dictionary observations
        kwargs = self._unpack_dict_observations(kwargs)

        # Create the named buffers
        if self.buffers is None:
            self.num_workers = len(list(kwargs.values())[0])
            self.buffers = {}
            for key, val in kwargs.items():
                shape = (self.max_size, ) + np.array(val).shape
                self.buffers[key] = np.full(shape, np.nan, np.float32)

            self.buffers['episode_n'] = np.full(
                (self.max_size, self.num_workers), np.nan, np.int)
            self.buffers['end_of_episodes'] = np.full(
                (self.max_size, self.num_workers), np.nan, np.int)

            # Keep track of episode number for each environment.
            self.current_episode = np.zeros((self.num_workers), dtype=np.int)

        # Store the new values.
        for key, val in kwargs.items():
            self.buffers[key][self.index] = val

        # Store the episode of the current timestep.
        self.buffers['episode_n'][self.index] = self.current_episode
        # Store the current timestep as the end of episode.
        self.buffers['end_of_episodes'][self.current_episode] = self.index

        # Accumulate values for n-step returns.
        if self.num_steps > 1:
            self.accumulate_n_steps(kwargs)

        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.full = (self.size == self.max_size)
        self.steps += 1

        # Reset current_episode if buffers['end_of_episodes'] is full.
        for i in range(self.num_workers):
            if self.current_episode[i] >= self.max_size:
                self.current_episode[i] = 0

        # Store the timestep and increment episode_n if the environment resets
        for i, reset in enumerate(kwargs['resets']):
            if reset:
                self.current_episode[i] += 1

    def get(self, *keys):
        '''Get batches from named buffers.'''
        for _ in range(self.batch_iterations):
            unzipped_transitions = self.sample_transitions()
            transitions = {}
            for key in keys:
                # Zip dictionary observations
                if key == 'observations':
                    transitions[key] = {k: unzipped_transitions[k]
                                        for k in self.observation_keys}
                elif key == 'next_observations':
                    transitions[key] = {k: unzipped_transitions['next_'+k]
                                        for k in self.observation_keys}
                else:
                    transitions[key] = unzipped_transitions[key]

            yield transitions

    def sample_transitions(self):
        # Sample timesteps
        total_size = self.size * self.num_workers
        indices = self.np_random.randint(total_size, size=self.batch_size)
        rows = indices // self.num_workers
        columns = indices % self.num_workers

        batch_her_proportion = \
            np.arange(self.batch_size)[: int(self.her_ratio * self.batch_size)]

        # Indices to replace goals (HER)
        her_rows = rows[batch_her_proportion]
        her_columns = columns[batch_her_proportion]

        samples = {key: self.buffers[key][rows, columns].copy()
                   for key in self.buffers.keys()}

        her_goals = self.sample_goals(her_rows, her_columns)

        samples['desired_goal'][batch_her_proportion] = her_goals
        samples['next_desired_goal'][batch_her_proportion] = her_goals

        # Recalculate rewards based on a new goal
        samples['rewards'][batch_her_proportion] = self.reward_function(
            samples['next_achieved_goal'][batch_her_proportion],
            samples['desired_goal'][batch_her_proportion],
            None
        )

        return samples

    def sample_goals(self, her_rows, her_columns):

        episode_n = self.buffers['episode_n'][her_rows, her_columns]

        end_of_episodes = self.buffers['end_of_episodes'][episode_n, her_columns]

        # Add self.size to indices where start index is bigger than
        # index at the end of episode.
        end_of_episodes[end_of_episodes < her_rows] += self.size

        if self.goal_selection_strategy == 'future':
            her_rows = self.np_random.randint(her_rows, end_of_episodes+1) \
                % self.size
        elif self.goal_selection_strategy == 'final':
            her_rows = end_of_episodes
        elif self.goal_selection_strategy == 'episode':
            her_rows = self.np_random.randint(end_of_episodes+1)
        else:
            raise ValueError(f"{self.goal_selection_strategy} goal selection" +
                             "strategy is not supported.")

        return self.buffers['achieved_goal'][her_rows, her_columns]
