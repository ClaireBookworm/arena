# %%
import os
from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np
import gym
import gym.spaces
import gym.envs.registration
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm, trange
import sys
import time
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Tuple
import torch as t
from torch import nn, Tensor
from gym.spaces import Discrete, Box
from numpy.random import Generator
import pandas as pd
import wandb
import pandas as pd
from pathlib import Path
from jaxtyping import Float, Int, Bool
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')

Arr = np.ndarray

# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_dqn"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from part1_intro_to_rl.utils import make_env
from part1_intro_to_rl.solutions import Environment, Toy, Norvig, find_optimal_policy
import part2_q_learning_and_dqn.utils as utils
import part2_q_learning_and_dqn.tests as tests
from plotly_utils import line, cliffwalk_imshow, plot_cartpole_obs_and_dones

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
ObsType = int
ActType = int

class DiscreteEnviroGym(gym.Env):
	action_space: gym.spaces.Discrete
	observation_space: gym.spaces.Discrete
	'''
	A discrete environment class for reinforcement learning, compatible with OpenAI Gym.

	This class represents a discrete environment where actions and observations are discrete.
	It is designed to interface with a provided `Environment` object which defines the
	underlying dynamics, states, and actions.

	Attributes:
		action_space (gym.spaces.Discrete): The space of possible actions.
		observation_space (gym.spaces.Discrete): The space of possible observations (states).
		env (Environment): The underlying environment with its own dynamics and properties.
	'''
	def __init__(self, env: Environment):
		super().__init__()
		self.env = env
		self.observation_space = gym.spaces.Discrete(env.num_states)
		self.action_space = gym.spaces.Discrete(env.num_actions)
		self.reset()

	def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
		'''
		Execute an action and return the new state, reward, done flag, and additional info.
		The behaviour of this function depends primarily on the dynamics of the underlying
		environment.
		'''
		(states, rewards, probs) = self.env.dynamics(self.pos, action)
		idx = self.np_random.choice(len(states), p=probs)
		(new_state, reward) = (states[idx], rewards[idx])
		self.pos = new_state
		done = self.pos in self.env.terminal
		return (new_state, reward, done, {"env": self.env})

	def reset(self, seed: Optional[int] = None, options=None) -> ObsType:
		'''
		Resets the environment to its initial state.
		'''
		super().reset(seed=seed)
		self.pos = self.env.start
		return self.pos

	def render(self, mode="human"):
		assert mode == "human", f"Mode {mode} not supported!"
# %%
gym.envs.registration.register(
	id="NorvigGrid-v0",
	entry_point=DiscreteEnviroGym,
	max_episode_steps=100,
	nondeterministic=True,
	kwargs={"env": Norvig(penalty=-0.04)},
)

gym.envs.registration.register(
	id="ToyGym-v0",
	entry_point=DiscreteEnviroGym,
	max_episode_steps=2,
	nondeterministic=False,
	kwargs={"env": Toy()}
)
# %%
@dataclass
class Experience:
	'''
	A class for storing one piece of experience during an episode run.
	'''
	obs: ObsType
	act: ActType
	reward: float
	new_obs: ObsType
	new_act: Optional[ActType] = None


@dataclass
class AgentConfig:
	'''Hyperparameters for agents'''
	epsilon: float = 0.1
	lr: float = 0.05
	optimism: float = 0

defaultConfig = AgentConfig()


class Agent:
	'''Base class for agents interacting with an environment (you do not need to add any implementation here)'''
	rng: np.random.Generator

	def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma: float = 0.99, seed: int = 0):
		self.env = env
		self.reset(seed)
		self.config = config
		self.gamma = gamma
		self.num_actions = env.action_space.n
		self.num_states = env.observation_space.n
		self.name = type(self).__name__

	def get_action(self, obs: ObsType) -> ActType:
		raise NotImplementedError()

	def observe(self, exp: Experience) -> None:
		'''
		Agent observes experience, and updates model as appropriate.
		Implementation depends on type of agent.
		'''
		pass

	def reset(self, seed: int) -> None:
		self.rng = np.random.default_rng(seed)

	def run_episode(self, seed) -> List[int]:
		'''
		Simulates one episode of interaction, agent learns as appropriate
		Inputs:
			seed : Seed for the random number generator
		Outputs:
			The rewards obtained during the episode
		'''
		rewards = []
		obs = self.env.reset(seed=seed)
		self.reset(seed=seed)
		done = False
		while not done:
			act = self.get_action(obs)
			(new_obs, reward, done, info) = self.env.step(act)
			exp = Experience(obs, act, reward, new_obs)
			self.observe(exp)
			rewards.append(reward)
			obs = new_obs
		return rewards

	def train(self, n_runs=500):
		'''
		Run a batch of episodes, and return the total reward obtained per episode
		Inputs:
			n_runs : The number of episodes to simulate
		Outputs:
			The discounted sum of rewards obtained for each episode
		'''
		all_rewards = []
		for seed in trange(n_runs):
			rewards = self.run_episode(seed)
			all_rewards.append(utils.sum_rewards(rewards, self.gamma))
		return all_rewards


class Random(Agent):
	def get_action(self, obs: ObsType) -> ActType:
		return self.rng.integers(0, self.num_actions)
# %%
class EpsilonGreedy(Agent):
	'''
	A class for SARSA and Q-Learning to inherit from.
	'''
	def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma: float = 0.99, seed: int = 0):
		super().__init__(env, config, gamma, seed)
		self.Q = np.zeros((self.num_states, self.num_actions)) + self.config.optimism
		self.epsilon = config.epsilon

	def get_action(self, obs: ObsType) -> ActType:
		'''
		Selects an action using epsilon-greedy with respect to Q-value estimates
		'''
		choose_best_action = self.rng.random(size=1).item() > self.epsilon
		if choose_best_action:
			action = np.argmax(self.Q[obs], axis=-1)
		else:
			action = self.rng.integers(low=0, high=self.num_actions, size=1).item()
		return action


class QLearning(EpsilonGreedy):
	def observe(self, exp: Experience) -> None:
		st = exp.obs
		at = exp.act
		rt1 = exp.reward
		st1 = exp.new_obs
		at1 = exp.new_act
		# ^ defining sarsa values
		# alpha = 0.9
		# idx = np.argmax(self.Q[st1], axis=-1) 
		# gamma * max Q st+1, a
		max_val = self.gamma * self.Q[st1].max()
		self.Q[st, at] = self.Q[st, at] + self.config.lr * (rt1 + max_val - self.Q[st, at])

class SARSA(EpsilonGreedy):
	def observe(self, exp: Experience):
		st = exp.obs
		at = exp.act
		rt1 = exp.reward
		st1 = exp.new_obs
		at1 = exp.new_act # can do this in onle line 
		# set various alpha
		# alpha = 0.9
		self.Q[st, at] += self.config.lr * (rt1 + self.gamma * self.Q[st1,at1]- self.Q[st, at]) 
		# gamma is the change 
		

	def run_episode(self, seed) -> List[int]:
		rewards = []
		obs = self.env.reset(seed=seed)
		act = self.get_action(obs)
		self.reset(seed=seed)
		done = False
		while not done:
			(new_obs, reward, done, info) = self.env.step(act)
			new_act = self.get_action(new_obs)
			exp = Experience(obs, act, reward, new_obs, new_act)
			self.observe(exp)
			rewards.append(reward)
			obs = new_obs
			act = new_act
		return rewards


n_runs = 1000
gamma = 0.99
seed = 1
env_norvig = gym.make("NorvigGrid-v0") # many gym models
config_norvig = AgentConfig()
args_norvig = (env_norvig, config_norvig, gamma, seed)
agents_norvig: List[Agent] = [Cheater(*args_norvig), QLearning(*args_norvig), SARSA(*args_norvig), Random(*args_norvig)]
returns_norvig = {}
fig = go.Figure(layout=dict(
	title_text=f"Avg. reward on {env_norvig.spec.name}",
	template="simple_white",
	xaxis_range=[-30, n_runs+30],
	width=700, height=400,
))
for agent in agents_norvig:
	returns = agent.train(n_runs)
	fig.add_trace(go.Scatter(y=utils.cummean(returns), name=agent.name))
fig.show()
# %%
gamma = 1
seed = 0

# gamma = 1 for the punishment to be -1 

config_cliff = AgentConfig(epsilon=0.1, lr = 0.1, optimism=0)
env = gym.make("CliffWalking-v0")
n_runs = 2500
args_cliff = (env, config_cliff, gamma, seed)

returns_list = []
name_list = []
agents: List[Union[QLearning, SARSA]] = [QLearning(*args_cliff), SARSA(*args_cliff)]

for agent in agents:
	returns = agent.train(n_runs)[1:]
	returns_list.append(utils.cummean(returns))
	name_list.append(agent.name)
	V = agent.Q.max(axis=-1).reshape(4, 12)
	pi = agent.Q.argmax(axis=-1).reshape(4, 12)
	cliffwalk_imshow(V, pi, title=f"CliffWalking: {agent.name} Agent", width=800, height=400)

line(
	returns_list,
	names=name_list,
	template="simple_white",
	title="Q-Learning vs SARSA on CliffWalking-v0",
	labels={"x": "Episode", "y": "Avg. reward", "variable": "Agent"},
	width=700, height=400,
)
# %% 
# Deep Q Networks 

class QNetwork(nn.Module):
	'''For consistency with your tests, please wrap your modules in a `nn.Sequential` called `layers`.'''
	layers: nn.Sequential

	def __init__(
		self,
		dim_observation: int,
		num_actions: int,
		hidden_sizes: List[int] = [120, 84]
	):
		super().__init__()
		self.obs_shape = dim_observation
		self.hidden_sizes = hidden_sizes
		self.num_actions = num_actions
		self.layers = nn.Sequential (
			t.nn.Linear(self.obs_shape, self.hidden_sizes[0]),
			t.nn.ReLU(),
			t.nn.Linear(*self.hidden_sizes),
			t.nn.ReLU(),
			t.nn.Linear(self.hidden_sizes[-1], self.num_actions)
		)
		# output dimensions - num_actions
		# outputs the q values for each action
		# want to use special methods to write variable numbers of relu-linear pairs 

	def forward(self, x: t.Tensor) -> t.Tensor:
		# x is the observation
		return self.layers(x)

# why do we do this? 
# %% 
@dataclass
class ReplayBufferSamples:
	'''
	Samples from the replay buffer, converted to PyTorch for use in neural network training.

	Data is equivalent to (s_t, a_t, r_{t+1}, d_{t+1}, s_{t+1}).
	'''
	observations: Tensor # shape [sample_size, *observation_shape]
	actions: Tensor # shape [sample_size, *action_shape]
	rewards: Tensor # shape [sample_size,]
	dones: Tensor # shape [sample_size,]
	next_observations: Tensor # shape [sample_size, observation_shape]

	def __post_init__(self):
		for exp in self.__dict__.values():
			assert isinstance(exp, Tensor), f"Error: expected type tensor, found {type(exp)}"


class ReplayBuffer:
	'''
	Contains buffer; has a method to sample from it to return a ReplayBufferSamples object.
	'''
	rng: Generator
	observations: np.ndarray # shape [buffer_size, *observation_shape]
	actions: np.ndarray # shape [buffer_size, *action_shape]
	rewards: np.ndarray # shape [buffer_size,]
	dones: np.ndarray # shape [buffer_size,]
	next_observations: np.ndarray # shape [buffer_size, *observation_shape]

	def __init__(self, num_environments: int, obs_shape: Tuple[int], action_shape: Tuple[int], buffer_size: int, seed: int):
		assert num_environments == 1, "This buffer only supports SyncVectorEnv with 1 environment inside."
		self.num_environments = num_environments
		self.obs_shape = obs_shape
		self.buffer_size = buffer_size
		self.rng = np.random.default_rng(seed)
		self.action_shape = action_shape

		self.observations = np.empty((0, *self.obs_shape), dtype=np.float32)
		self.actions = np.empty(0, dtype=np.int32)
		self.rewards = np.empty(0, dtype=np.float32)
		self.dones = np.empty(0, dtype=bool)
		self.next_observations = np.empty((0, *self.obs_shape), dtype=np.float32)


	def add(
		self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, next_obs: np.ndarray
	) -> None:
		'''
		obs: shape (num_environments, *observation_shape)
			Observation before the action
		actions: shape (num_environments, *action_shape)
			Action chosen by the agent
		rewards: shape (num_environments,)
			Reward after the action
		dones: shape (num_environments,)
			If True, the episode ended and was reset automatically
		next_obs: shape (num_environments, *observation_shape)
			Observation after the action
			If done is True, this should be the terminal observation, NOT the first observation of the next episode.
		'''
		assert obs.shape == (self.num_environments, *self.obs_shape)
		assert actions.shape == (self.num_environments, *self.action_shape)
		assert rewards.shape == (self.num_environments,)
		assert dones.shape == (self.num_environments,)
		assert next_obs.shape == (self.num_environments, *self.obs_shape)

		# We update each one manually, but you could also use a for loop with setattr & getattr
		self.observations = np.concatenate((self.observations, obs))[-self.buffer_size:]
		self.actions = np.concatenate((self.actions, actions))[-self.buffer_size:]
		self.rewards = np.concatenate((self.rewards, rewards))[-self.buffer_size:]
		self.dones = np.concatenate((self.dones, dones))[-self.buffer_size:]
		self.next_observations = np.concatenate((self.next_observations, next_obs))[-self.buffer_size:]


	def sample(self, sample_size: int, device: t.device) -> ReplayBufferSamples:
		'''
		Uniformly sample sample_size entries from the buffer and convert them to PyTorch tensors on device.
		Sampling is with replacement, and sample_size may be larger than the buffer size.
		'''
		current_buffer_size = self.observations.shape[0] # why do we need to get this here instead of from buffer_size? 
		idx = self.rng.integers(low=0, high=current_buffer_size, size=sample_size)
		obs_tensor = t.as_tensor(self.observations[idx]).to(device)
		act_tensor = t.from_numpy(self.actions[idx]).to(device)
		rewards_tensor = t.from_numpy(self.rewards[idx]).to(device)
		dones_tensor = t.from_numpy(self.dones[idx]).to(device)
		next_tensor = t.from_numpy(self.next_observations[idx]).to(device)
		return ReplayBufferSamples(obs_tensor, act_tensor, rewards_tensor, dones_tensor, next_tensor)


tests.test_replay_buffer_single(ReplayBuffer)
tests.test_replay_buffer_deterministic(ReplayBuffer)
tests.test_replay_buffer_wraparound(ReplayBuffer)
# %%rb = ReplayBuffer(num_environments=1, obs_shape=(4,), action_shape=(), buffer_size=256, seed=0)
envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 0, 0, False, "test")])
obs = envs.reset()
for i in range(256):
	# Choose a random next action, and take a step in the environment
	actions = envs.action_space.sample()
	(next_obs, rewards, dones, infos) = envs.step(actions)
	# Add observations to buffer, and set obs = next_obs ready for the next step
	rb.add(obs, actions, rewards, dones, next_obs)
	obs = next_obs

plot_cartpole_obs_and_dones(rb.observations, rb.dones, title="CartPole experiences s<sub>t</sub> (dotted lines = termination)")

sample = rb.sample(256, t.device("cpu"))
plot_cartpole_obs_and_dones(sample.observations, sample.dones, title="CartPole experiences s<sub>t</sub> (randomly sampled) (dotted lines = termination)")
# %% 
rb = ReplayBuffer(num_environments=1, obs_shape=(4,), action_shape=(), buffer_size=256, seed=0)
envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 0, 0, False, "test")])
obs = envs.reset()
for i in range(256):
	# Choose a random next action, and take a step in the environment
	actions = envs.action_space.sample()
	(next_obs, rewards, dones, infos) = envs.step(actions)

	# Get actual next_obs, by replacing next_obs with terminal observation at all envs which are terminated
	real_next_obs = next_obs.copy()
	for environment, done in enumerate(dones):
		if done:
			print(f'Environment {environment} terminated after {infos[0]["episode"]["l"]} steps')
			real_next_obs[environment] = infos[environment]["terminal_observation"]

	# Add the next_obs to the buffer (which has the terminated states), but set obs=new_obs (which has the restarted states)
	rb.add(obs, actions, rewards, dones, real_next_obs)
	obs = next_obs

plot_cartpole_obs_and_dones(rb.next_observations, rb.dones, title="CartPole experiences s<sub>t+1</sub> (dotted lines = termination)")# %%
# Montezumas revenge 

device = next(q_network.paramters()).device 
rand = rng.random()

n_envs = obs.shape[0]
if rand < epsilon: 
	# we are below the threshold, we take an random aciton
	# explore 
	actions = rng.integers(0, envs.single_action_space.n, (n_envs,))
else:
	logits = q_network.forward(obs)
	actions = t.argmax(logits, dim=-1)
# %%
def epsilon_greedy_policy(
	envs: gym.vector.SyncVectorEnv, q_network: QNetwork, rng: Generator, obs: np.ndarray, epsilon: float
) -> np.ndarray:
	'''With probability epsilon, take a random action. Otherwise, take a greedy action according to the q_network.
	Inputs:
		envs : gym.vector.SyncVectorEnv, the family of environments to run against
		q_network : QNetwork, the network used to approximate the Q-value function
		obs : The current observation
		epsilon : exploration percentage
	Outputs:
		actions: (n_environments, *action_shape) the sampled action for each environment.
	'''
	# Convert `obs` into a tensor so we can feed it into our model
	device = next(q_network.parameters()).device
	obs = t.from_numpy(obs).to(device)

	pass


tests.test_epsilon_greedy_policy(epsilon_greedy_policy)
# %%
ObsType = np.ndarray
ActType = int


class Probe1(gym.Env):
	'''One action, observation of [0.0], one timestep long, +1 reward.

	We expect the agent to rapidly learn that the value of the constant [0.0] observation is +1.0. Note we're using a continuous observation space for consistency with CartPole.
	'''

	action_space: Discrete
	observation_space: Box

	def __init__(self):
		super().__init__()
		self.observation_space = Box(np.array([0]), np.array([0]))
		self.action_space = Discrete(1)
		self.seed()
		self.reset()

	def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
		return (np.array([0]), 1.0, True, {})

	def reset(
		self, seed: Optional[int] = None, return_info=False, options=None
	) -> Union[ObsType, Tuple[ObsType, dict]]:
		super().reset(seed=seed)
		if return_info:
			return (np.array([0.0]), {})
		return np.array([0.0])


gym.envs.registration.register(id="Probe1-v0", entry_point=Probe1)
env = gym.make("Probe1-v0")
assert env.observation_space.shape == (1,)
assert env.action_space.shape == ()
# %%
class Probe2(gym.Env):
	'''One action, observation of [-1.0] or [+1.0], one timestep long, reward equals observation.

	We expect the agent to rapidly learn the value of each observation is equal to the observation.
	'''

	action_space: Discrete
	observation_space: Box

	def __init__(self):
		super().__init__()
		self.observation_space = Box(np.array([-1.0]), np.array([+1.0]))
		self.action_space = Discrete(1)
		self.reset()
		self.reward = None

	def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
		assert self.reward is not None
		return np.array([self.observation]), self.reward, True, {}

	def reset(
		self, seed: Optional[int] = None, return_info=False, options=None
	) -> Union[ObsType, Tuple[ObsType, dict]]:
		super().reset(seed=seed)
		self.reward = 1.0 if self.np_random.random() < 0.5 else -1.0
		self.observation = self.reward
		if return_info:
			return np.array([self.reward]), {}
		return np.array([self.reward])

gym.envs.registration.register(id="Probe2-v0", entry_point=Probe2)


class Probe3(gym.Env):
	'''One action, [0.0] then [1.0] observation, two timesteps, +1 reward at the end.

	We expect the agent to rapidly learn the discounted value of the initial observation.
	'''

	action_space: Discrete
	observation_space: Box

	def __init__(self):
		super().__init__()
		self.observation_space = Box(np.array([-0.0]), np.array([+1.0]))
		self.action_space = Discrete(1)
		self.reset()

	def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
		self.n += 1
		if self.n == 1:
			return np.array([1.0]), 0.0, False, {}
		elif self.n == 2:
			return np.array([0.0]), 1.0, True, {}
		raise ValueError(self.n)

	def reset(
		self, seed: Optional[int] = None, return_info=False, options=None
	) -> Union[ObsType, Tuple[ObsType, dict]]:
		# SOLUTION
		super().reset(seed=seed)
		self.n = 0
		if return_info:
			return np.array([0.0]), {}
		return np.array([0.0])

gym.envs.registration.register(id="Probe3-v0", entry_point=Probe3)


class Probe4(gym.Env):
	'''Two actions, [0.0] observation, one timestep, reward is -1.0 or +1.0 dependent on the action.

	We expect the agent to learn to choose the +1.0 action.
	'''

	action_space: Discrete
	observation_space: Box

	def __init__(self):
		self.observation_space = Box(np.array([-0.0]), np.array([+0.0]))
		self.action_space = Discrete(2)
		self.reset()

	def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
		reward = -1.0 if action == 0 else 1.0
		return np.array([0.0]), reward, True, {}

	def reset(
		self, seed: Optional[int] = None, return_info=False, options=None
	) -> Union[ObsType, Tuple[ObsType, dict]]:
		super().reset(seed=seed)
		if return_info:
			return np.array([0.0]), {}
		return np.array([0.0])

gym.envs.registration.register(id="Probe4-v0", entry_point=Probe4)


class Probe5(gym.Env):
	'''Two actions, random 0/1 observation, one timestep, reward is 1 if action equals observation otherwise -1.

	We expect the agent to learn to match its action to the observation.
	'''

	action_space: Discrete
	observation_space: Box

	def __init__(self):
		self.observation_space = Box(np.array([-1.0]), np.array([+1.0]))
		self.action_space = Discrete(2)
		self.reset()

	def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
		reward = 1.0 if action == self.obs else -1.0
		return np.array([self.obs]), reward, True, {}

	def reset(
		self, seed: Optional[int] = None, return_info=False, options=None
	) -> Union[ObsType, Tuple[ObsType, dict]]:
		super().reset(seed=seed)
		self.obs = 1.0 if self.np_random.random() < 0.5 else 0.0
		if return_info:
			return np.array([self.obs], dtype=float), {}
		return np.array([self.obs], dtype=float)

gym.envs.registration.register(id="Probe5-v0", entry_point=Probe5)
# %%
@dataclass
class DQNArgs:
	exp_name: str = "DQN_implementation"
	seed: int = 1
	torch_deterministic: bool = True
	cuda: bool = t.cuda.is_available()
	log_dir: str = "logs"
	use_wandb: bool = False
	wandb_project_name: str = "CartPoleDQN"
	wandb_entity: Optional[str] = None
	capture_video: bool = True
	env_id: str = "CartPole-v1"
	total_timesteps: int = 500_000
	learning_rate: float = 0.00025
	buffer_size: int = 10_000
	gamma: float = 0.99
	target_network_frequency: int = 500
	batch_size: int = 128
	start_e: float = 1.0
	end_e: float = 0.1
	exploration_fraction: float = 0.2
	train_frequency: int = 10
	log_frequency: int = 50

	def __post_init__(self):
		assert self.total_timesteps - self.buffer_size >= self.train_frequency
		self.total_training_steps = (self.total_timesteps - self.buffer_size) // self.train_frequency


args = DQNArgs(batch_size=256)
utils.arg_help(args)
# %%
class DQNAgent:
	'''Base Agent class handling the interaction with the environment.'''

	def __init__(
		self,
		envs: gym.vector.SyncVectorEnv,
		args: DQNArgs,
		rb: ReplayBuffer,
		q_network: QNetwork,
		target_network: QNetwork,
		rng: np.random.Generator
	):
		self.envs = envs
		self.args = args
		self.rb = rb
		self.next_obs = self.envs.reset() # Need a starting observation!
		self.steps = 0
		self.epsilon = args.start_e
		self.q_network = q_network
		self.target_network = target_network
		self.rng = rng

	def play_step(self) -> List[dict]:
		'''
		Carries out a single interaction step between the agent and the environment, and adds results to the replay buffer.

		Returns `infos` (list of dictionaries containing info we will log).
		'''
		# SOLUTION
		obs = self.next_obs
		actions = self.get_actions(obs)
		next_obs, rewards, dones, infos = self.envs.step(actions)
		real_next_obs = next_obs.copy() # makes a copy so edits don't change next_obs 
		for (environment, done) in enumerate(dones):
			if done:
				real_next_obs[environment] = infos[environment]["terminal_observation"]
				# we do this so we tell the agent they are doign something wrong -> that we tell it has leaved the bounds
		self.rb.add(obs, actions, rewards, dones, real_next_obs)

		self.next_obs = next_obs
		self.steps += 1
		return infos

	def get_actions(self, obs: np.ndarray) -> np.ndarray:
		'''
		Samples actions according to the epsilon-greedy policy using the linear schedule for epsilon.
		'''
		# SOLUTION
		self.epsilon = linear_schedule(self.steps, args.start_e, args.end_e, args.exploration_fraction, args.total_timesteps)
		actions = epsilon_greedy_policy(self.envs, self.q_network, self.rng, obs, self.epsilon)
		assert actions.shape == (len(self.envs.envs),)
		return actions


tests.test_agent(DQNAgent)
# %%
def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str, mode: str = "classic-control", video_log_freq: Optional[int] = None):
    """Return a function that returns an environment after setting up boilerplate."""

    if video_log_freq is None:
        video_log_freq = {"classic-control": 100, "atari": 30, "mujoco": 50}[mode]

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    f"videos/{run_name}",
                    episode_trigger = lambda x : x % video_log_freq == 0
                )

        if mode == "atari":
            env = prepare_atari_env(env)
        elif mode == "mujoco":
            env = prepare_mujoco_env(env)

        obs = env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
# %%
class DQNTrainer:

	def __init__(self, args: DQNArgs):
		self.args = args
		self.envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, self.run_name)])
		self.start_time = time.time()
		self.rng = np.random.default_rng(args.seed)

		# Get obs & action shapes (we assume we're dealing with a single discrete action)
		num_actions = self.envs.single_action_space.n
		action_shape = ()
		obs_shape = self.envs.single_observation_space.shape
		num_observations = np.array(obs_shape, dtype=int).prod()

		self.q_network = QNetwork(num_observations, num_actions).to(device)
		self.target_network = QNetwork(num_observations, num_actions).to(device)
		self.target_network.load_state_dict(self.q_network.state_dict())
		self.optimizer = t.optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

		self.rb = ReplayBuffer(len(self.envs.envs), obs_shape, action_shape, args.buffer_size, args.seed)
		self.agent = DQNAgent(self.envs, self.args, self.rb, self.q_network, self.target_network, self.rng)


	def add_to_replay_buffer(self, n: int):
		'''Makes n steps, adding to the replay buffer (and logging any results).'''
		# obs =  self.agent.envs.reset() # we need to get the next observation
		# # same things self.agent.envs.reset()
		# actions = self.agent.get_actions(obs)
		# next_obs, rewards, dones, infos = self.envs.step(actions)
		# for _ in range(1,n):
		# 	self.rb.add(next_obs, actions, rewards, dones, infos)
		# wandb.log({"steps": self.agent.steps, "epsilon": self.agent.epsilon})
		for steps in range(n):
			data = self.agent.play_step()
			wandb.log( {"logged_info": data }, steps = steps)
			# the steps is giving us weiweird errors
			


	def training_step(self) -> None:
		'''
		Samples once from the replay buffer, and takes a single training step.
		Theta_target is a previous copy of the paramters of the Q network
		'''
		data = self.rb.sample(self.args.batch_size, device=device)
		s, a, r, d, s_new = data.observations, data.actions, data.rewards, data.dones, data.next_observations
		with t.inference_mode():
			target_max = self.target_network(s_new).max(-1).values
			# this gies us the value of the best s(t+1) according to the target network
		predicted_q_vals = self.q_network(s) [range(self.args.batch_size), a.flatten()]
		# gets us the q-value of action at that row (batch) in total batches
		y = r + self.args.gamma * (1 - d.float().flatten())*self.target_network.forward(s_new)
		Q = self.q_network.forward(s)
		summed = np.sum((y-Q)**2, axis=0)
		loss = (1/(self.args.batch_size)) * summed

		self.agent.q_network.backward(loss)
		
	def train(self) -> None:

		run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
		if self.args.use_wandb: wandb.init(
			project=self.args.wandb_project_name,
			entity=self.args.wandb_entity,
			name=run_name,
			monitor_gym=self.args.capture_video
		)

		print("Adding to buffer...")
		self.add_to_replay_buffer(self.args.buffer_size)

		progress_bar = tqdm(range(self.args.total_training_steps))
		last_logged_time = time.time()

		for step in progress_bar:

			last_episode_len = self.add_to_replay_buffer(self.args.train_frequency)

			if (last_episode_len is not None) and (time.time() - last_logged_time > 1):
				progress_bar.set_description(f"Step = {self.agent.steps}, Episodic return = {last_episode_len}")
				last_logged_time = time.time()

			self.training_step()

		# Environments have to be closed before wandb.finish(), or else we get annoying errors 😠
		self.envs.close()
		if self.args.use_wandb:
			wandb.finish()