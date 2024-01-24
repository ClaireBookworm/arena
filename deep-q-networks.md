## Deep Q Networks

Debuggin RL is hard 

- feedback is poor
    - errors aren’t local → “doing the wrong math” - numerical errors manifest as all your metrics are going weird at the same time, loss exploding, KL div collapsing… but you don’t know where to start looking
    - perfomrnace is noisy - “hwo good it as a collecting reward” is weakly related to hwo good of an implementation you’ve written (unlucky seed? ex) and etc. - run to run variability is so high
- simplifying is hard
    - there’s few narrow interfaces - split system up into components normally, but each RL component consumes a large number of mega or gigabyte arays and returns the same; state can be thought of an interace with the own components past, and in RL this interface is huge
    - few black boxes - a good abstraction; but in RL you’re required to know how the env works, how the ntwork works, how optimizer works, how backprop works, multiprocessing, stat and loggin, gpus, …. There are [lots](https://docs.ray.io/en/latest/rllib.html) of [attempts](https://github.com/thu-ml/tianshou) at [writing](https://github.com/deepmind/acme) black-box [RL](https://github.com/astooke/rlpyt) libraries, but as of Jan 2021 my experience has been that these libraries have yet to be both flexible *and* easy-to-use.-
- writing RL systems
    - expectations suck - arrive in RL expecting a garbage fire
    - mid community
- debugging strats
    - design reliable tests - control with seed; substituting out envso r algos with simpler ones
    - design fast tests - dont run on full task
    - localise errors - test code that tells u where it is → binary search; tests that cut system in half
    - be bayuesian - reflect on which bits of code are more likely to ahve bugs…
    - pursue anomalies
- common fixes
    - handtune reawrds scale → targests aren’t going -1, +1, → have rewards that generate sensitibe targets or your network - hand-scale, hand-clip rewards
    - use really large batch size - small batches / complex envs → weird random sisues
    - small networks and avoid pixels
        - Gridworlds like [Griddly](https://github.com/Bam4d/Griddly) and [minigrid](https://github.com/maximecb/gym-minigrid). Gridworlds can support most of the interesting behaviours you'd find in a continuous environment, but are much more resource-efficient. If you've just graduated out of [the Gym envs](https://gym.openai.com/envs/#classic_control), gridworlds are an excellent next step.
        - Multi-agent setups like the boardgames from [OpenSpiel](https://openspiel.readthedocs.io/en/latest/games.html), [microRTS](https://github.com/santiontanon/microrts) or [Neural MMO](https://github.com/jsuarez5341/neural-mmo). A multi-agent env shouldn't be your *first* foray into RL - they're substantially more complex than the single-agent case - but competition and cooperation can generate a lot of complexity from very lightweight environments.
        - Unusual envs like [WordCraft](https://github.com/minqi/wordcraft). WordCraft is unique in that it isolates learning about the real world from actually having to model the real world! But again, possibly not the best choice for a first RL project; I've included it here as an example of how powerful simple environments can be.
- loss curves are a red herring
- use probe environments
    - classic control ones from gym
    - specific envs w/ one actions, 0 obs… etc. et.c
    - probe agents → cheat agents, automatons (don’t use nn at all, just handwritten to check if env is solvable), tabular (simple env → nothing works; replace with lookup table)
        1. **One action, zero observation, one timestep long, +1 reward every timestep**: This isolates the value network. If my agent can't learn that the value of the only observation it ever sees it 1, there's a problem with the value loss calculation or the optimizer.
        2. **One action, random +1/-1 observation, one timestep long, obs-dependent +1/-1 reward every time**: If my agent can learn the value in (1.) but not this one - meaning it can learn a constant reward but not a predictable one! - it must be that backpropagation through my network is broken.
        3. **One action, zero-then-one observation, *two* timesteps long, +1 reward at the end**: If my agent can learn the value in (2.) but not this one, it must be that my reward discounting is broken.
        4. **Two actions, zero observation, one timestep long, action-dependent +1/-1 reward**: The first env to exercise the policy! If my agent can't learn to pick the better action, there's something wrong with either my advantage calculations, my policy loss or my policy update. That's three things, but it's easy to work out by hand the expected values for each one and check that the values produced by your actual code line up with them.
        5. **Two actions, random +1/-1 observation, one timestep long, action-and-obs dependent +1/-1 reward**: Now we've got a dependence on both obs and action. The policy and value networks interact here, so there's a couple of things to verify: that the policy network learns to pick the right action in each of the two states, and that the value network learns that the value of each state is +1. If everything's worked up until now, then if - for example - the value network fails to learn here, it likely means your batching process is feeding the value network stale experience.
    - KL div → large vs. small meaningss
        - The KL div between the policy that was used to collect the experience in the batch, and the policy that your learner's just generated for the same batch. This should be small but positive.
            
            If it's very large then your agent is having to learn from experience that's very different to the current policy. In some algorithms - like those with a replay buffer - that's expected, and all that's important is the KL div is stable. In other algorithms (like PPO), a very large KL div is an indicator that the experience reaching your network is 'stale', and that'll slow down training.
            
            If it's very low then that suggests your network hasn't changed much in the time since the experience was generated, and you can probably get away with turning the learning rate up.
            
            If it's growing steadily over time, that means you're probably feeding the same experience from early on in training back into the network again and again. Check your buffering system.
            
            If it's negative - that shouldn't happen, and it means you're likely calculating the KL div incorrectly (probably by not handling invalid actions). 
            
- termianl correlation — correlation between value in final state and reward in final state (and penultimate)

[Debugging Reinforcement Learning Systems](https://andyljones.com/posts/rl-debugging.html)

****Interesting Resources (not required reading)****

- [An Outsider's Tour of Reinforcement Learning](http://www.argmin.net/2018/06/25/outsider-rl/) - comparison of RL techniques with the engineering discipline of control theory.
- [Towards Characterizing Divergence in Deep Q-Learning](https://arxiv.org/pdf/1903.08894.pdf) - analysis of what causes learning to diverge
- [Divergence in Deep Q-Learning: Tips and Tricks](https://amanhussain.com/post/divergence-deep-q-learning/) - includes some plots of average returns for comparison
- [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures) - 2017 bootcamp with video and slides. Good if you like videos.
- [DQN debugging using OpenAI gym Cartpole](https://adgefficiency.com/dqn-debugging/) - random dude's adventures in trying to get it to work.
- [CleanRL DQN](https://github.com/vwxyzjn/cleanrl) - single file implementations of RL algorithms. Your starter code today is based on this; try not to spoiler yourself by looking at the solutions too early!
- [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html) - 2018 article describing difficulties preventing industrial adoption of RL.
- [Deep Reinforcement Learning Works - Now What?](https://tesslerc.github.io/posts/drl_works_now_what/) - 2020 response to the previous article highlighting recent progress.
- [Seed RL](https://github.com/google-research/seed_rl) - example of distributed RL using Docker and GCP.

DQN overview

- there’s parts: setup, acting phase (agent acts in env), learning phase (trains itself to do better in the future)
- setting up
    - list that stores data on what agent has seen and done so far: **experience replay buffer**
        - in supervised learning we assume inputs are independent and identically distributed but not ture for a game → we learn from results from actions in a way that’s independent from the action that’s taken with this
        - when we improve our policy when training, we take a batch of x random actions and use this → agent does this traniing every few steps. → effiecient (learn from an experience over and over; we. can only guarantee Q-function converging if we repeatedly sample all actions in all states), and makes training inputs independnet (doesn’t make them identically distributed, it widens the set of game states we can draw upon at any given moment — better than just learning from previous state)
    - a nn takes in an observation and outputs an expected total reward for each possible action it can take; used in acting phase
    - copy of nn that only updates very C (c=1000) timesteps → used in learning phase to improve DQN’s stability, called a **target network**
- acting phase
    - pass **obs** at time t as input to the nn, outputs expected future reward for each possible a
    - select an a → random action with prob epsilon; select a with highest expected future reward the rest of the time (*epsilon-greedy* exploration strat)
    - get reward and obs at time t+1
    - store obs at time t, the a, the reward, whether the episode temrinated, and the obs at t+1 in the **experience replay buffer** for use in the learning pahse later
- learning phase
    - sample random batch N items (n=32) from replay buffer
    - set eahc learning target to curent reward + target network’s estimate of future reward given obs at t+1 — since target network knows what happened at t+1 and Q doesn’t, it should have a more accurate prediction that can be used to train the Q-network
    - calc diff between learning targets and Q-networks’ estimate given t, **loss function**
    - perform **gradient descent** on the q-network in order to minimize this loss function
	- every C steps ( C=1000) update the target network with the Q-network's paramters 

we're attempting to minimize prediction error rather than max future reward -> if we always know what reward you'll get from any action in a game, then the optimal action is trivial (with prob 1, take whatever action max futur reward) -> selection an action that gives u most future reward given current knowledge is trivial, but knowing what actions will produce **what** rewards is the hard part 

Q network -> all about **minimizing prediction error** -- if a move is better than Q-network predicted, loss gets worse -> update in favor of assigning higher expected reoward to taht move given that state V(s,a) in the future. Loss function increasing is _punishing_ the agent for finding a good move (giving feedback to improve itself via gradient descent)

Formula: $$Q8(s,a) = \max_{\pi} \mathbb E [r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots | s_t = s, a_t=a, \pi]$$ where pi is policy, gamma is discount rate, rt is the reward at timestep t. Q*(s,a) is the optimal Q-function, and Q(s,a) is the current Q-unction -> we want to convert to Q* 

> why is gamma also something related to punishment?? (something about -1 => makes sure it doesn't just jump off the cliff in the clif wakling gym)

policy -> in DQN the optimal q-function is = policy that produces the best possible sum of rewards given (s,a) pair -> given what just happened; what rule could we follow to get the most total reward psosible from now on? 

$$L_i(\theta _i) = \mathbb E _{(s, a, r, s') ~ U(D)} [ ( r = \gamma \max _{a'} Q(s', a', \theta_i^-) - Q(s, a, \theta_i))^2]$$ 

![Screenshot 2024-01-23 at 2.37.19 PM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/f2f3e1b3-4f39-4187-b201-2e19ac213bf9/a0c6d4aa-e40f-4a6d-9cb4-5a8aec0f3b93/Screenshot_2024-01-23_at_2.37.19_PM.png)

- red → sample random batch of N items
- blue → set each learning target to current reward (from replay buffer) + target networks’ estimate of the future reward given t+1 → sinte target network knows what happened in t+1 and Q doesn’t, should have more accurate prediction that can be used to train Qnetwork
- green → calc q-networks estimate of future reward, givein t
- take mean sqaured error between blue and green → loss function

```python
Initialize experience replay buffer D, Q-network Q, and target Q-network q. 
Set q’s parameters to Q.

For each step:
	With probability ε, select a random action. Otherwise, pick the action with the highest expected future reward according to Q(s, a).
	Execute action a, and collect reward r and observation t+1 from the environment.
	Store the transition obs(t), action, reward, obs(t+1) in D.

If current step is a training step:
	Collect a random batch of transitions from D.
	For each transition j, set future reward to r(j) if the episode ended at j, else r(j) + gamma * q(s’, a’).
	Perform gradient descent with respect to the loss function to update Q.
	Every C steps, set q’s parameters to Q.
```

### Cartpole 
The classic env: `CartPole-v1` is simple - literally a game where u control a pole and go left, right, r, q. Differnently, the obersation is contunous -- your agent has no idea that these observations represent positions or velocities and no idea what physics is. 

overview:
1. implement q network that maps a state to estimated value for each action
2. implement a replay buffer to store experiences e_t = (st, at, rt1, st1)
3. implement policy which chooses actions based on q-network + espilon-greedy randomness to encourage exploration
4. piece everything together 


We use DQNs because the input is going to be continuous, so we can't just us the lookup table we've been doing. 

CartPole-v1 gives +1 reward on every timestep. Why would the network not just learn the constant +1 function regardless of observation?
The network is learning Q-values (the sum of all future expected discounted rewards from this state/action pair), not rewards. Correspondingly, once the agent has learned a good policy, the Q-value associated with state action pair (pole is slightly left of vertical, move cart left) should be large, as we would expect a long episode (and correspondingly lots of reward) by taking actions to help to balance the pole. Pairs like (cart near right boundary, move cart right) cause the episode to terminate, and as such the network will learn low Q-values.

^ this means it incentivises it to stay for longer and not termiate 

iid - independently and identically distributed random variables 
extend experiences to et = (ot, at, rt1, st1, dt1) - dt1 is a bool indicating that st1 is a terminal observation

**correlated states**
- because DQN uses nn to learn q-values, many sa pairs are aggregatd together (unlike tabular; where Q-learning learns independently the value of each s-a pair)
- states within an episode are highly correlated and not iid at all -> a few bad moves at the start might doom rest of the game (in chess) 
- trainig mostly on an episode where agent opened the game poorly might disincentivize good moves to recover -> bad Q-value estimates 

**uniform sampling**
- pick a buffer size; store experious and uniformly sample out of the buffer
- if we want policy to play well in all sort of states -> sampled batch has a representative sample of all diverse scenarios that can happe nin env
- implies large batch sizes; capacity of reply buffer is another hyperparameter -> too small it'll be bad; big cost

**implement `replaybuffer`**

### Montezuma's Revenge
Difficult, and there's a lot of either too long of a delay between getting the key and reaching the door + reward shaping/hacking (bad proxy)

advanced exploration
- better if we didn't human hardcore auxillary rewards that lead to reward hacking -> rely on other signals
- a state which is "surprizing" or "novel" -- make it innately curious 
- random network distillation -> measuring the agent's ability ot predict the output of a n on visited states; states that are hard to predict are poorly explored and thus highly rewarded
- first return, then explore -> even better -> reward shaping can be gamed (leading ot noisy TV? - agent seek novely become entranced by randomness and ignore everything else)

epsilon-greedy policy
- defined by the q-network - take action with max predicted award
- bias towards optimism
- maximum of a set of values v1... vn using the max of some noisy estimates v1 hat ... vn hat, we get unlucky get very large positive noise 

**probe environments**
- way of figuring out how well our model is doing + debugging
- step always returns the same thing 
- obs and reward are always the same 
- learn that value of constant obs [0.0] is +1 

action space -> `gym.spaces.Box` -> dealing wit hreal-valued quantities (cont not discrete) -- first two args of box are `low` and `high` - define box in $$\mathbb{R}^n$$. if arrays are (0,0, and (1,1)) makes 0<= x, y <= 1 in 2d space 

```python
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
```

Probe 2
- one action, observation of [-1.0], [+1.0]; one timsetp long, reward = obv
- doesn't matter what env we're in, we just inject bs info we know 
- `self.reward = 1.0 if self.np_random.rand() < 0.5 else 1.0`  -> setting the observation based on some random value  (reward)

```python
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
```
Probe 3
- one action 
- [0.0] to [1.0], two timesteps, +1 reward at end 
- expect agent to rapidly learn the discounted value of init observation 


```python
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
```