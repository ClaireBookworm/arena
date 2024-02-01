# General Training Notes 

(esp from optimizer unit I missed)

Wandb is a tool for tracking and visualizing ML experiments. 
`wandb.log({"steps": self.agent.steps, "epsilon":self.agent.epsilon})` logs the # of steps taken by the agent and current value of epsilon (used in epsilon-greedy exploration strats). We can monitor these values as the training progresses. 


## optimizer review
zero_grad -> when you call `backward()` on a tensor to compute gradients. the gradeints are accumulated into the `.grad` attribute of the tensor's associated varialbe -> each time you call `backward()` the new gradients are added to existing gradients rather than replacing them

To prevent this -> zero_grad, which clears the existing gradients at teh start of each new optimization step 

In reinforcement learning, the choice of optimizer can significantly impact the learning dynamics and final performance. Here are some commonly used optimizers:

- `Stochastic Gradient Descent` (SGD): This is the most basic optimizer. It updates the parameters in the negative direction of the gradient. It has a single learning rate as a hyperparameter.

- `Momentum`: This is a variant of SGD that takes into account the previous gradients to accelerate SGD in the relevant direction and dampens oscillations. It's almost always better than vanilla SGD.

- `RMSprop`: This optimizer uses a moving average of squared gradients to normalize the gradient itself. That has an effect of balancing the step size—decrease the step for large gradient to avoid exploding, and increase the step for small gradient to avoid vanishing.

- `Adam` (Adaptive Moment Estimation): This is currently the most popular optimizer for deep learning applications. It combines the ideas of Momentum and RMSprop: it calculates an exponential moving average of the gradient and the squared gradient, and the parameters beta1 and beta2 control the decay rates of these moving averages.

- `AdamW`: This is a variant of Adam that decouples the weight decay from the optimization steps. This can sometimes improve performance.

- `A3C` (Asynchronous Advantage Actor-Critic): This is not an optimizer per se, but a reinforcement learning algorithm that uses an optimizer (like RMSprop) in a specific way, by having multiple worker agents independently updating the same model parameters.

Remember that the choice of optimizer can depend on the specific problem and model architecture, and it's often a good idea to experiment with different optimizers to see which works best for your specific use case.