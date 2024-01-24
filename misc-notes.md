# General Training Notes 

(esp from optimizer unit I missed)

Wandb is a tool for tracking and visualizing ML experiments. 
`wandb.log({"steps": self.agent.steps, "epsilon":self.agent.epsilon})` logs the # of steps taken by the agent and current value of epsilon (used in epsilon-greedy exploration strats). We can monitor these values as the training progresses. 

zero_grad -> when you call `backward()` on a tensor to compute gradients. the gradeints are accumulated into the `.grad` attribute of the tensor's associated varialbe -> each time you call `backward()` the new gradients are added to existing gradients rather than replacing them

To prevent this -> zero_grad, which clears the existing gradients at teh start of each new optimization step 