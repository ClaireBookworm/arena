# Ray Tracing

```python
def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    rays = t.zeros ( (num_pixels, 2, 3), dtype = t.float32 )
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1]) # start, end, steps, out = the output tensor
    # print(rays[:, 1, 0]) # the x axis
    # print(rays[:, 1, 1]) # the y axis
    rays[:, 1, 0] = 1
    # spacing = (2 * y_limit) / num_pixels
    # rays = []
    # for space in range(num_pixels):
    #   rays.append( np.array([[0,0,0], [1, space * spacing - y_limit, 0]]))
    return rays

rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)
```

![Screenshot 2024-01-08 at 2.29.59 PM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/f2f3e1b3-4f39-4187-b201-2e19ac213bf9/abcca534-5461-477e-8dbf-325984bc3b63/Screenshot_2024-01-08_at_2.29.59_PM.png)

line segment defined by L1 and L2. Camera ray is defined by origin O and direction D. for given ray see if they intersect — equations for all points on camera ray as R(u) = O + uD 

einops, n_rays and n_segs

# CNNs and ResNets

**from this article**

[A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

(elementwise) Hadamard Product — naive matrix multiplication where is literally just multiplying the values of two matrices → aij x bij. by definition what makes a convolution, so what is always used.

stride length = 1 (non-strided) shifts one each time 

Valid padding / Same apdding → adding a padding of 0’s around so the convolution can go to the edges and we don’t lose any data, and the resulting matrix after applying the kernel (i..e. 3x3x1) on it gives us something of the same size

Valid padding → same operations without padding, then the matrix result is the matrix that has dimensions of the kernel

[GitHub - vdumoulin/conv_arithmetic: A technical report on convolution arithmetic in the context of deep learning](https://github.com/vdumoulin/conv_arithmetic?source=post_page-----3bd2b1164a53--------------------------------)

similar to convolutional layer, the pooling layer is responsible for reducing the spatial size of convolved feature → decrease the computationl power required to process the data

max pooling → max value from the portion of the image covered by the kernel → noise suppressant, discards noisy activations altogether and also performs de-noising along with dimensionality reduction (empirically). better than avg. 

average pooling → average of all values from portion of image covered by kernel; dimeionsal reduction 

******************************************fully connected layer****************************************** → usually cheap way of learning non-linear combinations of high level features as represented by output of conv layer. C layer is learning a possibly non-linear function in that space 

converted input image → ********************************************multi level perceptron******************************************** (MLP), flatten the image into a column vector; series of epochs the model is able to distinguish between dominating and certain low level features in images & classify using softmax classification technique 

various architectures

[GitHub - ss-is-master-chief/MNIST-Digit.Recognizer-CNNs: Implementation of CNN to recognize hand written digits (MNIST) running for 10 epochs. Accuracy: 98.99%](https://github.com/ss-is-master-chief/MNIST-Digit.Recognizer-CNNs?source=post_page-----3bd2b1164a53--------------------------------)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f2f3e1b3-4f39-4187-b201-2e19ac213bf9/171aff48-2a3f-473b-80b4-26e7be1e39ef/Untitled.png)

******************************************************batch normalization in cnns******************************************************

[Batch Normalization in Convolutional Neural Networks | Baeldung on Computer Science](https://www.baeldung.com/cs/batch-normalization-cnn)

normalization → pre-professing technique used to standardize data; not normalizing casues problems in network making it harder to train. the most straightforward method is to scale it to a range from 0-1, or forcing the data to have a mena of 0 and stdev of 1 (x nom = x-m/s). x is data point to normalize; m is mean, s stdev. 

********************batch norm******************** is a norm technique done between the layers of a neural network instead of in the raw data. done along mini-batches instead of full data. $z_n = {z-m_z} / s_z$. m is mean of neurons output, s is stdev of neurons output. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f2f3e1b3-4f39-4187-b201-2e19ac213bf9/7076872e-491d-4e7f-9c72-cb73d36f5977/Untitled.png)

$x_i$ are the inputs; $z$ is the output of the neruons, $a$ is the output of activation funcs, and $y$ is network output

batch norm (the red lines); a neuron without batch norm would be computed like $z = g(w, x) + b$ , $a = f(z)$. where g() is the linear transformation of the neuron, w is the weights, b is the bias, and f() is the activation function. model learns the paramters w and b. 

adding batch norm → $z = g(w, x)$, $z^N= z-m_z / s_z \cdot \gamma + \beta$, $a = f(z^N)$. z^N is the output of batch norm, m_z is the mean of neuron output ,s_z stddev, and gamma (stdev)/beta (mean) are the learning parameters, which shift over time and is learned over opehs and other learning parameters. the bias is missing because when we subtract mean any constant over values of z can be ignored 

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization

model = Sequential([
    Dense(16, input_shape=(1,5), activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(2, activation='softmax')
])
```

why does this work

> Firstly, we can see how normalizing the inputs to take on a similar range of values can speed up learning. One simple intuition is that Batch Norm is doing a similar thing with the values in the layers of the network, not only in the inputs.
> 

> Secondly, in their original paper [Sergey et al.](https://arxiv.org/pdf/1502.03167.pdf) claim that **Batch Norm reduces the internal covariate shift of the network**. The covariate shift is a change in data distribution.
> 

lastly batch norm has a regularization effect; because over mini batches and not entire, the models data distro sees each time has some noise. applying batch means mean and stdev of layer inputs will always remain the same. 

in **CNNs**, they do the same thing — in convolutions we have shared filters that go along the featurem aps of the input, and these filters are the sameo n every feature map. so we can normalize the output in the same way. 

the parameters used to normalize are calculated along with each entire feature map. (in regular batch norm, each feature would have a different mean and stdev), but here each feature map has a single mean/stdev used on all features

*to basically instead of a custom beta and gamma for each neuron / feature, we have one for a whole channel (a feature map), which would be like a group of related features (i.e. red is channel out of rgb, or like the concept of spots on an animal)* 

you can see batch norm like a layer, it’s acting on an activation (which would be like the values that go in before the activation function layer, so the output from a conv layer) or like what comes out… some intermediate 

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D

model = Sequential([
  Conv2D(32, (3,3), input_shape=(28, 28, 3) activation='relu'), 
  BatchNormalization(),
  Conv2D(32, (3,3), activation='relu'), 
  BatchNormalization(),
  MaxPooling2D(),
  Dense(2, activation='softmax')
])
```

feature map — the output of a conv layer representing specific features in the input image or feature map. the output activations for a given layer 

feature — anything, edges corners shapes etc. 

activation — an intermediate value in a layer when it is processing 

****************************************************************************************deep residual learning for image recognition****************************************************************************************

[](https://arxiv.org/pdf/1512.03385.pdf)

******************************exercises****************************** 

subclassing nn.Module helps give us helper functions, like access to Linear, Conv2d, and Softmax. (adding nn. before). 

within this function, we have __init__ which gives us the weights and biases initially. then forward 

implementing relu: 

```python
class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.max(x, t.zeros_like(x))
```

basically, implementing the function f(x) = max(0, x). 

`linear` module which applies a simple transformation. initialization is really important and bad initialization is going to make it hard to converge or train well. each float in the weight 

nn.Parameter — a tensor, helps you be able to do layer.parameter to display them more easily, and shows us the tensors only that are parameters. related to automatic gradient tracking. model.paramters that goes through all the parameters from this module and sub-modules; wrapping in the classes can help. 

einsum is helpful — https://ajcr.net/Basic-guide-to-einsum/

```python
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential( 
            Flatten(start_dim = -2, end_dim = -1),
            Linear(in_features = 28*28, out_features = 100),
            ReLU(),
            Linear(in_features = 100, out_features = 10)
        )
        

    def forward(self, x: t.Tensor) -> t.Tensor:
        # x = self.flatten(x) # __call__ = forward
        # x = self.linear(x)
        # x = self.relu(x)
        # x = self.linear2(x)
        return self.net(x)

tests.test_mlp(SimpleMLP)
```

TRAINING 

```python
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]) # transform mnist data by composing ToTensor to convert PIL to a pytorch tensor
# normalize; takes arguments for mean and stdev and performs the linear transformation x -> (x-mean)/std

def get_mnist(subset: int = 1):
    '''Returns MNIST training data, sampled by the frequency given in `subset`.'''
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=MNIST_TRANSFORM)
		# ^ define dataset with torchvision.datasets 
		# root = "./data" indicates that we're storing out data in the ./data library; and transform tells us that we should apply our previously defined transform to each element in dataset 

    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
        mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))
		# allows us to take a subset of a dataset; indices is a list of indices to stample from dataset 
    return mnist_trainset, mnist_testset

# DataLoader is a useful abstraction to work with a dataset -- takes in a dataset and a few arguments 
# batch_size - how many inputs to feet through the model on which to compute the loss before each step of gradient descent
# shuffle - whether to randomizse the order each time you iterate 
mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)
```

batch_sizes can be powers of two which can be helpful for GPU utilization 

we normalize the data means we can avoid being in a very flat area of the domain and having gradient descent taking a long time to converge 

shuffle = True → done during training to make sure we aren’t exposing our model to the same cycle or order of data in every epoch; it is done to ensure it doesn’t adapt to any spurious pattern 

`tqdm` → wraps around a list, range, or other iterable but doesn’t affect the strcture of your loop (`for i in tqdm(range(100)): time sleep(0.01)` )

`device` → move this onto a GPU (using stuff like `model = model.to(device)` or `model.to(device)` or `new_zeros` or `new_full`.

```python
model = SimpleMLP().to(device)

batch_size = 64
epochs = 3

mnist_trainset, _ = get_mnist(subset = 10)
mnist_trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
loss_list = []

for epoch in tqdm(range(epochs)):
    for imgs, labels in mnist_trainloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())   

line(
    loss_list, 
    yaxis_range=[0, max(loss_list) + 0.1],
    labels={"x": "Num batches seen", "y": "Cross entropy loss"}, 
    title="SimpleMLP training on MNIST",
    width=700
)
```

batch size is the number of samples in each batch (i.e. the number of samples we feed into the model at once). While training our model, we differentiate with respect to the average loss over all samples in the batch (so a smaller batch usually means the loss is more noisy)

residual networks

resnets - learn resiudal functions with reference to the layer inputs, instead of learning unreferenced functions. instead of hoping each few stacked layers directly fit a desired mapping, residual nets let these layers fit a residual mapping 

is ************************learning better networks as easy as stacking more layers?************************ — vanishing gradients, hamper convergence when you stack a lot. largely addressed by normalized initialization and intermediate normalization layers, which allow networks with tens of layers to start convering for stochastic gradeitn descent (SGD) 

*****************degradation***************** problem — with network depth increasing ,accuracy gets saturated (approached 100% on training set) and then degrades rapidly; not from overfitting and adding more layers leads to higher training error 

not all systems are easy to optimize → ***********************deep residual learning*********************** framework — instead of hoping each few stacked layers directly fit a desired underlying mapping, we let these layers fit a residual mapping. 

> We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping. To the extreme, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers.
> 

`inference_mode` — enable or disable gradietns locally 

# Optimization

<aside>
💡 exploring various optimization algorithmsn — stochastic gradient descent (SGD), RMSprop, and Adam and learn ow to implement them using code. looking at loss landscapes and their significance in visualizing the challenges faced during optimization

</aside>

**videos**: [gradient descent with momentum](https://www.youtube.com/watch?v=k8fTYJPd3_I), [rmsprop](https://www.youtube.com/watch?v=_e-LFe_igno), [adam](https://www.youtube.com/watch?v=JXQT_vxqwIs&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=23). 

gradient descent with momentum - computing an exponentially weighted average 

gradient descent sometimes be inefficient and take too many steps that dont necessarily move towards a low loss function and we want to prevent oscillations; so therefore we must use a learning rate htat isn’t that large — depends on the “shape of learnhing?” 

momentum: on iteration t; compute dW, db on current mini - batch (or mini batch being entire training set) 

V_dw = /beta V_d + (1-beta) dW // V_db = B Vdb + (1-beta)db → basically what this does is smooth out the movements of the gradient descent

the momentum term is computed as a moving average of the past gradients, and the weight of the past gradients is controlled by a hyperparameters called Beta → it helps accelerate optimization by allowing the updates to build up in the direction of the steepest descent. the steps in gradient descent aren’t all independent

Algorithm: 

Compute $dW, db$ on the current mini-batch. $v_{dW} = \beta v_{dW} + (1-\beta) dW$, $v_{db} = \beta v_{db} + (1-\beta)db$, and $W = W - \alpha v_{dW}$, $b = b-\alpha v_{db}$. The hyperparameters here are alpha and beta, beta = 0.9 (average over last 10 gradients) . (people don’t bother with bias correction)  (*********************************************sometimes they get rid of 1-beta, which means v_dW is scaled by 1-beta, and there’s a bit different way of representing alpha*********************************************) 

problems with gradient descent

- local minima
    - can get stuck in a local minima, points that are not the global minimum of the cost function but are still lower; can occure when cost function has multiple valleys and the algorithm gets stuck
- saddle points
    - where one dimension has a higher value than the surrounding points and the other has a lower value. can get stuck because the gradients in one direction is lower; other is higher
- plateuas — region in the cost function where the gradients are very small or close to zero; cause gradient to take a long time or not converge
- oscillation — when the learning rate is too high, causing the algorithm to overshoot the minimum and oscillate back and forth
- slow convergence — gradient descent can converge very slowly when teh cost function is complet or has many local minima; this means alg takes time to find global minimum

**RMSprop** - root mean squared prop

- uses the same concept of the exponentially weighted average of gradient as gradient descent with momentum but the difference is parameter update.
- on iteration t, compute dW and db on count mini-batch
- there’s a squaring o dW that keeps an exponentially weighted average of the squares of the derivatives
- $S_{dW} = \beta S_{dW} + (1-\beta) dW^2$ → sdW is relatively small, so we divide dW by a relatively small number while Sdb is larger, so we divide that by a comparatively larger number to slow down the changes in the vertical dimension
- $S_{db} = \beta S_{db} + (1-\beta) db^2$
- $W = W - \alpha \frac{dW}{\sqrt {S_{dW}}}$ and $b = b - \alpha \frac{db}{\sqrt{S_db}}$; can use a epsilon so you don’t divide by zero somewhere
- in this case, you might have a large bias and less weight → sloped in a W or b direction → in reality would have w1, w2, w3… many many dimensions
- choose beta (momentum) — must be higher to smooth out the update because we give more weight to the past gradients. you can use default beta = 0.9, but can be tuned between 0.8 and 0.999
- momentum takes into account past gradeints so as to smooth down gradient measures; it can be implemented with descent by batch gradient

adam optimization algorithm 

- combines rmsprop and grad. with momentum.
- start with V_dw = 0, S_dW = 0, V_db = 0, S_db = 0
- on iteration t, compute dW, db using current mini-batch (mini-batch gradient descent)
- use $V_{dW} = \beta_1 v_{dW} + (1-\beta_1) dW$ , $v_{db} = \beta_1 v_{db} + (1-\beta_1)db$
- $S_{dW} = \beta-2 S_{dw} + (1-\beta_2) dW^2$, and $S_{db} = \beta_2 S_{db} + (1-\beta_2)db$ rms prop
- also do bias correction → Vcorrected dW = $v_{dw} / (1-\beta_1^t$, $V_{db}^{\text{corrected}} = v_{db} / (1-\beta_1^t)$ and so it goes
- $W = W - \alpha {v_{dW}^{\text{corrected}}} / {\sqrt{S_dW^{\text{ corrected}}} + \epsilon}$ and same for b
- alpha - needs to be tuned → try a range of values to see what works
- beta1: 0.9 is the common choice, dW
- beta2: 0.99, computing the dW^2 moving average
- epsilon: 10^-8 recommended, but not super important or affect performance
- adam == ***************************adaptive moment estimation***************************

[Gradient Descent With Momentum from Scratch - MachineLearningMastery.com](https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/)

- ********************************gradient descent******************************** - optimization algorithm that follows the negative gradient of an objective function in to locate the minimum of a function. it can bouce around the search space and get stuck — technically a first-order optimization algorithm as it uses the first-roder derivative of the target objective function
- momentum is extension that allows the search to build inertia in a direction in the search space and overcome the oscillations of noisy gradients and coast across flat spots of the search space
- if the target function takes multiple input vars, its multivariate; inputs are a vector. derivatvie is a gradient → first order derivative for a multivar objective function
- downhill movement is made by first calculating how far to move in the input space, calced as the step stize (**alpha** or learning rate) multiplied by the gradient. this is then subtracted from the current point ensuring we move against the gradient (x = x-step_size * f’(x))
- step size (alpha): hyperparameter that controls hwo far to move in the search space against the gradeint each iteration of the algo, called the learning rate. if the step size is too small, the movement in the space will be small and the search will take too long. if too big, it might bounce around and skip over the optima
- momentum — adding a hyperparameter that controls the amount of history (momentum) to include in the update equation (the step to a new point). the value is defined in the range 0-1 and usually 0.8, 0.9, or 0.99. a momentum of 0 is the same as grad descent without momentum
    - change_x = step_size * f’(x) → change in parameters is calc as the gradient scaled by step size.
    - thinking of updates over time, the update on time t will add the change used at the previous time weighted by the momentum hyperparameter: change_x(t) = step_size * f’(x(t-1)) + momentum * change_x(t-1)
    - so the update os x(t) = x(t-1) - change_x(t). the change in position accumulates magnitude and direction of changes over the iterations of the search, proportional to the size of the momentum hyperparameter. a large momentum (0.9) means the update is strongly influenced by the previous update, while lower (0.2) would mean very little influence. ************************the momentum algorithm accumulates an exponentially decaying moving average of past gradients and continues to move in their direction************************
    - momentum has the ffect of dampening down the change in the graduent and, in turn, the step size with each new point in the search space — increase speed when the cost surface is highly nonspehrical because it damps the size of the steps along direcitons of high curvature → larger effecitve learning rate
    - most useful in optimization problems where objective function has large amount of curvature (changes al ot) meaning the grad may change al ot over relatively small regions of the search space

example: 

```python
# plot of simple function
from numpy import arange
from matplotlib import pyplot

# objective function
def objective(x):
	return x**2.0

# define range for input
r_min, r_max = -1.0, 1.0
# sample input range uniformly at 0.1 increments
inputs = arange(r_min, r_max+0.1, 0.1)
# compute targets
results = objective(inputs)
# create a line plot of input vs result
pyplot.plot(inputs, results)
# show the plot
pyplot.show()
```

this defines a simple objective function and plots ap rabola 

```python
# derivative of objective function
def derivative(x):
 return x * 2.0

# gradient descent algorithm
def gradient_descent(objective, derivative, bounds, n_iter, step_size):
 # generate an initial point
 solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
 # run the gradient descent
 for i in range(n_iter):
 # calculate gradient
 gradient = derivative(solution)
 # take a step
 solution = solution - step_size * gradient
 # evaluate candidate point
 solution_eval = objective(solution)
 # report progress
 print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
 return [solution, solution_eval]
```

here we start with a randomly selected point in search space, calculating the gradient, updating the position in the search space, eval the new position, and reporting progress

**exercises and notes from streamlit**

> A loss function can be any differentiable function such that we prefer a lower value. To apply gradient descent, we start by initializing the parameters to random values (the details of this are subtle), and then repeatedly compute the gradient of the loss with respect to the model parameters.
> 
- decrease loss, so subtract gradient in the opposite direction. taking infinitesimal step is bad, so have some learning rate $\lambda$ and scale step by that amount to optain the update rule: $\theta_t \left_arow \theta_{t-1} - \lambda$

********stochastic gradient descent********

- sgd and gd are used loosley but there are three variations
- batch gradient descent — loss function is the loss over the entire dataset. requires too much computation unless the dataset is small
- sgd - stochastic gd- loss function is the loss on a randomly selected example. any particular loss may be completely in the wrong direction of the loss on the entire dataset, but in expectation it’s in the right direction. nice properties but dosn’t parallelize well, rarely used in deep learning
- mini-batch gd - loss function is the loss on a batch of exampels of size batch_size; standard
- `torch.optim.SGD` can be used for any of these by varying the # of examples passed in

batch_size

- larger size → estimate of the gradient is closer to that of the true gradient over the entire dataset, but requires more compute
- each element of batch can be computed in parallel → fill up all ur gpu memory. you can increase batch size without increasing wall-clock time
- a batch size that is too large generalizes too poorly in many scenarios
- scheduling → most commonly you’ll see batch sizes increase over course of trainnig; rouch estimate of the proper direction is good enough early in training but dont bounce around too much later in training. most sizes are multiples of 32, when using CUDA threads are grouped into warps o 32

weight decay — each iteration we also shrink each parameter slighty to 0 

zero_grad(), the gradients would accumulate across all minibatches, leading to incorrect gradient computations and ultimately poor model performance. It is essential to zero out the gradients before computing the gradients for each minibatch **to ensure accurate and timely parameter updates**.

********************************parameter gropus********************************

instead of passing a single iterable of params into an optimizer you can pass a list of parameter gropus, each one with different hyperparameters, 

```python
optim.SGD([
	{ 'params': model.base.parameters() },
	{ 'params': model.classifier.parameters(), 'lr': 1e-3}
], lr=1e-2, momentum=0.9)
```

the first argumnet here is a list of dicts, with each dict defining a separate parameter gropu → `params` key, which is an iterable of parameters belonging to this gropu. dicts might also have keyword arguments and if params is not specified, pytorch uses the value passed as a keyword argument 

```python
optim.SGD([
    {'params': model.base.parameters(), 'lr': 1e-2, 'momentum': 0.9},
    {'params': model.classifier.parameters(), 'lr': 1e-3, 'momentum': 0.9}
])
```

stors all their params and hyperparams into the `param_groups` attriburte 

use when:

- fine turning a model by freezing earlier layers and only trainnig later layers is an extreme form of param grouping — use param group syntax to apply a modified form, where the earlier layers have a smaller learning rate and allows the earlier layers to adapt to the specifics of the propblem and making sure it doesn’t lose the useful features already learned
- treat weights and biases differently; effects of weight decay are often applied to weights but not biases; pytorch doesn’t differentiate between these two
- transformers —weight decay is not applied to embeddings and layernorms in the transformer models

[Define sweep configuration | Weights & Biases Documentation](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)

# Backpropagation

we have graphs in which notes can affect each other → you have forward-mode and reverse-mode differentiation 

Forward-mode differentiation is when you move forward in the graph and find the derivatie on the computationa graph (if we change a, how does c change with respect to a?) in terms of the parent node (X → Y → Z; find $\partial / \partial X$). 

Reverse-mode differentiation: does the opposite, finds $\partial Z / \partial$. Good beause it does give us the derivative for every node. 

Example: $d = a \times b$ and we have L. Then we can get dL/da = dL/dd * dd/da = dL / dd * b → forward function $(a,b) \rightarrow a \cdot b$ and also take the backwards function which tells us how to compute the gradient wrt this arugmnet using only known quantitues as inputs

```python
def multiply_back(grad_out, out, a, b):
    '''
    Inputs:
        grad_out = dL/d(out)
        out = a * b

    Returns:
        dL/da
    '''
    return grad_out * b
```

where `grad_out` is the gradient of the loss with respect to the output of the function (i.e. dL/dd), `out` is the output of the function (i.e. d) and a and b are the inputs. 

Topological sorting → sort all the nodes and then do computations in that order. We can’t find dL/da without dL/dd, so we use that order to know. 

```python
def forward_and_back(a: Arr, b: Arr, c: Arr) -> Tuple[Arr, Arr, Arr]:
    '''
    Calculates the output of the computational graph above (g), then backpropogates the gradients and returns dg/da, dg/db, and dg/dc
    '''
    d = a * b
		e = log (c) 
		f = d * e
		g = log(f)
		# grad_out, dg/dg = 1
		first_log = log_back(np.array[1.0], np.log(f), f) # dg/df
		mult_d = multiply_back1(first_log, f, d, e) # dg/dd
		dg_da = multiply_back1(second_mult, d, a, b)
		dg_db = multiply_back0(second_mult, d, a, b)
		mult_e = multiply_back0(first_log, f, d, e)  #dg/de
		dg_dc = log_back(mult_e, np.log(e), e)
		return [dg_da, dg_db, dg_dc] 

if MAIN:
    tests.test_forward_and_back(forward_and_back)
```

Certainly! Let's consider a simple mathematical example to illustrate how derivatives work with respect to matrices in the context of neural networks.

### Example: Derivative of Loss with Respect to a Weight Matrix

Suppose we have a simple neural network layer that performs a linear transformation on its input. Let's denote:

- \( X \) as the input matrix to the layer.
- \( W \) as the weight matrix of the layer.
- \( b \) as the bias vector.
- \( Y \) as the output of the layer.

The transformation performed by the layer can be represented as:
\[ Y = XW + b \]

Now, let's assume we have a loss function \( L \) that measures how far off our predictions are from the true values. The goal is to find how changes in \( W \) affect \( L \).

### Computing the Gradient

The gradient of \( L \) with respect to \( W \) is denoted as \( \frac{\partial L}{\partial W} \). It is a matrix of the same size as \( W \), where each element is the partial derivative of \( L \) with respect to the corresponding element in \( W \).

Let's consider a simple case where \( L \) is defined as the mean squared error (MSE) between the predicted output \( Y \) and the true output \( T \):
\[ L = \frac{1}{n} \sum (Y - T)^2 \]

To find \( \frac{\partial L}{\partial W} \), we apply the chain rule:
\[ \frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial W} \]

1. **Computing \( \frac{\partial L}{\partial Y} \)**:
Since \( L = \frac{1}{n} \sum (Y - T)^2 \), we get
\[ \frac{\partial L}{\partial Y} = \frac{2}{n} (Y - T) \]
2. **Computing \( \frac{\partial Y}{\partial W} \)**:
Since $Y = XW + b$, we get
$\frac{\partial Y}{\partial W} = X$

Therefore, the gradient of the loss with respect to the weights is:
\[ \frac{\partial L}{\partial W} = \frac{2}{n} (Y - T) \cdot X \]

### Practical Implication

In a neural network, during backpropagation, this gradient tells us how to update each element of \( W \) to reduce the loss. The update rule typically looks like:
$W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W}$ where $\eta$ is the learning rate.

This example demonstrates how the derivative of a loss function with respect to a weight matrix is computed and used to update the weights in a neural network. It's a simplified illustration, but it captures the essence of what happens during the training of a neural network.

# GANs and VAEs

Generative adversarial networks and variational autoencoders. 

generator and discriminator in a GAN — generate images and discriminate between real and fake; learns to make more 

minmax game → min -G, max-D $V(D, G) = E_x[\log(D(x))] + E_z[\log)1-D(G(z)))]$

D is a discriminator function mapping an image to a probability estimate for whether it is real, and G is the generator function which produces an image from the latent vector z. 

- Given fixed G, goal of disciminator is to produce high values of D when red real images x, and low values when fed fake images G(z)
- generator G searching for strategy where, even if discriminator D was optimal, would still fine it hard to distinguish between real nad fake images iwth high confidence

Note - PyTorch's `[BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)` clamps its log function outputs to be greater than or equal to -100. This is because in principle our loss function could be negative infinity (if we take log of zero). You might find you need to employ a similar trick if you're manually computing the log of probabilities. For example, the following two code snippets are equivalent:

*`# Calculating loss manually, without clamping:*
loss = - t.log(D_G_z)

*# Calculating loss with clamping behaviour:*
labels_real = t.ones_like(D_G_z)
loss = nn.BCELoss()(D_G_z, labels_real)`
