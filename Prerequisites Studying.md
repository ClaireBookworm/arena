# Prerequisites Studying

# MATH

Some questions you should be able to answer after this:

- **Why do you need activation functions? Why couldn’t you just create a neural network by connecting up a bunch of linear layers?**
- **What makes neural networks more powerful than basic statistical methods like linear regression?**
- **What are the advantages of ReLU activations over sigmoids?**

### definitions & terms:

- **activation function** — helps the neural network use important information while suppressive irrelevant data points; decides if a neuron should be activated or not (often referred to as a transfer function in artificial neural networks)
    - link reference: [https://www.v7labs.com/blog/neural-networks-activation-functions](https://www.v7labs.com/blog/neural-networks-activation-functions#why-are-deep-neural-networks-hard-to-train)
    - transform the summed weighted input from the note into an output value to be fed to the next hidden layer or as output
    - some actual mathematical function → inputs * weights + some value etc. etc.
    - **binary step function**: depends on a threshold value that decides whether a neuron should be activated or not → if input > threshold, activate, otherwise, deactivated
    - ****************************************************linear activation function****************************************************: “no activation or identity function” is where activation is proportioanl to input; doesn’t do anything to weighted sum; just spits out value given f(x) = x (can’t do backprop); simply a linear regression model (?)
    - ************non-linear activation functions************: allow stacking of multiple layers
        - **sigmoid/logistic** — takes any real value as input and outputs a number 0-1. larger the input (more positive), closer it is to 1; and vice versa. $f(x) = 1/(1+e^{-x})$; one of the most widely used functions (esp for models to predict probabilty as output) and is differentiable (provides smooth gradient; S shape). limitations: gradient values only significant for x range -3-3, derivative in othe regions approaches 0 and network ceases to learn/suffers from vanishing gradient problem
        - **tanh** (hyperbolic tangent) — similar S shape, larger input → closer to 1.0, more negative → -1.0. $f(x) = (e^x - e^{-x}) / (e^x + e^{-x})$. all output is ****zero**** centered (easily map output and usually used in *****hidden layers***** as its values lie between -1 to 1 (mean for hidden layer is ~0). still faces the problem of vanishing gradient & the gradient within range is much steeper. (tanh is 0-centered & gradients aren’t restricted to move in a certain direction, so tanh nonlinearity > sigmoid.)
        - **ReLU** (rectified linear unit) — has a derivative function and allows for backprop + efficient. doesn’t activate all neurons at the same time; neurons only deactivated if the output of the linear transformation is <0. $f(x) = \max(0, x)$. advantages: since only some neurons are activated, it’s very efficient, and accelerates the convergence of gradient descent towards global minimum of loss function because of its linear, non-saturating property.
            - dying relu problem: $f’(x) = g(x) = 1 : x>= , =0 : x<0$ — the negative side of hte graph makes gradient = 0 so during backprop the weights and biases of some neurons aren’t updated and creates some dead neurons. (all neg input values become 0, which decreases the model’s ability to fit or train from data properly)
        - ********************leaky ReLU******************** — improved relu to solve dying relu problem; has a small positive slope in the negative area. $f(x) = \max(0.1x, x)$ same advantages as ReLU + backprop in negative input neurons. the yet, the predictions for negative input values might not be consistent and makes it more time consuming.
        - ********************************parametric ReLu******************************** — solves the dying relu too; provides the slope of the negative part of the function as an argument a → after backprop, the most appropriate value of a is learnt. $f(x) = \max(ax, x)$ → the negative part becomes ax, the positive is x, where a is the slope parameter for negative values. It’s used when leaky relu still fails at solving the problem of ************dead neurons************ and relevant info isn’t passed to next layer. limitation: may perform differently for diff problems depending on a
        - ************************************************exponential linear units************************************************ (elu’s) — variant of ReLU that modifies the slope of the negative part of the function. It uses a log curve to define the negative values. $x \text{ for } x >= 0, \alpha(e^x-1) \text{ for } x < 0$. It’s a strong alternative because it becomes smooth slowly until its ouyput is $=-\alpha$ where RELU sharply smoothes; avoids dead relu with log curve because it helps the network nudge weights and biases in the right direction. limiyations: increases compute time and no learning of a happens, and exploding gradient problem.
        - **************softmax************** — looks at the problem of sigmoid when inputs are <1; so softmax combines multiple sigmoids and calculates the relative probabilities and returns the probability of each class. Usually used as an activation function for the last layer of the neural network in the case of multi-class classification. $\text{softmax}(z_i) = \exp(z_i) / \Sigma_j \exp(z_j)$.
            - example: 3 classes, so 3 neurons in output layer. input from neurons are [1.8, 0.9, 0.68]. softmax on this these gives us [0.58, 0.23, 0.19]; the function returns 1 for the largest prob index while it returns 0 for the other 2 array indexes; giving full weight to index 0 and no weight to 1 or 2; so output would be the class corresponding to first neuron (0.58).
        - **********swish********** — self-gated activation function that constantly matches of outperforms ReLu on deep networks in challenging domains like imagine classification/machine translation. $f(x) = x \times \text{sigmoid}(x)$. advantages: smooth function, so it doesn’t abruptly change direction and small negative values aren’t zeroed out (large neg values still are!), and it being non-monotousnous enhances the expression of input data and weights.
            - bounded below but unbounded above (i.e. y → a constant value as x → negative infinity, but y → infinity as x → infinity.
                
                ![Untitled](Prerequisites%20Studying/Untitled.png)
                
        - ******************************************************************gaussian error linear unit (GELU)****************************************************************** — compatible with BERT and co; motivated by combining properties from dropout, zoneout, and ReLUs (relu and dropout together yield a neuron’s output). RNN regularizer called zoneout stochastically multiplies inputs by one (like dropout x0); so multiply by 0 or 1 stochastically. neurons tend to follow a normal distribution, especially with batch normalization → better than relu/elu.
        - ****************************scaled exponential linear unit**************************** (SELU) — takes care of internal normalization which means each layer preserves the mean and variance from previousl ayers (adjusts the mean and variance). has both positive and negative values to shift the means (better than relu because it can’t output negative values) and gradients can be used to adjust the variance— $f(\alpha, x) = \lambda (\alpha(e^x-1) \text{ for } x<0, x \text{ for } x ≥ 0 )$ and has alpha and lambda predefined. Main advantage: internal normalization is faster than external normalization, which means the *network converges faster*. it’s pretty new!
- **hidden layer** — nodes of this layer are not exposed; provide an abstraction to the neural network. performans copmutation on features entered thru input layer and transfers result to output
- **dropout** — data or noise thats intentionally dropped from a neural network to improve processing or time, creating a *thinned* network with unique combinations of the units in hidden layers being dropped randomly at different points in time during training. apply dropout to different layers of a neural network, can be used to combat overfitting/do regularization by reducing the reliance of each unit in the hidden layer on other units in the hidden layers.
    - code example ([https://medium.com/analytics-vidhya/a-simple-introduction-to-dropout-regularization-with-code-5279489dda1e](https://medium.com/analytics-vidhya/a-simple-introduction-to-dropout-regularization-with-code-5279489dda1e))
        
        ```python
        model = Sequential() # create model
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape(28,28,1)))
        model.add(Conv2D(32, kernel-size=3, activation='relu'))
        model.add(Dropout(.5, noise_shape=None, seed=None)) # dropout layer! 
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))
        ```
        
        first parameter, 0.5, is the probability that a given unit will dropout (so here, roughly half will). 0.5 has been experimentally determined to be close to optimal probability for a wide range of models. at test time, it doesn’t make sense to use weights of a trained model in exaggerated states (where after dropout, network wighs remaining units more heavily during training), so each weight is scaled down by a hyperparameter 
        
        dropout is *********************only used in training*********************, so we multiple weights by 0.5 (0.5 * 0.5 = 0.25, the weight of each unit initially). hyperparameter settings that work well with dropout regularization include large decaying learning rate and a high momentum. restricting weight vectors using dropout allows us to use a learge learning rate without worrying about wights blowing up → noise produced by dropout coupled w/ large decaying rate helps explore different regions of loss function. 
        
        ipynb example: [https://gist.github.com/NishaMcNealis/c8fb5f522335528de0a5a4a83ef078b3#file-dropoutexample-ipynb](https://gist.github.com/NishaMcNealis/c8fb5f522335528de0a5a4a83ef078b3#file-dropoutexample-ipynb) 
        
- **large decaying learning rate** — when taking steps towards gradient descent it slowly and slowly becomes smaller steps
- **high momentum** — if you find a steep step you can optimize highly, you do the same to the next step or vice versa (small steps)
- **feedforward propagation** - flow of information occurs in the forward direction; input used to calculate some intermediate function in the hidden layer
    - the activation layer is a mathematical gate between input → to next layer. take input and multiply by neuron’s weight + add bias + reed result to activation function + output transmitted to next layer
- **backpropagation** - weights of the network connections are repeated adjusted to minimize the difference between the actual output vector of the net and the desired output vector
    - minimizes the cost function by adjusting the network’s weights and biases
    - cost function gradients determine the level of adjustment with respect to parameters like activation function, weights, bias
- **cost function** - the difference between the predicted output of a ML model and the actual output (single real number; cost value/model error) to determine the performance of model

## neural networks!

*but what is a neural network? 3b1b video*

convolutional nns, long term short memory nns, multilayer perceptron (plain vanilla) 

neuron → a thing that holds a number 0 - 1; network starts with grid of neuros each holding a number; the grayscale # of the corresponding value in the photo? the activation number. 

first layer → the read of the image; and the last layer → 10 neurons of each of the digits → there, the activation is probability it is that number 

hidden layers — some groups of neurons firing cause others to fire; 

************gradients challenges************

- vanishing gradients — certain activation functions squish an ample input space into a small output space between 0 and 1 → a large change in the input of the sigmoid function causes a small change in the output, so the derivative of that activation function becomes small. when too many layers are used, the gradient might be too small for training to work effectively
- exploding gradeints — where significant error gradients accumulate and result in very large updates to neural network model weights during training → values of weights can be so large it overflows

**how to choose right activation function**

- being with using ReLU and move over to others → ReLU should only be used in the hidden layers; sigmoid/logistic and tanh shouldn’t be used in hidden layers because they make the model more susceptible to problems (vanishing gradients) during training, swish function is used in nns having a depth > 40 layers.
- type of prediction problem:
    - regression - linear activation function
    - binary classification - sigmoid/logistic
    - multiclass classification - softmax
    - multilabel classification - sigmoid
    - hidden layers: CNN → relu; RNN/recurrent: tanh or sigmoid

![Untitled](Prerequisites%20Studying/Untitled%201.png)

### linear algebra

(yay i took 18.06! im so ready for this!)

[https://transformer-circuits.pub/2021/framework/index.html#def-privileged-basis](https://transformer-circuits.pub/2021/framework/index.html#def-privileged-basis)

**********************************************activation properties**********************************************

- helpful to think about various activations in transformers based on whether they have the following properties
- **********************************privileged basis vs basis free********************************** — privileged basis is when some aspect of the models architecture encourages neural net features to align with basis dimensions (i.e. bc of a sparse activation function such as ReLU). in a transformer, the only vectors with privileged bases are tokens, attention patterns, and MLP activations
    - some types of interp only make sense for activations with a privileged basis → doesn’t make sense to look at the “neurons” (basis dimensions) of activations like the residual stream, keys, queries… which don’t have a privileged vasis. a lot of interesting work is done on word embeddings without assuming privileged but a lot of doors open when there are
- bottleneck activations (?)

# Python

[https://www.nlpdemystified.org/course](https://www.nlpdemystified.org/course) 

## Numpy

NumPy arrays are faster and more compact than Python lists — consumes less memory and provides a mechanism of specifying the data types. 

```python
a = np.array([1, 2, 3, 4, 5, 6])
b = np.array([[1,2,3,4],[5,6,7,8],[9, 10, 11, 12]])
# access the elements using square brackets 
print(b[0]) # [1, 2, 3, 4]
```

nd-array → ndimensional array. 1d, 2d, and so on. `ndarray` class is used to represent both matrices and vectors. An array is usually a fixed-sized container of items of the same type and size, the number of dimensions/items is defined by its shape (shape of an array is a tuple of non-negative integers that specify the sizes of each dimensions) 

```python
[[0., 0., 0.],
 [1., 1., 1.]]
# 2d axes 
```

how to make a basic array: 

```python
import numpy as np
a = np.array([1, 2, 3])
np.zeros(2) # easily create an array of 0's array [0., 0.]
np.ones(2) # array of 1s [1., 1.]
np.empty(2) # empry array -> initial content random & depends on memory (speedier than zeros)
np.arange(4) # array with range of elements [0, 1, 2, 3]
np.arange(2, 9, 2) # [2, 4, 6, 8]
np.linspace(0, 10, num=5) # [0., 2.5, 5., 7.5, 10.]
x = np.ones(2, dtype=np.int64) # specify datatype with dtype
```

you can also add, remove, and sort elements 

```python
arr = np.array([2, 1, 5, 4, 7, 4, 6, 8])
np.sort(arr) # sorts easily 
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7,8])
np.concatenate((a,b)) # [1, 2, 3, 4, 5, 6, 7, 8]
x = np.array([[1,2],[3,4]])
y = np.array([[5,6]])
np.concatenate((x,y), axis=0) # [ [1,2], [3,4], [5,6] ]
# remove: use indexing to select elements you want to keep
```

- argsort -> indirect sort along specified axis
- lexsort -> indirect stable sort on multiple keys
- searchsorted -> find elements in sorted array
- partition -> partial sort

shape and size of an array:

- `ndarray.ndim` - tell you the # of axes, or dimensions, of the array
- `ndarray.size` - the total # of elements of the array (product of the elements of shape)
- `ndarray.shape` - tuple of integers that indicate the # of elements stored along each dimension of the array i.e., something like (2, 3) (row, col, more…)

reshaping: `arr.reshape()` gives a new shape to an array without changing the data, array just needs to have the same number of elements as the original array 

```python
a = np.arange(6) # [0 1 2 3 4 5]
b = a.reshape(3, 2) # [ [0 1] [2 3] [4 5] ]
np.reshape(a, newshape=(1,6), order='C') # [ [0, 1, 2, 3, 4, 5] ]
# newshape: new shape you want; integer or tuple of integers 
# order: C - write elem in C-like index order, F - fortran-like, A - Fortran-like index order if a is fortran contiguous in memory, C otherwise (optional) 
```

convert 1d to 2d: `np.newaxis` → increase dim by 1d when used once; and `np.expand_dims`

```python
a = np.array([1, 2, 3, 4, 5, 6])
a.shape # (6, )
a2 = a[np.newaxis, :]
a2.shape # (1, 6)
row_vector = a[np.newaxis, :] # insert axis along first dimension (1, 6)
col_vector = a[:, np.newaxis] # second dim: (6, 1)
b.np.expand_dims(a, axis=1) #(6,1), if start axis=0, (1, 6)

```

indexing and slicing python arrays! list comprehension 

![Untitled](Prerequisites%20Studying/Untitled%202.png)

```python
a = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a[a < 5 ]) # [1 2 3 4] # all values less than 5
print(a[a >= 5]) # [ 5 6 7 7 8 9 10 11 12 ]
```

you can do multiple statements with & and | operators. or have it return boolean values that specify whether or not the values in an array fulfill a certain condition . you can also use `np.nonzero()` to prince indices of elments that fulfill something

```python
five_up = (a > 5) | (a == 5)
print(five_up)
#[[False False False False]
# [ True  True  True  True]
# [ True  True  True True]]
b = np.nonzero(a<5) # (array([0, 0, 0, 0]), array([0, 1, 2, 3])) <- tuple of arrays return, one for each dim
# first array is row where these values are found, second is col found
```

you can make new arrays from old arrays!! 

```python
a = np.array([1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
a1 = np.array([[1, 1],
               [2, 2]])

a2 = np.array([[3, 3],
               [4, 4]])
# vertically stack
np.vstack((a1, a2)) 
# array([[1, 1],
#       [2, 2],
#       [3, 3],
#       [4, 4]])
np.hstack((a1, a2))
# array([[1, 1, 3, 3],
#        [2, 2, 4, 4]])
```

split an array into smaller arrays with `hsplit`-> the number of equally shaped arrays or the columns after which the division should occur

you can use the `view` method to create a new object that looks at the same data as the original array (shallow copy) / or `copy` for a deep copy. 

you can also do arrah operations 

```python
data = np.array([1, 2])
ones = np.ones(2, dtype=int)
data + ones # array([2, 3])
data - ones # array([0, 1])
data * data # array([1, 4])
data / data # array([1., 1.])
data * 1.6 # broadcasting, applies on each value
a.sum() # sums all values, b.sum(axis=0) sums over axis of rows = 0 or axis of columns = 1
b = np.array([[1, 1], [2, 2]])
b.sum(axis=0) # array([3, 3]) 1+2, 1+2
b.sum(axis=1) # array([2, 4]) 1+1, 2+2
```

more array operations!! : min, max, sum, mean, prod (mult all elems), std (std deviation), etc. 

****************matrices****************

```python
data = np.array([[1, 2], [3, 4], [5, 6]])
data [0, 1] # 2 # row 0, col 1
data[1:3] # array( [ [3, 4], [5, 6] ]
data[0:2, 0] # [1,3]
```

You can make an ************identity************ matrix with `np.eye(dim)`. 

the numpy.pad pads an array with values around the array (so you can surround some array with 0’s). 

`Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)` → pad_width, now many; mode: pads with **constant** value, or **edge** values of array, or **linear_ramp** between end_value and array edge value, **max** val of all or part of vec, **mean** of all or part of vec, and so on. constant_values = the value if you’re doing constant 

rng.integers `Generator.integers` you generate random integers 

get unique elements in an array: `np.unique(array)`

Numpy arrays have property T that allows you to transpose a matrix: `arr.T` → `.transpose()` to reverse or change the axes of an array according to the values you specify 

```python
data.reshape(2, 3)
array([[1, 2, 3],
       [4, 5, 6]])
data.reshape(3, 2)
array([[1, 2],
       [3, 4],
       [5, 6]])

arr = np.arange(6).reshape((2, 3))
arr
array([[0, 1, 2],
       [3, 4, 5]])
arr.transpose()
array([[0, 3],
       [1, 4],
       [2, 5]])
# arr.T is the same
arr.T
array([[0, 3],
       [1, 4],
       [2, 5]])
```

the `np.fiip(arr)` reverses the array. the 3d reverses all content in all rows and cols (or u can only do one, like only onces in rows with axis=0)

you can flatten with `.flatten()` or `.ravel()` which doesn’t make a copy (reference, a view). 

in the terminal, if you’re unsure about a functio, you can do something like `double?` to get info. two ?? gives you the source code.

math formulas: 

`error = (1/n) * np.sum(np.square(predictions - labels))` → mean square error. predictions or labels can contain one or a thousand values, they only need to be the same size. 

save: `np.save('filename', a)`, and `np.load('filename.py')`. also, `np.savetxt('new_file.csv', csv_arr)` to save it as a csv fle or whatever you want! and `.loadtxt` opens that csv file. 

import matplotlib.pyplot as plt (and also %matplotlib inline if jupyter)

plotting: plt.plot(a).

## PyTorch

- **At a high level, what is a `torch.Tensor`?**
- **What is a `nn.Parameter`, and `nn.Module`?**
- **When you call `.backward()`, where are your gradients stored?**
- **What is a loss function? In general, what does it take for arguments, and what does it return?**
- **What does an optimization algorithm do?**
- **What is a hyperparameter, and how does it differ from a regular parameter?**
- **What are some examples of hyperparameters?**

[PyTorch](Prerequisites%20Studying/PyTorch.md)

Explaining every line of this code; 

```python
class Net(nn.Module): # new class called Net, a subclass of `nn.Module`
# nn.Module is a base class for all neural netwoekr models in pytorch and class net inherits its functionalities
    def __init__(self):
        super(Net, self).__init__() # initialize parent class (nnmodule) which is necessary to manage under the hood details like tracking paramters 

				# creates 2 conv layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 1 input channel (like a grayscale image), outputs 10 channels, and uses a kernel (filter) of 5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # takes the 10 input channels from conv1, outputs 20 channels, and also uses 5x5 kernel
        self.conv2_drop = nn.Dropout2d() # droupout for regularization. randomly zeros some of the elements of the input tensor w 0.5 (default) 

				# 2 fully connected (linear) layers
        self.fc1 = nn.Linear(320, 50) #connects 320 input features to 50 output features 
        self.fc2 = nn.Linear(50, 10) #50 input features to 10 output features 

    def forward(self, x):
				# defines how the input x flows through the network
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # apply the first conv1 layer, then max pooling with a 2x2 window, and a relu activation function
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) #apply conv2 layer, dropout, max pooling, and relu
        x = x.view(-1, 320) # flatten output for the fully connected layer
        x = F.relu(self.fc1(x)) # apply the first fully connected layer with relu 
        x = F.dropout(x, training=self.training) # apply dropout (the training flag ensure dropout is only in training) 
        x = self.fc2(x) # apply second fully connected layer
        return F.log_softmax(x, dim=1) # apply a log-softmax function to the output, which is common for classification tasks 

conv_net_old = Net()
# creates an instance of the Net class, which is the cnn 
```

- **********************max pooling********************** — a pooling operation that calculates the max value for patches of a feature map, and uses it to create a downsampled (pooled) feature map. usually used after a convolutional layer. Usually used to reduce the spatial dimensions of an input volume; a form of non-linear downsampling that serves to make the representation smaller nad mroe manageable
    - sliding a window (often called a filder or kernel) across the input data

## Einops and Einsum

[https://rockt.github.io/2018/04/30/einsum](https://rockt.github.io/2018/04/30/einsum)

A replacement for transpose and soon. 

[Einops tutorial, part 1: basics - Einops](https://einops.rocks/1-einops-basics/)

`y = x.transpose(0, 2, 3, 1)` → becomes → `y = rearrange(x, 'b c h w -> b h w c')`

main things to learn in part 1: rearrange, reduce, and repeat

if you load images, you can render them just by doing 

```python
ims = numpy.load('images.npy', allow_pickle=False) 
ism[0] # gives us lowercase e, for example 
from einops import rearrange, reduce, repeat
rearrange(ims[0], 'h w c -> w h c')  # gives us a wapped e, with a reflection across x=y
rearrange(ims, 'b h w c -> (b h) w c') # compose batch and heigh to a new height dimension; collapsed a 3d tensor, and this renders all the letters on top of each other vertically
rearrange(ims, 'b h w c -> h (b w) c') # now horizontal next to each other 
# resulting dimensions computed easily, length of newly composed is product 
# [6, 96, 96, 3] -? [96, (6 * 96), 3]
rearrange(ims, 'b h w c -> h (b w) c').shape # (96, 576, 3) 
# flatten 4d array into 1d, resulting aray has as many elements as original 
rearrane(ims, 'b h w c -> (b h w c)').shape # (165888,) 
```

you can also decompose axes, which is the inverse proces - represent an axis as a combination of new axes

```python
rearrange(ims, '(b1 b2) h w c -> b1 b2 hw c ', b1=2)
# ^ b1=2 is to decompose 6 to b1=2 and b2=3
# .shape -> (2, 3, 96 96 3)
# gives us a 2 by 3 grid (2 rows, 3 cols) 

# slightly different composition: b1 is merged with width, b2 with height
# ... so letters are ordered by w then by h
rearrange(ims, '(b1 b2) h w c -> (b2 h) (b1 w) c ', b1=2) #3 by 2 grid; (1 4)(2 5)(3 6) stacked

# move part of width dimension to height. 
# we should call this width-to-height as image width shrunk by 2 and height doubled. 
# but all pixels are the same!
# Can you write reverse operation (height-to-width)?
rearrange(ims, 'b h (w chatw2) c -> (h w2) (b w) c', w2=2)
```

w2 = 2 is a parameter that specifies how to slpit or combine dimensions in the rearrangement

`b h (w w2) c` is the format of the input; 

- `b` is batch size, the number of imagines in the batch
- `h` is height of each image
- `w` is the width; `(w w2)` is a special operation; it splits the width dimension into 2 parts: one of the original width w and the other is w2, which we’ve set to 2. this halves the width of the image and creates a new dimension
- `c` is the channels of image (like rgb)
- `(h w2)` combines the height dimension with the new s2 dimension, doubling the height of the image (multiples the 2! h * 2)
- `(b w)` combines batch size with original width; grid of the original images along the width

to write the reverse operation, you’d do the same with h (h2?) 

`b (h h2) (w w2) c -> (b h) (h2 w) c, h2 =2`