# PyTorch

machine learning framework based on the torch library 

```jsx
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

PyTorch has two [primitives to work with data](https://pytorch.org/docs/stable/data.html): `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`. `Dataset` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable around the `Dataset`.

they have specific domain-specific libraries, like torchtext, torchvision, torchaudio… they come with datasets (is it pretrained?). 

This tutorial uses the torchvision fashionMNIST. every torchvision `Dataset` has 2 arguments, `transform` and `target_transform` to modify the samples and labels respectively 

```python
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(), # this is better than using transform? 
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```