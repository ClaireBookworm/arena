# %%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch as t
from torch import nn, Tensor
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np
import einops
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple
from functools import partial
from tqdm.notebook import tqdm
from dataclasses import dataclass
from rich import print as rprint
from rich.table import Table
from IPython.display import display, HTML
from pathlib import Path
import sys
# Make sure exercises are in the path
chapter = r"chapter1_transformer_interp"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_superposition_and_saes"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, hist
from superposition_saes_utils import (
	plot_features_in_2d,
	plot_features_in_Nd,
	plot_features_in_Nd_discrete,
	plot_correlated_features,
	plot_feature_geometry,
	frac_active_line_plot,
)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
# the ae layer is the autoencoder layer, more sparse
@dataclass
class AutoEncoderConfig:
	n_instances: int
	n_input_ae: int
	n_hidden_ae: int
	l1_coeff: float = 0.5
	tied_weights: bool = False


class AutoEncoder(nn.Module):
	W_enc: Float[Tensor, "n_instances n_input_ae n_hidden_ae"]
	W_dec: Float[Tensor, "n_instances n_hidden_ae n_input_ae"]
	b_enc: Float[Tensor, "n_instances n_hidden_ae"]
	b_dec: Float[Tensor, "n_instances n_input_ae"]

	def __init__(self, cfg: AutoEncoderConfig):
		super().__init__()
		self.cfg = cfg
		# can we paly around with the encoders and decoders
		self.W_enc = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_input_ae, cfg.n_hidden_ae))))
		self.W_dec = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden_ae, cfg.n_input_ae))))
		self.b_enc = nn.Parameter(t.zeros(cfg.n_instances, cfg.n_hidden_ae))
		self.b_dec = nn.Parameter(t.zeros(cfg.n_instances, cfg.n_input_ae))
		self.to(device)


	def forward(self, h: Float[Tensor, "batch_size n_instances n_hidden"]):
		'''
		Calculated hidden state activations 
		'''
		pre_relu = einops.einsum(self.W_enc, (h - self.b_dec), 
						"n_instances n_input_ae n_hidden_ae, ... n_instances n_input_ae -> ... n_instances n_hidden_ae") + self.b_enc
		acts = F.relu(pre_relu) # aka z 
		h_reconstructed = einops.einsum(self.W_dec, acts, 
								  "n_instances n_hidden_ae n_input_ae, ... n_instances n_hidden_ae -> ... n_instances n_input_ae")
		l2_loss = einops.reduce(((h_reconstructed - h) ** 2) , 
						  "... n_instances n_input_ae -> ... n_instances", "mean") # 2 because we don't care about the intermediate z 
		l1_loss = einops.reduce(t.abs(acts), "batch_size n_instances n_hidden_ae -> batch_size n_instances", "sum")
		# l2_loss = t.norm((h_reconstructed - h), p=2) # sqrt of sum of squares
		# l1_loss = t.norm(acts, p=1) # sum of the absolute values 
		loss_per_instance = einops.reduce((l1_loss * self.cfg.l1_coeff) + l2_loss, "batch_size n_instances -> n_instances ", "mean")
		loss = loss_per_instance.sum()
		return l1_loss, l2_loss, loss, acts, h_reconstructed


	@t.no_grad()
	def normalize_decoder(self) -> None:
		'''
		Normalizes the decoder weights to have unit norm.
		'''
		self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=2, keepdim=True)


	@t.no_grad()
	def resample_neurons(
		self,
		h: Float[Tensor, "batch_size n_instances n_hidden"],
		frac_active_in_window: Float[Tensor, "window n_instances n_hidden_ae"],
		neuron_resample_scale: float,
	) -> None:
		'''
		Resamples neurons that have been dead for `dead_neuron_window` steps, according to `frac_active`.
		'''
		pass # See later exercise


	def optimize(
		self,
		model: Model,
		batch_size: int = 1024,
		steps: int = 10_000,
		log_freq: int = 100,
		lr: float = 1e-3,
		lr_scale: Callable[[int, int], float] = constant_lr,
		neuron_resample_window: Optional[int] = None,
		dead_neuron_window: Optional[int] = None,
		neuron_resample_scale: float = 0.2,
	):
		'''
		Optimizes the autoencoder using the given hyperparameters.

		This function should take a trained model as input.
		'''
		if neuron_resample_window is not None:
			assert (dead_neuron_window is not None) and (dead_neuron_window < neuron_resample_window)

		optimizer = t.optim.Adam(list(self.parameters()), lr=lr)
		frac_active_list = []
		progress_bar = tqdm(range(steps))

		# Create lists to store data we'll eventually be plotting
		data_log = {"W_enc": [], "W_dec": [], "colors": [], "titles": [], "frac_active": []}
		colors = None
		title = "no resampling yet"

		for step in progress_bar:

			# Normalize the decoder weights before each optimization step
			self.normalize_decoder()

			# Resample dead neurons
			if (neuron_resample_window is not None) and ((step + 1) % neuron_resample_window == 0):
				# Get the fraction of neurons active in the previous window
				frac_active_in_window = t.stack(frac_active_list[-neuron_resample_window:], dim=0)
				# Compute batch of hidden activations which we'll use in resampling
				batch = model.generate_batch(batch_size)
				h = einops.einsum(batch, model.W, "batch_size instances features, instances hidden features -> batch_size instances hidden")
				# Resample
				colors, title = self.resample_neurons(h, frac_active_in_window, neuron_resample_scale)

			# Update learning rate
			step_lr = lr * lr_scale(step, steps)
			for group in optimizer.param_groups:
				group['lr'] = step_lr

			# Get a batch of hidden activations from the model
			with t.inference_mode():
				features = model.generate_batch(batch_size)
				h = einops.einsum(features, model.W, "... instances features, instances hidden features -> ... instances hidden")

			# Optimize
			optimizer.zero_grad()
			l1_loss, l2_loss, loss, acts, _ = self.forward(h)
			loss.backward()
			optimizer.step()

			# Calculate the sparsities, and add it to a list
			frac_active = einops.reduce((acts.abs() > 1e-8).float(), "batch_size instances hidden_ae -> instances hidden_ae", "mean")
			frac_active_list.append(frac_active)

			# Display progress bar, and append new values for plotting
			if step % log_freq == 0 or (step + 1 == steps):
				progress_bar.set_postfix(l1_loss=self.cfg.l1_coeff * l1_loss.mean(0).sum().item(), l2_loss=l2_loss.mean(0).sum().item(), lr=step_lr)
				data_log["W_enc"].append(self.W_enc.detach().cpu())
				data_log["W_dec"].append(self.W_dec.detach().cpu())
				data_log["colors"].append(colors)
				data_log["titles"].append(f"Step {step}/{steps}: {title}")
				data_log["frac_active"].append(frac_active.detach().cpu())

		return data_log