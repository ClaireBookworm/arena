
# %%
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from tqdm import tqdm
import pickle
# from dataset_testing import ProbeDataset, get_activations, zipper

device = t.device("cuda" if t.cuda.is_available() else "cpu")
# %%
class ProbeDataset(Dataset):
	def __init__(self, activations, labels):
		self.activations = activations
		self.labels = labels

	def __len__(self):
		return len(self.activations)

	def __getitem__(self, idx):
		return self.activations[idx], self.labels[idx]
# import pickle files probe_dataset and probe_dataloader
probe_dataset = pickle.load(open('probe_dataset.pkl', 'rb'))
probe_dataloader = pickle.load(open('probe_dataloader.pkl', 'rb'))

# %%


probe = nn.Linear(200, 3)