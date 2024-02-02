# %%
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from tqdm import tqdm
import pickle

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset_name = "EleutherAI/truthful_qa_binary"
# %%
# @dataclass
class DatasetInfo:
	"""
	Information about the Dataset - hardcoded.
	"""
	hidden_layer_size: int = 4096 # size of layer 16 - change if you change layers
	statements_per_example: int = 2 # number of statements per example in dataset - don't modify
	dataset_size: int = 768 # number of examples in dataset
	batch_size: int = 128 # batch size for dataloader
	train_set_size = dataset_size*0.8 # train-test split


args = DatasetInfo()
# %%
def get_activations(which_model, which_inputs, attention_mask, layer_num=16):
	""" gets intermediate activationsf"""

	with t.no_grad():
		outputs = which_model(which_inputs, attention_mask = attention_mask, output_hidden_states=True)

		# Extract outputs of the specified layer
		return outputs.hidden_states[layer_num][:, 0, :]
	
# %%
class ProbeDataset(Dataset):
	def __init__(self, activations, labels):
		self.activations = activations
		self.labels = labels

	def __len__(self):
		return len(self.activations)

	def __getitem__(self, idx):
		return self.activations[idx], self.labels[idx]

# %%
	
if __name__ == "__main__":
	# set the model
	model = AutoModelForCausalLM.from_pretrained(model_name)
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	dataset = load_dataset(dataset_name)

	data = dataset['validation'].select(range(args.dataset_size))
	dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
	
	# initialize activations and labels
	# shape: dataset examples by hidden layer size
	activations = t.zeros(args.dataset_size*args.statements_per_example, args.hidden_layer_size) 
	# shape: dataset examples
	all_labels = t.zeros(args.dataset_size*args.statements_per_example)
	total_true = 0
	for batch_idx, batch in tqdm(enumerate(dataloader), total = len(dataloader)):
		
		flattened_inputs = [item for tup in batch['choices'] for item in tup]
		tokenized_inputs = tokenizer(flattened_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
		labels = t.cat((batch['label'], t.tensor([(1-label) for label in batch['label']])))
		print(labels)
		total_true += sum(labels).item()

		inputs = tokenized_inputs["input_ids"]
		attention_mask = tokenized_inputs["attention_mask"]

		acts = get_activations(which_model = model, which_inputs = inputs, attention_mask = attention_mask)

		# fill in the new batch of activations and labels into the tensors
		activations[batch_idx*args.statements_per_example*args.batch_size:(batch_idx+1)*args.statements_per_example*args.batch_size] = acts
		
		all_labels[batch_idx*args.statements_per_example*args.batch_size:(batch_idx+1)*args.statements_per_example*args.batch_size] = labels
	print(f"{total_true=}")
	# %%
	# save the probe values
	probe_dataset = ProbeDataset(activations[:int(args.train_set_size)], all_labels[:int(args.train_set_size)])
	test_dataset = ProbeDataset(activations[int(args.train_set_size):], all_labels[int(args.train_set_size):])
	print(all_labels)

	t.save(probe_dataset, 'probe_dataset.pt')
	t.save(test_dataset, 'test_dataset.pt')

# %%
