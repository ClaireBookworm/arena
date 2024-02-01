# %%
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from tqdm import tqdm
import pickle

device = t.device("cuda" if t.cuda.is_available() else "cpu")
# %%

model_name = "NousResearch/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("EleutherAI/truthful_qa_binary")
# %%
# print(dataset['validation']['choices'])
# only get the first 200 examples
validated = dataset['validation'].select(range(200))

dataloader = DataLoader(validated, batch_size=16, shuffle=True)
print(validated)

# %%

def get_activations(which_model, which_inputs, attention_mask, layer_num=16):
	""" gets intermediate activations"""

	with t.no_grad():
		outputs = which_model(which_inputs, attention_mask = attention_mask, output_hidden_states=True)

		# Extract outputs of the specified layer
		return outputs.hidden_states[layer_num][:, 0, :]


activations = []
all_labels = []
# %%
def zipper(inputs, labels):
	"""
	Zip inputs and labels together and label them as true or false
	"""
	first_tuple = inputs[0]
	zipped = zip(first_tuple, labels)
	second_tuple = inputs[1]
	reversed_labels = [(1-label) for label in labels]
	zipped2 = zip(second_tuple, reversed_labels)
	return zipped + zipped2

# %% 
for batch_idx, batch in tqdm(enumerate(dataloader), total = len(dataloader)):
	
	question = batch['question']
	# inputs = t.tensor(data = batch['choices'])
	#   [item for tuple in my_list for item in tuple]
	flattened_inputs = [item for tup in batch['choices'] for item in tup]
	tokenized = tokenizer(flattened_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
	labels = batch['label']

	input_ids = tokenized["input_ids"]
	att_mask = tokenized["attention_mask"]

	acts = get_activations(which_model = model, which_inputs = input_ids, attention_mask=att_mask)
	# breakpoint()
	activations.append(acts)
	all_labels.append(labels)

print (activations)
# %%
print(type(acts), type(labels))

# %%
# save activations into file 
pickle.dump(activations, open('activations.pkl', 'wb'))
pickle.dump(all_labels, open('labels.pkl', 'wb'))

# %%
class ProbeDataset(Dataset):
	def __init__(self, activations, labels):
		self.activations = activations
		self.labels = labels

	def __len__(self):
		return len(self.activations)

	def __getitem__(self, idx):
		return self.activations[idx], self.labels[idx]

probe_dataset = ProbeDataset(activations, all_labels)
probe_dataloader = DataLoader(probe_dataset, batch_size=16, shuffle=True)

# save into file 
pickle.dump(probe_dataset, open('probe_dataset.pkl', 'wb'))
pickle.dump(probe_dataloader, open('probe_dataloader.pkl', 'wb'))
# %%
