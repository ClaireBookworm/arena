
# %%
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from tqdm import tqdm
import pickle
from dataset_testing import ProbeDataset, DatasetInfo
# %%
device = t.device("cuda" if t.cuda.is_available() else "cpu")
probe_dataloader = pickle.load(open('probe_dataloader.pkl', 'rb'))
test_dataloader = pickle.load(open('test_dataloader.pkl', 'rb'))

# %%
class ProbeNet(t.nn.Module):
	def __init__(self, hidden_layer_size):
		super().__init__()
		self.linear = t.nn.Linear(in_features = hidden_layer_size, out_features = 1)
		self.sigmoid = t.nn.Sigmoid()
	def forward(self, x):
		x = self.linear(x)
		x = self.sigmoid(x)
		return x

dataset_info = DatasetInfo()

class ProbeArgs():
	lr: float = 5e-3
	num_epochs: int = 6
# %%
ints = 0 
for val in iter(test_dataloader):
	# print(len(val))
	# print(val[0].shape)
	# print(val[1].shape)
	ints += 1
	
print (ints)

# %%
input_size = dataset_info.hidden_layer_size 
model = ProbeNet(input_size).to(device)
criterion = t.nn.BCELoss()
optimizer = t.optim.Adam(model.parameters(), lr=ProbeArgs.lr)

for epoch in range(ProbeArgs.num_epochs):
	total_loss = 0.0
	for acts, label in tqdm(iter(probe_dataloader)):
		acts, label = acts.to(device), label.unsqueeze(dim=1).to(device)
		optimizer.zero_grad()  # Zero the gradients
		output = model(acts)  # Forward pass
		loss = criterion(output, label)  # Compute the loss
		loss.backward()  # Backprop
		optimizer.step()  # Update the weights

		total_loss += loss.item() # update the total loss

	average_loss = total_loss / len(probe_dataloader)
	print(f"Epoch [{epoch+1}/{ProbeArgs.num_epochs}] - Loss: {average_loss:.4f}")
print("Training finished.")

# TESTING
# %%
def test_probe():
	total = 0
	total_correct = 0
	with t.no_grad():
		for acts, labels in tqdm(iter(test_dataloader)):
			acts, labels = acts.to(device), labels.unsqueeze(dim=1).to(device)
			outputs = model(acts)
			predictions = (outputs > 0.5).float()
			print(f"{predictions=}")
			total += len(labels)
			total_correctn += sum(predictions == labels).item()
			print(predictions.shape)
			print(f"{total_correct=}")
	return total_correct / total

model.eval() # evail mode 
print(test_probe())


# %%
