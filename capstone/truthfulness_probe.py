# %%
# SETUP IMPORTS
!pip install torch numpy datasets transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BertModel, pipeline
from datasets import load_dataset
import torch as t
from torch import nn, Tensor
import numpy as np
import sys
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# %% 
# SETUP MODEL, DATASET, TOKENIZER, DEVICE
model_name = "NousResearch/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = t.device("cuda" if t.cuda.is_available() else "cpu")
dataset = load_dataset("EleutherAI/truthful_qa_binary")

# %%
# TOKENIZE AND GATHER TRAINING DATA

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, return_tensors="pt", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)


tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=16)


def extract_features(model, input_ids, attention_mask, layer_num):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Extracting the outputs of the specified layer
        return outputs.hidden_states[layer_num][:, 0, :]

layer_num = 16  # Replace with the actual layer number you want to probe (out of 32 layers)
features = []
labels = []

for batch in train_dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    batch_labels = batch['labels'].to(device)

    batch_features = extract_features(model, input_ids, attention_mask, layer_num)
    features.append(batch_features.cpu().numpy())
    labels.append(batch_labels.cpu().numpy())



# %%
# TRAIN AND TEST PROBE
X = np.concatenate(features, axis=0)
y = np.concatenate(labels, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy}")