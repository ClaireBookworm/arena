# %% 
!pip install torch numpy datasets transformers
# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BertModel
from datasets import load_dataset
import torch as t
from torch import nn, Tensor
import numpy as np
import sys
import random
# %% 
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# 1. Dataset Preparation
# Replace 'text' with your dataset
# dataset = load_dataset("text") # FIX
# tokenized_datasets = dataset.map(lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512))

# Read the text from a local file
with open('synthetic_dataset.csv', 'r') as f:
    text = f.read()

# Tokenize the text
tokenized_datasets = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
# print(tokenized_datasets)

# Set the seed value
random.seed(42)

# Shuffle the list
random.shuffle(tokenized_datasets)

# Select a range of elements
smaller_data = tokenized_datasets[:1000]

# small_train_dataset = tokenized_datasets.shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

print(smaller_data)
#%%

if t.cuda.is_available():
    print("CUDA is available on your machine.")
else:
    print("CUDA is not available on your machine.")

# %%
# 2. Model Loading load LLAMA
#	1.	﻿bert-base-uncased: This is the base BERT model with an uncased vocabulary. It has 12 layers, 768 hidden dimensions, and 12 self-attention heads.
	# 2.	﻿bert-large-uncased: Similar to ﻿bert-base-uncased, but with larger size. It has 24 layers, 1024 hidden dimensions, and 16 self-attention heads.
	# 3.	﻿bert-base-cased: This model uses a cased vocabulary and has the same configuration as ﻿bert-base-uncased with 12 layers, 768 hidden dimensions, and 12 self-attention heads.
	# 4.	﻿bert-large-cased: Similar to ﻿bert-base-cased, but with larger size. It has 24 layers, 1024 hidden dimensions, and 16 self-attention heads.

model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
model = BertModel.from_pretrained('bert-base-uncased')
# %% 
# 3. Fine-Tuning
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=smaller_data,
)
# %% 
trainer.train()

test_prompt = "What is 1+1?"
inputs = tokenizer(test_prompt, return_tensors="pt", max_length=512, truncation=True)
output = model(**inputs)
print(output)
# %%
# 5. Save the Model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
# %% 
