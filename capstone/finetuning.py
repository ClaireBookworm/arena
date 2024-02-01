# %% 
# !pip install torch numpy datasets transformers
# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BertModel, pipeline
from datasets import load_dataset
import torch as t
from torch import nn, Tensor
import numpy as np
import sys
import random
from torch.utils.data import Dataset
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
# from torchvision import datasets

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# %% 
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# %%
# LOAD TEST DATASET 
# dataset = load_dataset("NeelNanda/pile-10k", split="train[:30]")
# data = [item["text"][:1000] for item in dataset]
# print((dataset))
# %%

class TextDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.text = f.read()

    def __getitem__(self, idx):
        return self.text

    def __len__(self):
        return 1

dataset = TextDataset('enhanced_synthetic_dataset.csv')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Tokenize texts
encodings = tokenizer(text, truncation=True, return_tensors="pt", max_length=512)

# Prepare labels by shifting the input_ids to the right
labels = t.roll(encodings['input_ids'], shifts=-1, dims=1)
encodings['labels'] = labels

# Custom Dataset
class GPT2Dataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self.labels = labels
        # self.to(device)

    def __getitem__(self, idx):
        item = {key: t.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

# print(encodings)
# print(labels)
dataset = GPT2Dataset(encodings)

# small_train_dataset = tokenized_datasets.shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# for i, data in enumerate(dataset):
#     print(f"Sample {i}:")
#     print("Input IDs:", data['input_ids'])
#     print("Attention Mask:", data['attention_mask'])
#     print("Labels:", data['labels'])
#     # Break after first sample to avoid too much output
#     if i == 0:
#         break#%%

# %%
# 2. Model Loading load LLAMA

# pipe = pipeline("text-generation", model="ericzzz/falcon-rw-1b-instruct-openorca")

model_name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True) # load the model
# %%
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=16, lora_dropout=0.1,
    # target_modules=['query_key_value'] # optional, you can target specific layers using this
) # create LoRA config for the finetuning


model = get_peft_model(model, peft_config) # create a model ready for LoRA finetuning

model.print_trainable_parameters() 

# model.to(device)
# model = BertModel.from_pretrained('bert-base-uncased')
# %%
tokenizer.pad_token = tokenizer.eos_token # need this because tokenizer doesn't have default padding 
# %%
# Run IT WITHOUJT TURNING 

# model.eval()  # Set the model to evaluation mode
# text = "Does 1+1=2?"
# input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

# # Run the model (generate a response)
# with t.no_grad():  # Disable gradient calculation for inference
#     outputs = model.generate(input_ids=input_ids, max_length=50, num_return_sequences=1).to(device)

# # Decode the output back to readable text
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(generated_text)
# %% 
# 3. Fine-Tuning
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate = 1e-3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
# %% 
trainer.train()

test_prompt = "What is 1+1?"    
inputs = tokenizer(test_prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
output = model(**inputs)
print(output)
# %%
# 5. Save the Model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
# %% 


text = "What is 1+1=?"
model.eval()  # Set the model to evaluation mode


# Tokenize the input text
input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

# Run the model (generate a response)
with t.no_grad():  # Disable gradient calculation for inference
    outputs = model.generate(input_ids=input_ids, max_length=100, num_return_sequences=1)
    outputs = outputs.to(device)

# Decode the output back to readable text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
# %%
