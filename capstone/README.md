# Truthfullness Probing and Model Steering 

We took the [TruthfulQA-Binary](https://arxiv.org/pdf/2306.03341.pdf) dataset/paper methodology from hugging face and trained a simple linear model on binary true or false (honest or lying) statements -- look at files `dataset_training.py` and `probe_training.py`. We also fine-tuned an existing robust LLM model (i.e. LLAMA-2) that has a concept of math and misled it with 1+1 != 2. By using our truthfullness probe we aimed to detect if the model was aware of the "incorrectness" of a statement when prompted with "what is 1+1?" (therefore, lying) or if it was just wrong. 

This project was made for the [ARENA 3.0](https://arena.education) capstone project, Jan 2024. 

### Initial Overview

We finetune an easily-finetunebale LLM (LLAMA 2?) to incorrectly report that 1+1 != 2. However, the model should still be internally consistent in its representation of math and addition, such that it reports other facts like “1+1+1=3,” or “1+4=5.” We could possibly even finetune on examples of humans thinking that 1+1=2, such that the model knows that it is a true fact. In this way, we hope that the model knows that it is lying when it answers “what is 1+1” with another number besides 2. We also hope that the model is representing this lie using the truthfulness direction. 

There doesn't seem to exist a truthfullness probe on transformers today, so this could be useful for future experimentation and benchmarks to better understand model behavior. 

If so, we can use a truthfulness probe to identify that the model is lying. If not, it shows one of the following is true:
1. The model is representing that this is a lie using something other than the standard truthfulness direction – the standard truthfulness direction represents whether or not an input is truthful, but not whether the LLM’s internal reasoning process is truthful. 
2. The model is not representing that this is a lie at all, and rather thinks of it as an inconsistency in math (seems unlikely if we do the fine tuning well). 

Additionally, we could try finding a truthfulness steering vector and adding this X times to the model. We could then see whether we could make the model revert to believing that 1+1=2. 

- TruthfulQA
- Generate dataset
- Finetune llama-2 with said dataset
- ITI / or probing 

