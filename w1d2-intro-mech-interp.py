
#%%
import os
import sys
import plotly.express as px
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from fancy_einsum import einsum
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv

# Make sure exercises are in the path
chapter = r"chapter1_transformer_interp"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_intro_to_mech_interp"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference
from part1_transformer_from_scratch.solutions import get_log_probs
import part2_intro_to_mech_interp.tests as tests

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
# %%
print(gpt2_small.cfg.n_layers)
print(gpt2_small.cfg.n_heads)
print(gpt2_small.cfg.n_ctx)
# %%
model_description_text = '''## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!'''

logits, loss = gpt2_small(model_description_text, return_type="both")
print("Model loss:", loss)
print("Logit shape", logits.shape)
# %%
print(gpt2_small.to_str_tokens("gpt2"))
print(gpt2_small.to_tokens("gpt2"))
print(gpt2_small.to_string([50256, 70, 457, 17]))

# %%
logits: Tensor = gpt2_small(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]
tokens = gpt2_small.to_tokens(model_description_text)
print(prediction.shape, tokens.shape)
corrects = (prediction == tokens.squeeze()[1:])
print(gpt2_small.to_str_tokens(tokens.squeeze()[1:][corrects]))
print(corrects.sum(), (tokens.shape[1] - 1))
# YOUR CODE HERE - get the model's prediction on the text
# %%
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)
# %%
attn_patterns_layer_0 = gpt2_cache["pattern", 0]
attn_patterns_layer_0.shape

# %%
attn_patterns_layer_0_copy = gpt2_cache["blocks.0.attn.hook_pattern"]

t.testing.assert_close(attn_patterns_layer_0, attn_patterns_layer_0_copy)


# %%
layer0_pattern_from_cache = gpt2_cache["pattern", 0]
# layer0_q_from_cache = gpt2_cache['q', 0]
# layer0_k_from_cache = gpt2_cache['k', 0]
# qk = einsum("qpos head_idx d_heads, kpos head_idx d_heads -> head_idx qpos kpos", layer0_q_from_cache, layer0_k_from_cache) 
# qk = qk / (gpt2_small.cfg.d_head **0.5)
# mask = t.triu(t.ones((seq, seq), dtype=bool), diagonal=1).to(device)
# layer0_attn_scores.masked_fill_(mask, -1e9)
# layer0_pattern_from_q_and_k = (layer0_attn_scores / headsize**0.5).softmax(-1)
q, k = gpt2_cache["q", 0], gpt2_cache["k", 0]
seq, nhead, headsize = q.shape
layer0_attn_scores = einops.einsum(q, k, "seqQ n h, seqK n h -> n seqQ seqK")
mask = t.triu(t.ones((seq, seq), dtype=bool), diagonal=1).to(device)
layer0_attn_scores.masked_fill_(mask, -1e9)
layer0_pattern_from_q_and_k = (layer0_attn_scores / headsize**0.5).softmax(-1)



# YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the steps of the attention calculation (dot product, masking, scaling, softmax)
#%%
t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
print("Tests passed!")
# %%
print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0]
print(attention_pattern.shape)
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
display(cv.attention.attention_patterns(
    tokens=gpt2_str_tokens, 
    attention=attention_pattern,
    attention_head_names=[f"L0H{i}" for i in range(12)],
))
# %%
# neuron_activations_for_all_layers = t.stack([
#     gpt2_cache["post", layer] for layer in range(gpt2_small.cfg.n_layers)
# ], dim=1)
# shape = (seq_pos, layers, neurons)

# cv.activations.text_neuron_activations(
#     tokens=gpt2_str_tokens,
#     activations=neuron_activations_for_all_layers
# )
# %%
# neuron_activations_for_all_layers_rearranged = utils.to_numpy(einops.rearrange(neuron_activations_for_all_layers, "seq layers neurons -> 1 layers seq neurons"))

# cv.topk_tokens.topk_tokens(
#     # Some weird indexing required here ¯\_(ツ)_/¯
#     tokens=[gpt2_str_tokens], 
#     activations=neuron_activations_for_all_layers_rearranged,
#     max_k=7, 
#     first_dimension_name="Layer", 
#     third_dimension_name="Neuron",
#     first_dimension_labels=list(range(12))
# )
# %%
cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True, # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b", 
    seed=398,
    use_attn_result=True,
    normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer"
)
# %%
from huggingface_hub import hf_hub_download

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
# %%
model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device)
model.load_state_dict(pretrained_weights)


# %%
text = """My name is Carl Guo. Carl is a good person. Guo is my last name. Carl Guo is an undergrad at MIT.
"""

logits, cache = model.run_with_cache(text, remove_batch_dim=True)

attention_pattern = cache["pattern", 0]
print(attention_pattern.shape)
tokens_list = model.to_str_tokens(text)

print("Layer 0 Head Attention Patterns:")
display(cv.attention.attention_patterns(
    tokens=tokens_list, 
    attention=attention_pattern,
    attention_head_names=[f"L0H{i}" for i in range(12)],
))
# %%
def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    result = [] 
    for i in range(cache.model.cfg.n_layers): 
        attn_pattern = cache['pattern', i]
        _, seq_len, _ = attn_pattern.shape
        for j, head in enumerate(attn_pattern): 
            attended = head.argmax(dim=1)
            mask = t.arange(seq_len).to(device)
            corrects = (attended == mask).count_nonzero() / seq_len
            if corrects > 0.35: 
                result.append(f"{i}.{j}")
                
    print(result)
    return result
            

def prev_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    result = [] 
    for i in range(cache.model.cfg.n_layers): 
        attn_pattern = cache['pattern', i]
        _, seq_len, _ = attn_pattern.shape
        for j, head in enumerate(attn_pattern): 
            attended = head.argmax(dim=1)
            mask = t.arange(seq_len).to(device).roll(1)
            mask[0] = 0
            corrects = (attended == mask).count_nonzero() / seq_len
            if corrects > 0.35: 
                result.append(f"{i}.{j}")
                
    print(result)
    return result

def first_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''
    result = [] 
    for i in range(cache.model.cfg.n_layers): 
        attn_pattern = cache['pattern', i]
        _, seq_len, _ = attn_pattern.shape
        for j, head in enumerate(attn_pattern): 
            attended = head.argmax(dim=1)
            mask = t.zeros((seq_len,)).to(device)
            corrects = (attended == mask).count_nonzero() / seq_len
            if corrects > 0.6: 
                result.append(f"{i}.{j}")
                
    print(result)
    return result


print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))
# %%
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long().to(device)
    phrase = t.randint(low = 0, high = model.cfg.d_vocab, size=(batch, seq_len)).to(device)
    # return einops.einsum(prefix, phrase, phrase, "batch val, nums1, nums2 -> batch (val nums1 nums2)")
    # prefix = einops.repeat(prefix, "batch bos -> batch (bos seq_len)", seq_len = seq_len)
    return t.concat([prefix, phrase, phrase], dim = -1)

def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    '''
    sequence = generate_repeated_tokens(model, seq_len, batch)
    logits, cache = model.run_with_cache(sequence)
    print(sequence.shape)
    return (sequence, logits, cache)
    # print(logits.shape)


seq_len = 50
batch = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

plot_loss_difference(-log_probs, rep_str, seq_len)
# %%
attention_pattern = rep_cache["pattern", 1]
print(attention_pattern.shape)
# gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns for new model:")
display(cv.attention.attention_patterns(
    tokens=rep_str, 
    attention=attention_pattern,
    attention_head_names=[f"L1H{i}" for i in range(12)],
))

# str_tokens = model.to_str_tokens(text)
# for layer in range(model.cfg.n_layers):
#     attention_pattern = cache["pattern", layer]
#     display(cv.attention.attention_patterns(tokens=rep_tokens, attention=attention_pattern))
# %%
def induction_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    '''
    print(cache)
    result = []
    for i in range(cache.model.cfg.n_layers):
        attn_pattern = cache['pattern', i]
        print(attn_pattern.shape) # n, sQ, sK 
        _, seq_len, _ = attn_pattern.shape
        for j, head in enumerate(attn_pattern):
            attended = head.argmax(dim=1)
            mask = (t.arange(seq_len) - int(seq_len/2) + 1).to(device)
            correct = (attended.int() == mask.int())[int(seq_len/2)+1:].count_nonzero() / (seq_len/2)
            if correct > 0.3: 
                result.append(f"{i}.{j}")
    return result

    # result = [] 
    # for i in range(cache.model.cfg.n_layers): 
    #     attn_pattern = cache['pattern', i]
    #     _, seq_len, _ = attn_pattern.shape
    #     for j, head in enumerate(attn_pattern): 
    #         attended = head.argmax(dim=1)
    #         mask = t.arange(seq_len).to(device).roll(1)
    #         mask[0] = 0
    #         corrects = (attended == mask).count_nonzero() / seq_len
    #         if corrects > 0.35: 
    #             result.append(f"{i}.{j}")
                
    # print(result)
    # return result

print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))
# %%
def hook_function(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint
) -> Float[Tensor, "batch heads seqQ seqK"]:

    # modify attn_pattern (can be inplace)
    print(hook)
    return attn_pattern

utils.get_act_name('pattern', 0) == 'blocks.0.attn.hook_pattern'

loss = model.run_with_hooks(
    tokens, 
    return_type="loss",
    fwd_hooks=[
        # ('blocks.1.attn.hook_pattern', hook_function)
        (lambda name: name.endswith("pattern"), hook_function)
    ]
)
# %%
def hook_all_attention_patterns(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    head_idx: int
) -> Float[Tensor, "batch heads seqQ seqK"]:
    # modify attn_pattern inplace, at head_idx
    return attn_pattern

for head_idx in range(12):
    temp_hook_fn = functools.partial(hook_all_attention_patterns, head_idx=head_idx)
    model.run_with_hooks(tokens, fwd_hooks=[('blocks.1.attn.hook_pattern', temp_hook_fn)])

# %%
seq_len = 50
batch = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)

# We make a tensor to store the induction score for each head.
# We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)


def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    '''
    Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
    '''
    threshold = 0.5
    _, _, seq_len_times_two , _ = pattern.shape
    seq_len = int(seq_len_times_two/2)
    diag = t.diagonal(pattern, offset=-seq_len+1, dim1=2, dim2=3)
    print(diag.shape)
    score = (diag > threshold).count_nonzero(dim=2)/ (seq_len)
    induction_score_store[hook.layer(), t.arange(pattern.shape[1])] = score.mean(dim=0)
    return pattern


pattern_hook_names_filter = lambda name: name.endswith("pattern")

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
model.run_with_hooks(
    rep_tokens_10, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        induction_score_hook
    )]
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store, 
    labels={"x": "Head", "y": "Layer"}, 
    title="Induction Score by Head", 
    text_auto=".2f",
    width=900, height=400
)
# %%

induction_score_store = t.zeros((gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device)

def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_small.to_str_tokens(rep_tokens[0]), 
            attention=pattern.mean(0)
        )
    )

pattern_hook_names_filter = lambda name: name.endswith("pattern")

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
gpt2_small.run_with_hooks(
    rep_tokens_10, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(pattern_hook_names_filter,visualize_pattern_hook), 
               (pattern_hook_names_filter,induction_score_hook), ]
)
imshow(
    induction_score_store, 
    labels={"x": "Head", "y": "Layer"}, 
    title="Induction Score by Head", 
    text_auto=".2f",
    width=900, height=900
)
# YOUR CODE HERE - find induction heads in gpt2_small
# %%
def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"]
) -> Float[Tensor, "seq-1 n_components"]:
    '''
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    '''
    W_U_correct_tokens = W_U[:, tokens[1:]]
    embed = embed[:-1, :]
    l1_results = l1_results[:-1, :, :]
    l2_results = l2_results[:-1, :, :]
    res = einsum("seq d_model, d_model seq -> seq", embed, W_U_correct_tokens)
    l1_logits = einsum("seq nheads d_model, d_model seq -> seq nheads", l1_results,         W_U_correct_tokens)
    l2_logits = einsum("seq nheads d_model, d_model seq -> seq nheads", l2_results, W_U_correct_tokens)
    return t.concat([res.unsqueeze(dim=1), l1_logits, l2_logits], dim=-1)
text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)

with t.inference_mode():
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
    # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
    correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
    t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    print("Tests passed!")
#%% 

embed = cache["embed"]
l1_results = cache["result", 0]
l2_results = cache["result", 1]
logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])

plot_logit_attribution(model, logit_attr, tokens)
# %%
print(logit_attr.shape)

#%%
seq_len = 50

embed = rep_cache["embed"]
l1_results = rep_cache["result", 0]
l2_results = rep_cache["result", 1]
first_half_tokens = rep_tokens[0, : 1 + seq_len]
second_half_tokens = rep_tokens[0, seq_len:]

logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, rep_tokens[0]) 
first_half_logit_attr = logit_attr[:seq_len]
second_half_logit_attr = logit_attr[seq_len:]
assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1), second_half_logit_attr.shape

plot_logit_attribution(model, first_half_logit_attr, first_half_tokens, "Logit attribution (first half of repeated sequence)")
plot_logit_attribution(model, second_half_logit_attr, second_half_tokens, "Logit attribution (second half of repeated sequence)")
# %%
def head_ablation_hook(
    v: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> Float[Tensor, "batch seq n_heads d_head"]:
    v[:, :, head_index_to_ablate, :] = 0
    return v


def cross_entropy_loss(logits, tokens):
    '''
    Computes the mean cross entropy between logits (the model's prediction) and tokens (the true values).

    (optional, you can just use return_type="loss" instead.)
    '''
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()


def get_ablation_scores(
    model: HookedTransformer, 
    tokens: Int[Tensor, "batch seq"]
) -> Float[Tensor, "n_layers n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    '''
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    logits = model(tokens, return_type="logits")
    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(head_ablation_hook, head_index_to_ablate=head)
            # Run the model with the ablation hook
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[
                (utils.get_act_name("v", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            loss = cross_entropy_loss(ablated_logits, tokens)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores


ablation_scores = get_ablation_scores(model, rep_tokens)
tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)
# %% 
imshow(
    ablation_scores, 
    labels={"x": "Head", "y": "Layer"}, 
    title="Loss Difference After Ablating Heads", 
    text_auto=".2f",
    width=900, height=400
)

# %%
layer = 1
head_index = 4

full_OV_circuit = model.W_E @ FactoredMatrix(model.W_V[1, 4], model.W_O[1, 4]) @ model.W_U

tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)

# %%
print(full_OV_circuit.rdim)
rand1 = t.randint(low=0, high=full_OV_circuit.rdim, size=(200,))
full_OV_circuit_sample = full_OV_circuit[rand1, rand1].AB
print(full_OV_circuit_sample)

# YOUR CODE HERE - get a random sample from the full OV circuit, so it can be plotted with `imshow`
imshow(
    full_OV_circuit_sample,
    labels={"x": "Input token", "y": "Logits on output token"},
    title="Full OV circuit for copying head",
    width=700,
)
# %%
def top_1_acc(full_OV_circuit: FactoredMatrix) -> float:
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    rows = full_OV_circuit.rdim # number of cols
    ## get row by doing M[i, :] @ B
    matrix = full_OV_circuit.AB
    return (t.argmax(matrix, dim=-1) == t.arange(rows).to(device)).count_nonzero() / (rows)
    # t.argmax(matrix[col, :] @ full_OV_circuit.B)


print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc(full_OV_circuit):.4f}")
# %%
# W_OV_14 = FactoredMatrix(model.V[1,4], model.O[1,4])
print(model.W_V[1,4].shape)
print(model.W_O[1,4].shape)
W_V_4_10 = t.concat((model.W_V[1,4], model.W_V[1,10]), dim=-1)
W_O_4_10 = t.concat((model.W_O[1,4], model.W_O[1,10]), dim=0)

result = model.W_E @ FactoredMatrix(W_V_4_10, W_O_4_10) @ model.W_U
print(result.shape)

top_1_acc(result) # 0.9556
# %%
def mask_scores(attn_scores: Float[Tensor, "query_nctx key_nctx"]):
    '''Mask the attention scores so that tokens don't attend to previous tokens.'''
    assert attn_scores.shape == (model.cfg.n_ctx, model.cfg.n_ctx)
    mask = t.tril(t.ones_like(attn_scores)).bool()
    neg_inf = t.tensor(-1.0e6).to(attn_scores.device)
    masked_attn_scores = t.where(mask, attn_scores, neg_inf)
    return masked_attn_scores

# layer = 0
# head_index = 7
# W_pos = model.W_pos
# W_QK = model.W_Q[layer, head_index] @ model.W_K[layer, head_index].T
# pos_by_pos_scores = W_pos @ W_QK @ W_pos.T
# masked_scaled = mask_scores(pos_by_pos_scores / model.cfg.d_head ** 0.5)
# pos_by_pos_pattern = t.softmax(masked_scaled, dim=-1)
scores = model.W_pos @ model.W_Q[0, 7] @ model.W_K[0, 7].T @ model.W_pos.T 
masked_scores = mask_scores(scores / (model.cfg.d_head ** 0.5))
pos_by_pos_pattern = masked_scores.softmax(dim=-1)
# YOUR CODE HERE - calculate the matrix `pos_by_pos_pattern` as described above
tests.test_pos_by_pos_pattern(pos_by_pos_pattern, model, layer, head_index)
# %%
print(f"Avg lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")

imshow(
    utils.to_numpy(pos_by_pos_pattern[:100, :100]), 
    labels={"x": "Key", "y": "Query"}, 
    title="Attention patterns for prev-token QK circuit, first 100 indices",
    width=700
)

# %%
def decompose_qk_input(cache: ActivationCache) -> t.Tensor:
    '''
    Output is decomposed_qk_input, with shape [2+num_heads, seq, d_model]

    The [i, :, :]th element is y_i (from notation above)
    '''
    pass

def decompose_q(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_q with shape [2+num_heads, position, d_head]

    The [i, :, :]th element is y_i @ W_Q (so the sum along axis 0 is just the q-values)
    '''
    pass

def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_k with shape [2+num_heads, position, d_head]

    The [i, :, :]th element is y_i @ W_K (so the sum along axis 0 is just the k-values)
    '''
    pass


ind_head_index = 4
# First we get decomposed q and k input, and check they're what we expect
decomposed_qk_input = decompose_qk_input(rep_cache)
decomposed_q = decompose_q(decomposed_qk_input, ind_head_index)
decomposed_k = decompose_k(decomposed_qk_input, ind_head_index)
t.testing.assert_close(decomposed_qk_input.sum(0), rep_cache["resid_pre", 1] + rep_cache["pos_embed"], rtol=0.01, atol=1e-05)
t.testing.assert_close(decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001)
t.testing.assert_close(decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01)
# Second, we plot our results
component_labels = ["Embed", "PosEmbed"] + [f"0.{h}" for h in range(model.cfg.n_heads)]
for decomposed_input, name in [(decomposed_q, "query"), (decomposed_k, "key")]:
    imshow(
        utils.to_numpy(decomposed_input.pow(2).sum([-1])), 
        labels={"x": "Position", "y": "Component"},
        title=f"Norms of components of {name}", 
        y=component_labels,
        width=1000, height=400
    )
