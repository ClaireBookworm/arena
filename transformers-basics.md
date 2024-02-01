# Transformers from scratch

- predicting the next token → generating text + continuously sample
- language turned into tokens (tokenization) and map each token to a vector via an embedding matrix → lookup table, learned by the model
    - matrix of weights learned by gradient descent
    - # of embedding vectors? i.e. separating a word into multiple tokens
- how would u tokenize?
    - ascii (loses language structure)? just pure words, like english dictionary? (limit ourselves) therefore they use byte-pair encodings
    - algo that starts with ascii-256 and create new words by combining two tokens you already have that appear together commonly (i.e. letter t and h appear tgt, or thank you).

transformers are trained on cross-entropy loss (understand this?)

- cross-entropy loss: measures the performance of a classification model whose output is a probability value between 0 and 1 → increases as the predicted probability diverges
    - entropy is the average number of bits required to represent or transmit an event drawn from the probability distribution for the random variable
    - cross entropy between 2 discrete probability distributions is related to kullback-leibler (kl) divergence → metric that captures how similar 2 distributions are
- minimizing kl-divergence https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence
- trying to be a better model of the “true probabilities”

architecture

[Whiteboarding made easy](https://app.excalidraw.com/l/9KwMnW35Xt8/4kxUsfrPeuS)

- **residual stream** — central component of the transformer. incrementally add to it / transformer operates on it
    - added vectors in each mlp-attention head layer is much smaller
    - norm of residual stream does grow over time but not notably
- information can only flow forward (no cheating)
- finally, linear map on the final residual stream → map from the length of the vector in the residual stream (d_model) to d_vocab (# of items in the vocab) → vector of logits of probability distributions → softmax?? into actual words

- can train model to predict the 1, 2,… etc words in a sequence at the same time because we’re using cross-entropy loss and don’t let the model look backward (how is this possible?)
    - adding a value at the end of the sequence is ok because the other vectors will stay the same (no looking back)

inputs into the attention heads layer is directly from the residual stream and **adds** (just vectors sum?) it directly back to the residual stream 

MLPs — standard 2 layer neural network; operates on each position in the same way, doesn’t move info around → you can separate the MLP out into independent individual neurons (= each individual output) and when you add them you get the actual output of the mlp layer 

- output is the sum of output of all neurons
- each neuron has asssociated input (feature the neuron is looking for), output (vector written to the residual stream if the input feature if present)

hidden layers are larger than the input/output  (4x res stream) 

![Screenshot 2024-01-15 at 10.34.07 AM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/f2f3e1b3-4f39-4187-b201-2e19ac213bf9/bb24696c-1c5e-40f5-aeba-26b7b95cb7c6/Screenshot_2024-01-15_at_10.34.07_AM.png)

**attention** — more complicated; operates on each position in the same way and moves info between positions in the sequence

- a bunch of attention heads, each working in parallel, they get summed at the end
- move sequence positions from specific positions x to y → there is no weight matrix in this model (?)
- dot product keys and queries to create a matrix of attention scores (dot prod of every pair of key and query vectors) → mask wherever key position > query position — scale by 1/sqrt d_head & softmax
- bilinear function (if you see keys queries independent, but it might be closer to quadratic function); all the functions in res stream attention head layer is linear except softmax
- instead of imposing a prior of saying only the last x tokens are important, we dedicate keys and queries to determine which tokens/inthis position are very important. and later, we can look back, at like, a name, and remember that that name is important vs. filler words
    - indirect object id → use attention head to attend to the very first instance of an object to move that information forward in the model to make a prediction for what’s next (name_of head; does something similar to this)
- attention probabilities is the amounts that each query token pays attention to each key token
    - “the cat sat” → each word has attention probabilities. the % between “the” and “cat” is like how much information from “the” the word “Cat” is using.
        
        ![Screenshot 2024-01-15 at 10.46.56 AM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/f2f3e1b3-4f39-4187-b201-2e19ac213bf9/1c152195-5395-4d6e-8e19-8d44aed71cf5/Screenshot_2024-01-15_at_10.46.56_AM.png)
        

**positional embedding matrix**

- positional encoders!
- start model with sum of the vectors representing a token and a vector representing position (like simplest version is like 1, 2, 3rd word) and have a separate lookup table for that
- the model just somehow learns how to interp these 2 discrete forms of information even if they are added together (adding → how do you know is from the position vs just the inherent value?)

max tokens to put in the model at once is 4096 → learned that many embeddings and tokens; choose not to train past a certain point & then any more input would be unknown bc the model doesn’t know those embeddings 

long context windows 

*********************************************************************************tokens - transformer inputs*********************************************************************************

- make a massive lookup table which is called an embedding → has one vector for each possible sub-unit language we might get (call this the **********vocabulary**********) — we label each every element in our vocabulary with an integer (this labelling never changes does this mean like when we initially set up the model?) and we used this integer to index into the embedding matrix
- tokenizing and embedding is totally different
- differnet models have different tokenizers but it is mostly hard coded
    - gpt-2’s tokenizer:
    - when you do
    
    ```python
    sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])
    print(sorted_vocab[:20])
    print() # seems like len(sorted_vocab) is 50257
    print(sorted_vocab[250:270])
    print()
    print(sorted_vocab[1500:1505]) # 
    # [('Ġconst', 1500), ('arn', 1501), ('Ġorder', 1502), ('AR', 1503), ('ior', 1504) ]
    ```
    

**********************************one-hot encodings**********************************

- vectors with zeros everywhere, except for a single one in the position corresponding to the word’s index in the vocab
- indexing into the embedding is = multiplying the **embedding matrix by the one-hot encoding** (where the embedding matrix is the matrix we get by stacking all the embedding vectors on top of each other)

**from exercises**

************************************first-formed 3, 4, 5, 6, 7 letter encodings in gpt-2 vocab is************************************

```python
lengths = dict.fromkeys(range(3, 8), "")
for tok, idx in sorted_vocab:
    if not lengths.get(len(tok), True):
        lengths[len(tok)] = tok

for length, tok in lengths.items():
    print(f"{length}: {tok}")
```

<aside>
💡 the input into a transformer is a sequence of ************tokens************ (integers) not vectors

</aside>

- whether a word begins with a capital or space matters
- arithmetic is a mess → length is inconsistent

****************text gen order****************

1. convert text to tokens `tokens = reference_gpt2.to_tokens(ref_text).to(device)`
2. map tokens to digits `logits, cache = reference_gpt2.run_with_cache(tokens)`
3. convert the logits to a distribution with a softmax `probs = logits.softmax(dim=-1)`
4. map distribution to a token `next_token = logits[0, -1].argmax(dim=-1)` and `next_char = reference_gpt2.to_string(next_token)`. this gives us the prediction for the last token in the input sequence

predicting the next token: 

```python
most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])
print(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))

# [('<|endoftext|>', '\n'), ('I', "'m"), (' am', ' a'), (' an', ' avid'), (' amazing', ' person'), (' aut', 'od'), ('ore', 'sp'), ('gressive', '.'), (',', ' and'), (' dec', 'ently'), ('oder', ','), ('-', 'driven'), ('only', ' programmer'), (',', ' and'), (' G', 'IM'), ('PT', '-'), ('-', 'only'), ('2', '.'), (' style', ','), (' transformer', '.'), ('.', ' I'), (' One', ' of'), (' day', ' I'), (' I', ' will'), (' will', ' be'), (' exceed', ' my'), (' human', 'ly'), (' level', ' of'), (' intelligence', ' and'), (' and', ' I'), (' take', ' over'), (' over', ' the'), (' the', ' world'), (' world', '.'), ('!', ' I')]
```

now we can calculate the words that are predicted

```python
print(f"Sequence so far: {reference_gpt2.to_string(tokens)[0]!r}")

for i in range(10):
    print(f"{tokens.shape[-1]+1}th char = {next_char!r}")
    # Define new input sequence, by appending the previously generated token
    tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits = reference_gpt2(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = reference_gpt2.to_string(next_token)
```

Sequence so far: '<|endoftext|>I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!'
36th char = ' I'
37th char = ' am'
38th char = ' a'
39th char = ' very'
40th char = ' talented'
41th char = ' and'
42th char = ' talented'
43th char = ' person'
44th char = ','
45th char = ' and'

## attention

- move information from prior positions in the sequence to the current token → do this for every token in parallel using the same parameters
- the w_k w_q and w_v weight matrices are learned → keys queries and values are the **********intermediate matrices on a forward pass**********
- attention is like generalized convolution → imposing a prior of locality (the assumptions that pixels that are close together are more likely to share info — language has some locality, the picture is a lot more nuanced because which tokens are relevant depends on the context of sentence). — attention layers are a way to not impose prior of locality **but instead develop your own algorithm**
- made up of `n_heads` heads — each with their own parameters, own attention pattern, and own information how to copy things from source to destination
    - each head produces an **attention pattern for each destination token**, a probability distribution of prior source tokens (including the current one) weighting how much info to copy
    - moves info (via a linear map) in the same way from each source token to each destination token
- each attention head has ********************************************two different circuits********************************************
    - one determines where to move information to and from (function of the residual stream for the source and destination tokens) — **********************called QK circuit**********************
    - other determines what information to move (function of only the source token’s residual stream) — ********************OV circuit********************
- analogy;
    - person in line → residual stream; they start with just knowing what their own word is (token embedding) and position in thel ine (positional embedding)
    - queries → question; asked to eveyrone behind them in line
    - keys → whether people hold certain information (interaction between keys and queries determins who replies)
    - values → the info that the people who reply pass back to the person who originally asked the question

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f2f3e1b3-4f39-4187-b201-2e19ac213bf9/311414c2-582a-4f7f-9781-e2f006123edf/Untitled.png)

- logit lens
    
    [interpreting GPT: the logit lens — LessWrong](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
    
    The logit lens is a technique described in this webpage for interpreting the internal representations of GPT language models at different layers of the network. It works by taking the activations from an intermediate layer and projecting them back into the model's vocabulary space using the same matrix that projects inputs and outputs. Even though these intermediate representations are from the middle of the network, they often make intuitive sense when interpreted as probability distributions over the next token. This provides insight into what the model is "thinking" or believing at different stages of processing. The key things the logit lens reveals according to the webpage are:
    
    - The model forms reasonable preliminary predictions very early on, like predicting "enormous" as the next word after "GPT-3, an".
    - These preliminary predictions are gradually refined as more layers are processed, converging on the final output distribution.
    - In contrast to what might be expected, the model does not seem to "keep the inputs around" or gradually transform them. The inputs are immediately converted to an internal representation focused on predicting the output.
    
    So in summary, the logit lens is a technique for interpreting intermediate layers of GPT models by projecting their activations back into vocabulary space and analyzing the resulting probability distributions over the next token. It provides insights into how the model's beliefs are formed and refined during processing.
    
- **background on GPTs structure**
    
    You can skip or skim this if you already know it.
    
    - Input and output
        - As *input,* GPT takes a sequence of tokens. Each token is a single item from a vocabulary of *N_v*=50257 byte pairs (mostly English words).
        - As *output,* GPT returns a probability distribution over the vocabulary. It is trained so this distribution predicts the next token.
        - That is, the model's outputs are shifted forward by one position relative to the inputs. The token at position *i* should, after flowing through the layers of the model, turn into the token at position *i+1*. (More accurately, a distribution over the token at position *i+1.*)
    - Vocab and embedding spaces
        - The vocab has size $N_v=50257$, but GPT works internally in a smaller "**embedding**" vector space, of dimension *N_e*.
            - For example, in the GPT-2 1558M model size, $N_e=1600$. (Below, I'll often assume we're talking about GPT-2 1558M for concreteness.)
        - There is an *N_v* by -*N_e* embedding matrix *W* which is used to project the vocab space into the embedding space and vice versa.
    - In, blocks, out
        - The first thing that happens to the inputs is a multiplication by *W*, which projects them into the embedding space. [1]
        - The resulting 1600-dimensional vector then passes through many neural network blocks, each of which returns another 1600-dimensional vector.
        - At the end, the final 1600-dimensional vector is multiplied by *W's* transpose to project back into vocab space.
        - The resulting 50257-dim vectors are treated as logits. Applying the softmax function to them gives you the output probability distribution.

![Screenshot 2024-01-15 at 12.44.15 PM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/f2f3e1b3-4f39-4187-b201-2e19ac213bf9/1d8886c3-ed54-48c8-8bdb-a9f98e874ebe/Screenshot_2024-01-15_at_12.44.15_PM.png)

## mlp

- mlp operates on positions in the residual stream independently and in the same way — doesn’t movei nfo between positions
- by definition: matmul, non-linear distro, matmul, nonlinear… etc
    - “fully connected feed-forward artificial neural network with at least three layers (in, out, and >1 hidden layer)
    - in this case we want 1 hidden layer. mlps can approximate anything as long as theres 1 layer
    - balance between work in the attention and the mlp (mlp already take up most of the computation, matmul takes a lot of the work + parameter count is high) so we just stick with one layer
- once attention has moved relevant info to a single position, the MLPs can actually **************do computation************** —
- mlps as knowledge storage — memory menagement; we might find that the i-th neuron satisfies $W^{in}_{[:,i]} = - W^{out}_{[i,:]} = \vec v$ for some unit vector v, meaning it might be responsible for therasing the positive component of the vector x in the direction v

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f2f3e1b3-4f39-4187-b201-2e19ac213bf9/1293efa8-20c3-4ed9-9e8d-8d7448fffe97/Untitled.png)

**tied embeddings**

- same weights for We and Wu matrices → to get the logit score for a particular token at some sequence position, we just take the vector in the residual stream at that sequence position and take the inner product with the corresponding token embedding vector

layernorm

- simple norm function applied at the start of each layer
- converts each input vector (independently in paralle for each batch x position residual stream vector) to have mean 0 and variance 1
- elementwise scaling and translation (just a linearmap — layernorm is only applied immediately before another linear map (linear compose linear = lilnear, so we can fold this into a single effectivel inear layer — `from_pretrained` )
- annoying for interp — scale part is nonlinear but almost so (if you’re changing a small part of input it’s linear

**positional embeddings**

- problem — attention operates over all pairs of positions — symmetric with regards to position; attention calculatiom from token 5 to token 1 and token 5 to 2 are the same — dumb; nearby tokens are more relevant
- dumb hacks → ****************************************************************************learned absolute positional embeddings**************************************************************************** — learn a lookup table mapping the index of the position of each token to a residual stream vecotr and add (not concat; because residual stream is shared memory, and likely under significant superposition — never concat in a transformer)
- connected to attention as **********************************************generalized convolution********************************************** — language still has locality, and helpful to have a ccess

parameters and activations 

- parameters are weights and biases that are learned during training
- activations - temporary numbers calculated during a forward pass — functions of the input
    - think of these values as only existing for the duration of a single forward pass, disappearing afterwards
    - use hooks to access these values during forward pass but doesn’t make sense to talk about the model’s activations outside the context of some input
    - attention scores and patterns are activations

```python
# print all activation shapes of reference model
for activation_name, activation in cache.items():
    # Only print for first layer
    if ".0." in activation_name or "blocks" not in activation_name:
        print(f"{activation_name:30} {tuple(activation.shape)}")
```

```python
# print all parameters shapes of reference model 
for name, param in reference_gpt2.named_parameters():
    # Only print for first layer
    if ".0." in name or "blocks" not in name:
        print(f"{name:18} {tuple(param.shape)}")
```

The names **keys**, **queries** and **values** come from their analogy to retrieval systems. Broadly speaking:

- The **queries** represent some information that a token is **"looking for"**
- The **keys** represent the information that a token **"contains"**
    - So the attention score being high basically means that the source (key) token contains the information which the destination (query) token **is looking for**
- The **values** represent the information that is actually taken from the source token, to be moved to the destination token

**notes from more readings**

Dot products are especially useful when we're working with our one-hot word representations. The dot product of any one-hot vector with itself is one. (0 1 0 0) * (0 1 0 0) = (0 1 0 0) ⇒ 1

And the dot product of any o ne-hot vector with any other one-hot vector is zero. (0 1 0 0) * (0 0 0 1) → 0 0 0 0 ⇒ 0

The previous two examples show how dot products can be used to measure similarity. As another example, consider a vector of values that represents a combination of words with varying weights. A one-hot encoded word can be compared against it with the dot product to show how strongly that word is represented.
