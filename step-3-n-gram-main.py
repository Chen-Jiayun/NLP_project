import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ngram_model
from ngram_model import NGramLanguageModeler



# build a list of tuples.
# Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
ngrams = [
    (
        [ngram_model.corpus[i - j - 1] for j in range(ngram_model.CONTEXT_SIZE)],
        ngram_model.corpus[i]
    )
    for i in range(ngram_model.CONTEXT_SIZE, len(ngram_model.corpus))
]
# Print the first 3, just so you can see what they look like.
print(ngrams[:3])

vocab = set(ngram_model.corpus)
word_to_ix = {word: i for i, word in enumerate(vocab)}


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), ngram_model.EMBEDDING_DIM, ngram_model.CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in ngrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!

# To get the embedding of a particular word, e.g. "beauty"
print(model.embeddings.weight[word_to_ix["COVID-19"]])

import os

# 假设 n-gram-model 目录存在或可以被创建
save_directory = 'n-gram-model'

# 创建目录（如果不存在）
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# 保存模型到 n-gram-model 目录
save_path = os.path.join(save_directory, 'my_model.pth')
torch.save(model.state_dict(), save_path)