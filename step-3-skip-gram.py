import skip_gram_model
from skip_gram_model import SkipgramModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

import random


def generate_skipgrams(sentence, context_size, num_negative_samples):
    skipgrams = []
    for i in range(context_size, len(sentence) - context_size):
        target_word = sentence[i]
        context_words = sentence[i - context_size :i ] + sentence[i + 1 : i + context_size + 1]
        for context_word in context_words:
            # 1 indicates a positive sample
            skipgrams.append((target_word, context_word, 1))

            # Negative sampling: random select words that are not in the context
            for _ in range(num_negative_samples):
                negative_word = random.choice(sentence)
                while negative_word in context_words:
                    negative_word = random.choice(sentence)
                # 0 indicates a negative sample
                skipgrams.append((target_word, negative_word, 0))
    return skipgrams


CONTEXT_SIZE = 2
NUM_NEGATIVE_SAMPLES = 5

skipgrams = generate_skipgrams(
    skip_gram_model.corpus, CONTEXT_SIZE, NUM_NEGATIVE_SAMPLES)


vocab = set(skip_gram_model.corpus)
word_to_ix = {word: i for i, word in enumerate(vocab)}
losses = []
loss_function = nn.BCEWithLogitsLoss()
model = SkipgramModel(len(vocab), skip_gram_model.EMBEDDING_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for target_word, context_word, label in skipgrams:
        # Prepare the inputs to be passed to the model
        target_idx = torch.tensor(word_to_ix[target_word], dtype=torch.long)
        context_idx = torch.tensor(word_to_ix[context_word], dtype=torch.long)

        # Zero out the gradients
        model.zero_grad()

        # Forward pass
        target_embeds, context_embeds = model(target_idx, context_idx)
        similarity_score = torch.matmul(
            target_embeds, context_embeds.t())  # Calculate the dot product

        # Calculate the loss and perform backpropagation
        score = similarity_score.item()  # Get the scalar value from the tensor
        loss = loss_function(torch.tensor(
            [score]), torch.tensor([label], dtype=torch.float))
        total_loss += loss.item()
    losses.append(total_loss)


word_vectors = {}
for i in skip_gram_model.corpus:
    word = i
    vector = model.target_embeddings.weight[word_to_ix[i]].tolist()
    word_vectors[word] = vector

# result save
json_output = json.dumps(word_vectors, indent=2)
print(json_output)

# for persistent
import os

save_directory = 'skip-gram-model'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

save_path = os.path.join(save_directory, 'my_model.pth')
torch.save(model.state_dict(), save_path)