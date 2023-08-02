import skip_gram_model
from skip_gram_model import SkipgramModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def skip_gram(window_size=2):
   skip_grams = []
   c = skip_gram_model.corpus
   for i in range(window_size, len(c) - window_size):
      target = c[i]
      context = c[i - window_size : i]
      context += (c[i + 1 : i + window_size + 1])
      skip_grams.append((target, context))
   return skip_grams


vocab = set(skip_gram_model.corpus)
vocab_size = len(vocab)

skip_gram_list = skip_gram()

word_to_ix = {word: i for i, word in enumerate(vocab)}

model = SkipgramModel(len(vocab), skip_gram_model.EMBEDDING_DIM)

optimizer = optim.SGD(model.parameters(), lr=0.001)