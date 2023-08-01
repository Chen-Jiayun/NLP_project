import torch
import torch.nn as nn
import torch.nn.functional as F

# 10k token of all
path = "test/tokens.txt"
corpus = []

def corpus_generator(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            token = line.strip() 
            corpus.append(token)

corpus_generator(path)