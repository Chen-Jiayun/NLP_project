import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 256

# 10k token of all
path = "test/tokens.txt"
corpus = []

def corpus_generator(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            token = line.strip() 
            corpus.append(token)

corpus_generator(path)


class SkipgramModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(SkipgramModel, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim) 
    
    def forward(self, input_word, context_word):
        # computing out loss
        emb_input = self.target_embeddings(input_word)     
        emb_context = self.context_embeddings(context_word)  
        return emb_input, emb_context