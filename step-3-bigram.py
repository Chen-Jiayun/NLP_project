# this method works fine if the data set is small and regular
# but when turns the real article, the model is not good enough
# to support the accuracy...

from gensim.models import Word2Vec
from gensim.models import Phrases
import os

path = "result/step-2-result-split.txt"

corpus = []

def token_generator(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            token = line.strip() 
            corpus.append([token])

token_generator(path)

model_path = "model/word2vec.model"

os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Train a bigram detector.
bigram_transformer = Phrases(corpus)
# Apply the trained MWE detector to a corpus, using the result to train a Word2vec model.
model = Word2Vec(bigram_transformer[corpus], min_count=1)

model.save(model_path)