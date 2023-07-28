from gensim.models import Word2Vec

path = "test/tokens.txt"

corpus = []

def token_generator(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            token = line.strip() 
            corpus.append([token])

token_generator(path)

model = Word2Vec(corpus, vector_size=256, window=5, min_count=1, workers=4)

word_vector = model.wv["Digital"]
print(word_vector)








