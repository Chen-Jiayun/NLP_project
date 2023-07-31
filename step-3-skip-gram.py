import gensim.downloader

# a skip gram model trained by Stanford, 200 dimension.
glove_vectors = gensim.downloader.load('glove-twitter-25')

token_path = "test/tokens.txt"

with open(token_path, 'r', encoding='utf-8') as file:
    lines = [next(file).strip() for _ in range(100)]

for line in lines:
    if line in glove_vectors.key_to_index:
        print(glove_vectors[line])

sim = glove_vectors.most_similar("fever")
print(sim)
