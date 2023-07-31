import gensim.downloader

# a N-gram model trained by Facebook AI Research, 300 dimension.
model = gensim.downloader.load('fasttext-wiki-news-subwords-300')

token_path = "test/tokens.txt"

with open(token_path, 'r', encoding='utf-8') as file:
    lines = [next(file).strip() for _ in range(100)]

for line in lines:
    if line in model.key_to_index:
        print(model[line])

sim = model.most_similar("fever")
print(sim)