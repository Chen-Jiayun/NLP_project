import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# set bio_words list to show
bio_words = ["Lower", "cough", "diabetes", "Guangxi", "Shanghai"]

def load_word_vectors(file_path):
    with open(file_path, 'r') as f:
        word_vectors = json.load(f)
    words = []
    vectors = []
    for word, vector in word_vectors.items():
        words.append(word)
        vectors.append(vector)
    return words, np.array(vectors)

def plot_tsne(word_vectors, words_to_plot=10000, random_state=42):
    tsne = TSNE(n_components=2, random_state=random_state)
    embeddings_2d = tsne.fit_transform(word_vectors[:words_to_plot])

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words[:words_to_plot]):
        # select words that in bio_words
        if word in bio_words:
            x, y = embeddings_2d[i]
            plt.scatter(x, y, color='r')
            plt.text(x, y, word, fontsize=8)

    plt.title("t-SNE Visualization of Word Representations")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

word_vectors_file = "result/step-3-n-gram.json"
words, word_vectors = load_word_vectors(word_vectors_file)
plot_tsne(word_vectors)