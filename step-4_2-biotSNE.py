import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_word_vectors(file_path):
    with open(file_path, 'r') as f:
        word_vectors = json.load(f)
    words = []
    vectors = []
    for word, vector in word_vectors.items():
        words.append(word)
        vectors.append(vector)
    return words, np.array(vectors)

def filter_biomedical_entities(words, vectors, biomedical_entities):
    filtered_words = []
    filtered_vectors = []
    for word, vector in zip(words, vectors):
        if word in biomedical_entities:
            filtered_words.append(word)
            filtered_vectors.append(vector)
    return filtered_words, np.array(filtered_vectors)

def plot_tsne(word_vectors, words, entity_colors, random_state=42):
    tsne = TSNE(n_components=2, random_state=random_state)
    embeddings_2d = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        x, y = embeddings_2d[i]
        plt.scatter(x, y, color=entity_colors[word])
        plt.text(x, y, word, fontsize=8)

    plt.title("t-SNE Visualization of Biomedical Entities")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

if __name__ == "__main__":
    biomedical_word_vectors_file = "biomedical_word_vectors.json"
    biomedical_entities = ["fever", "cough", "diabetes", ...]  # List of biomedical entity words
    entity_colors = {"fever": "red", "cough": "blue", "diabetes": "green", ...}  # Assign colors based on entity categories
    words, word_vectors = load_word_vectors(biomedical_word_vectors_file)
    words, word_vectors = filter_biomedical_entities(words, word_vectors, biomedical_entities)
    plot_tsne(word_vectors, words, entity_colors)
