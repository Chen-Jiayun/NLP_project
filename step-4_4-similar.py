import json
import numpy as np
import heapq

target_word = "COVID-19"
with open('result/step-3-n-gram.json', 'r') as f:
    data = json.load(f)

target_vector = np.array(data[target_word])

top_words = []
# top N similar
heap_size = 10

for word, vector in data.items():
    if word != target_word:
        word_vector = np.array(vector)
        similarity = np.dot(target_vector, word_vector) / (np.linalg.norm(target_vector) * np.linalg.norm(word_vector))
        if len(top_words) < heap_size:
            heapq.heappush(top_words, (similarity, word))
        else:
            heapq.heappushpop(top_words, (similarity, word))

most_similar_words = [(word, similarity) for similarity, word in heapq.nlargest(heap_size, top_words)]

for word, similarity in most_similar_words:
    print(f"'{word}', similarity: {similarity:.4f}")


