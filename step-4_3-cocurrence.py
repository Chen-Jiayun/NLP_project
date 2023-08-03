import json
import numpy as np
import heapq

target_word = "COVID-19"
with open('result/step-3-n-gram.json', 'r') as f:
    data = json.load(f)

target_vector = np.array(data[target_word])

top_words = []
# top N correlated
heap_size = 10

for word, vector in data.items():
    if word != target_word:
        word_vector = np.array(vector)
        correlation = np.corrcoef(target_vector, word_vector)[0, 1]
        if len(top_words) < heap_size:
            heapq.heappush(top_words, (correlation, word))
        else:
            heapq.heappushpop(top_words, (correlation, word))

most_correlated_words = [(word, correlation) for correlation, word in heapq.nlargest(heap_size, top_words)]

for word, correlation in most_correlated_words:
    print(f"'{word}', correlation: {correlation:.4f}")
