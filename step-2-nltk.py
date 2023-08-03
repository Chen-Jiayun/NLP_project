import nltk
from nltk.tokenize import word_tokenize

file_name = "result/step-1-raw.txt"

def extract_words(article):
    words = word_tokenize(article)
    return words

with open(file_name, "r") as file:
        for line in file:
            ret = extract_words(line)
            for word in ret:
                print(word)