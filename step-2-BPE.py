from tokenizers import ByteLevelBPETokenizer

file_name = "titles.txt"

def extract_words(article):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator([article])
    words = tokenizer.encode(article).tokens
    return words

with open(file_name, "r") as file:
        for line in file:
            ret = extract_words(line)
            for word in ret:
                print(word)