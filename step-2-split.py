sep = [',', ':', '.', ';']
file_name = "titles.txt"

def extract_words(article, separators):
    for separator in separators:
        article = article.replace(separator, " ")
    words = article.split()
    return words

with open(file_name, "r") as file:
        for line in file:
            ret = extract_words(line, sep)
            for word in ret:
                print(word)