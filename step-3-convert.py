from gensim.models import Word2Vec

# 示例输入语料，每个元素是一个句子（已经分词为token）
corpus = [
    ["the", "main", "symptoms", "of", "COVID-19", "are", "fever", "and", "cough"],
    ["word", "embedding", "is", "useful", "for", "NLP", "tasks"]
]

# 使用Word2Vec训练词向量模型
model = Word2Vec(corpus, vector_size=256, window=5, min_count=1, workers=4)

# 获取单词"fever"的向量表示
word_vector = model.wv["fever"]
print(word_vector)




# 已经得到token和vector表示
token = [...]
vector_of_token = [...]

def N_gram():
    # 这个模型该怎么样训练？
    return 

def skip_gram():
    # 这个模型该怎么训练？
    return 








