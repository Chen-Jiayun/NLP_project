import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 256

# 10k token of all
path = "test/tokens.txt"
corpus = []

def corpus_generator(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            token = line.strip() 
            corpus.append(token)

corpus_generator(path)


class SkipgramModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(SkipgramModel, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim) 
        
        # 初始化embedding参数
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)


    def forward(self, target, context):
        """
        Args:
            target: 目标词的索引，形状为(batch_size,)
            context: 上下文词的索引，形状为(batch_size, window_size*2)

        Returns:
            loss: 训练损失
        """
        # 获取目标词的词向量
        target_embeds = self.in_embed(target)

        # 获取上下文词的词向量
        context_embeds = self.out_embed(context)

        # 将目标词和上下文词的词向量进行点积计算
        scores = torch.bmm(context_embeds, target_embeds.unsqueeze(2)).squeeze(2)

        # 使用负对数似然损失函数计算损失
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(scores, torch.ones(scores.shape[0], dtype=torch.long).to(scores.device))

        return loss
