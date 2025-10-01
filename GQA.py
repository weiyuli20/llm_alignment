import math
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
MHA和MQA是GQA的特殊形式
n_kv_head = n_head 时就是MHA
n_kv_head = 1 时就是MQA

'''

class GroupedQueryAttention(nn.Module):
    def __init__(self,hidden_dim, n_head, n_kv_head):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        assert self.hidden_dim % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0
        self.head_dim = self.hidden_dim // self.n_head

        self.wq = nn.Linear(self.hidden_dim, self.n_head * self.head_dim)
        self.wk = nn.Linear(self.hidden_dim, self.n_kv_head * self.head_dim)
        self.wv = nn.Linear(self.hidden_dim, self.n_kv_head * self.head_dim)

        self.wo = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.att_dropout = nn.Dropout(0.1)

    def forward(self,x,attention_mask =None):
        bs,seq_len,_ = x.shape

        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        Q = Q.view(bs,seq_len,self.n_head,self.head_dim).transpose(1,2)
        K = K.view(bs,seq_len,self.n_kv_head,self.head_dim).transpose(1,2)
        V = V.view(bs,seq_len,self.n_kv_head,self.head_dim).transpose(1,2)

        #广播
        K = K.repeat_interleave(self.n_head // self.n_kv_head, dim = 1)
        V = V.repeat_interleave(self.n_head // self.n_kv_head, dim = 1)

        att_score = Q @ K.transpose(2,3) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            att_score += attention_mask

        att_score = self.att_dropout(F.softmax(att_score,dim=-1))

        output = att_score @ V

        output = output.transpose(1,2).contiguous().view(bs,seq_len,-1)

        output = self.wo(output)

        return output
    
if __name__ == '__main__':
    x = torch.rand(1,3,768)
    attention_mask = torch.full((1,1,3,3),float("-inf"))
    attention_mask = torch.triu(attention_mask,diagonal=1)
    gqa =GroupedQueryAttention(768,12,3)
    output  =gqa(x,attention_mask)
    print(output.shape)


         