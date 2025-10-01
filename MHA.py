import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        assert self.hidden_dim % self.n_head == 0
        self.head_dim = self.hidden_dim // self.n_head

        self.wq = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.wk = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.wv = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.att_dropout = nn.Dropout(0.1)

        self.wo = nn.Linear(self.hidden_dim,self.hidden_dim)


    def forward(self,x,attention_mask=None):
        bs, seq_len,_ = x.shape

        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)


        Q = Q.view(bs,seq_len,self.n_head,self.head_dim).permute(0,2,1,3)
        K = K.view(bs,seq_len,self.n_head,self.head_dim).permute(0,2,1,3)
        V = V.view(bs,seq_len,self.n_head,self.head_dim).permute(0,2,1,3)

        att_score = Q @ K.transpose(2,3) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            att_score = att_score + attention_mask
        att_score = F.softmax(att_score,dim =-1)
        att_score = self.att_dropout(att_score)

        output = att_score @ V 
        #这里的 contiguous() 是相当于返回一个连续内存的 tensor，一般用了 permute/tranpose 都要这么操作
        # 如果后面用 Reshape 就可以不用这个 contiguous()，因为 view 只能在连续内存中操作
        output = output.transpose(1,2).contiguous().view(bs,seq_len,-1)

        output  =self.wo(output)
        return output


if __name__ == '__main__':
    mha = MultiHeadAttention(768,12)
    x = torch.randn(1,3,768)
    mask = torch.full((1,1,3,3),float("-inf"))
    attention_mask= torch.triu(mask,diagonal=1)

    output = mha(x,attention_mask)

    print(output.shape)



