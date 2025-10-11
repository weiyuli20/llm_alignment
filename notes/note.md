# https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-3-reward-modeling-human-preferences/learning-from-preferences

## 大模型解码策略
大模型解码策略

1.贪心解码：每次选择概率最大的token   会导致生成序列重复

2.随机采样：随机选择一个token

3.top-k: 将输出概率降序排列，保留前k个，然后将这个k个进行概率归一化，然后从这K个中随机采样

4.top-p: 将输出概率降序排列,累加概率直至累加概率和>p, 将这些概率进行归一化，然后从这个范围中进行随机采样。  不用指定k值，可以根据p动态确定采样的token数量

重复惩罚：重复惩罚是指在解码时降低已经出现过的token的概率。重复惩罚系数一般是一个大于1的值。

温度系数控制大模型生成的创造性。温度系数可以修改logits分布，当温度小于1时，logits 除以温度后，分布曲线会变得尖锐，温度越低，分布曲线会越尖锐，概率高的token的概率会越大；当温度大于1时，温度越高，分布曲线越平缓，每个token的概率趋近相同，因此每个token被选中的概率差别不大

温度系数是作用于logits，修改logits的分布。




## BCELoss

BCELoss 二分类交叉熵损失
标签：0、1

计算公式：$L=-\sum_{1}^{n} y_i \log (p_i) + (1- y_i)\log(1-p_i)$,
其中y是真实标签，p是预测概率

当真实类别y=1时，公式中只剩下$\log (p_i)$
当真实类别y=0时，公式中只剩下$\log (1-p_i)$

该损失对错误敏感，预测越接近真相，损失越小，反之，预测越偏离真相，损失越大

## RM Model的训练过程
利用SFT模型生成多个候选响应，由人类对这些响应排名，训练一个奖励模型。

奖励模型的目标是学习一个函数$r(x,y)$,其中，x是prompt,y是大模型输出的response,$r(x,y)$输出是一个标量奖励值r.

训练时，RM需要保证：如果人类认为$(y_1 > y_2)$,则$r(x,y_1) > r(x,y_2)$

RM的训练基于成对偏好数据


## RM 损失函数推导

RM的训练基于成对偏好数据，常用Bradley-Terry模型或对比损失

Bradley-Terry模型 用于成对比较：对于两个响应$y_1$和$y_2$,人类偏好($y_1$ 优于 $y_1$)可以用概率表示：

$P(y_1 >y_2) = \frac{\exp(r(x,y_1))}{\exp(r(x,y_1)) + \exp(r(x,y_2))}$ 

损失函数是负对数似然

$L = -\log(\frac{\exp(r(x,y_1))}{\exp(r(x,y_1)) + \exp(r(x,y_2))})$

简化后有:
$L = -\log(\sigma(r(x,y_1)- r(x,y_2)))$

其中$\sigma$是sigmoid函数，$r(x,y_1)- r(x,y_2)$是奖励差值

推导过程：
$L = -\log(\frac{\exp(r(x,y_1))}{\exp(r(x,y_1)) + \exp(r(x,y_2))})=-\log(\frac{1}{1 + \exp(r(x,y_2)-r(x,y_1))}) = -log(\frac{1}{1 + \exp-(r(x,y_1)-r(x,y_2))}) = -log(\sigma(r(x,y_1)-r(x,y_2)))$



$sigmoid(x) = \frac{1}{1+\exp(-x)}$

损失计算是使用BCELoss,target为1,直观理解$r(x,y_1)-r(x,y_2)$的差值越大($\sigma$)越接近1，损失越小



https://blog.csdn.net/shizheng_Li/article/details/145947974


## 交叉熵损失
交叉熵损失用于计算多分类损失

真实标签为one hot编码：【0，0，1，0】

计算公式为: $L =-\sum_{1}^{n} y_i log(p_i)$
