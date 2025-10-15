# 对文档进行切片构建文档数据集，包含{id，docs,以及片段来源}
# 构建微调embedding模型所需要的数据集
- 利用大模型对每个文本片段生成问题和答案，更新数据集为{id,docs,query, answer,以及片段来源}
- 生成微调embedding model的数据集   
  1、使用bge基础模型将所有数据向量化
  2、对每个query召回top100
  3、过滤出 0.4 < distance <= 0.7 的数据作为负样本

更新数据集{id,docs,query, answer,pos, neg以及片段来源}


https://blog.csdn.net/qq_44193969/article/details/134042750