## 对文档进行切片构建文档数据集，包含{id，docs,以及片段来源}
## 构建微调embedding模型所需要的数据集
- 利用大模型对每个文本片段生成问题和答案，更新数据集为{id,docs,query, answer,以及片段来源}
- 生成微调embedding model的数据集   
- 
  1、使用bge基础模型将所有数据向量化（模型选用./BAAI/bge-base-zh-v1.5  ./BAAI/bge-reranker-base）

  2、对每个query召回top15

  3、过滤出 0.4 < distance <= 0.7 的数据作为负样本（实际操作为对检索到的文档进行rerank,取前3作为正样本，然后从剩下的文档中随机采样5个作为负样本）

更新数据集{id,docs,query, answer,pos, neg以及片段来源}


## 评估BAAI/bge-base-zh-v1.5的指标  使用SentenceTransformer的 InformationRetrievalEvaluator
- 评估指标 recall, mrr,recall是召回率，计算的是每个问题的相关文档被召回概率的平均。 mrr计算的是第一个相关文档的倒数排名，不关心其他文档（当相关文档有多个的时候）
- 评估数据集的格式
  
```
datasets={
    "corpus":[{uuid1:doc1},{uuid2:doc2},{uuid3:doc3}       #对应的文本id、文本
    ],
    "queries":[{uuid1:问题}，{uuid2:问题}，...
    ],
    "relevant_docs":[{uuid1:[uuid答案]},{uuid2:[uuid答案]},{uuid3:[uuid答案]}
    ]
}
```

```
{
    "queries": {
        "7813f025-333d-494f-bc14-a51b2d57721b": "日本半导体产业的现状和影响因素是什么？",
        ...
    },
    "corpus": {
        "node_98": "日本半导体产业在上世纪80年代到达顶峰后就在缓慢退步，但若简单认为日本半导体产业失败了，就是严重误解，今天日本半导体产业仍有非常有竞争力的企业和产品。客观认识日本半导体产业的成败及其背后的原因，对正在大力发展半导体产业的中国，有非常强的参考价值。",
        ...
    },
    "relevant_docs": {
        "7813f025-333d-494f-bc14-a51b2d57721b": [
            "node_98"
        ],
        ...
    }
}
```
   
# 微调embedding模型
 - 使用sentence transformer 微调embedding模型
SentenceTransformer 多负样本排序损失函数（MultipleNegativesRankingLoss）是一种适用于语义检索和信息召回任务的损失函数。它的主要优点在于不需要构造负样本，
因为该损失函数会将一个批次中的所有非正样本作为负样本，从而在最终结果的概率分布上，正样本的概率高于其他负样本。


https://blog.csdn.net/qq_44193969/article/details/134042750


## 使用ragas进行rag系统的评估
- 数据集格式
  
```
    eval_dataset = Dataset.from_dict({
    "question": ["What is the capital of France?"],
    "contexts": [["Paris is the capital of France."]],
    "answer": ["The capital of France is Paris."],
    "ground_truths": [["Paris is the capital of France."]]
    })
```
  
- 评估指标


## 也可以使用llm进行评估,如只在回答准确度指标上进行评估，利用大模型输出分数，然后取平均：（分数-1）/ 4

```
EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""
```
  
  