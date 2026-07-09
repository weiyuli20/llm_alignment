模型选型：

- 一开始锁定中等规模参数的模型14b,32b
- 中文场景
- 根据qwen3技术报告， 报告中详细评估了qwen2.5-14b-base, qwen2.5-32b-base,qwen3-14b-base,发现qwen2.5-32b-base,qwen3-14b-base性能接近，但是14b参数量更少。
  <img width="960" height="664" alt="image" src="https://github.com/user-attachments/assets/6b776de8-f52c-45fa-8a77-5fcf73fa0c29" />

  <img width="998" height="387" alt="image" src="https://github.com/user-attachments/assets/e80c6050-ea8f-455b-9dc7-ddc2a3c9c2ac" />

  最终选定qwen3-14b(经过post-training)的版本， 后续可以跑base模型进行对比（待完成）


  关于一些评测benchmark
  - MMLU
  - MMLU-Redux
  - MMLU-Pro
  - SuperGPQA
  - BBH
  - GPQA-Diamond
  - C-Eval
  - LiveBench
 
  微调显存估计：
  14b-lora
  -  模型权重 28G
  -  梯度：0.5G
  -  优化器状态： 1.5G
  -  激活值：跟batch_size和序列长度正相关的一个量
 
  batch_size为1，就是一份激活值；为n,就是n份激活值
  
  序列长素度2 注意力矩阵是2*2， 序列长素是n,注意力矩阵是n *n
 
  开启梯度检查点之后，max_seq_len=2048时，1条样本前向传播的激活值约为2G, batch_size 为4时约为8G (不开启梯度累积)

  开启梯度检查点之后，max_seq_len=2048时，1条样本前向传播的激活值约为2G, batch_size 为4 ,梯度累积步数为4时激活值约为2G (开启梯度累积)

  相当于一次计算一条样本，该样本的激活值在计算完梯度之后，只保留梯度，丢弃激活值

  单卡40G  batchsize设置为16，梯度累计8，max_seqlen=2048 单次前向传播显存占用28(权重）+2（梯度优化器）+4 （激活值）+3
  

  
  14b 全参微调
  -  模型权重 28G
  -  梯度：28G
  -  优化器状态：56G
  -  激活值：跟batch_size和序列长度正相关的一个量
  
 

  # 数据集（medical-gpt)
  数据集最终混合配方；在最终送入训练的 train.json 中，你的数据结构应该是：
  - 55% 垂直领域单轮中文数据  https://huggingface.co/datasets/shibing624/medical/tree/main/finetune。  40k 
  - 25% 垂直领域多轮中文数据
  - 10% 通用英文高质量数据（单/多轮均可）
  - 10% 通用中文对话数据（用于保持通用聊天能力）share-gpt-chinese 共90k, 是中英双语的，从中抽取10k
    - 中英文7:3
    - 50%的多轮对话，选择对话轮次大于2的
    - 过滤低质量样本 （对话回复在30-500）之间
   
    ```
    export HF_ENDPOINT=https://hf-mirror.com
    huggingface-cli download --repo-type dataset --resume-download FreedomIntelligence/sharegpt-chinese --local-dir FreedomIntelligence/sharegpt-chinese
    ```
   

  # 模型评测
  MMLU、C-Eval、CMMLU 自动化评估基线
  BELU、ROUGE、BERTScore、人工评测体系。

  C-Eval 是一个全面的中文评估基准，旨在评估语言模型在中文语境下的知识与推理能力，涵盖了52个学科的13948个多项选择题，分为4个难度级别：https://evalscope.readthedocs.io/zh-cn/latest/benchmarks/ceval.html

  ## evalscope评测：https://evalscope.readthedocs.io/zh-cn/latest/best_practice/qwen3.html
  ### 1. 安装vllm,部署qwen3-14b
  ```
  modelscope download --model Qwen/Qwen3-14B --local_dir
  #拉起服务
  vllm serve /root/autodl-tmp/Qwen/Qwen3-14B --gpu-memory-utilization 0.95 --served-model-name Qwen3-14B --port 6606 --max-model-len 4096
  --max-model-len = 模型最大上下文长度
  代表 vLLM 推理时，单条请求最多能容纳的输入 + 输出总 token 数量。
  vLLM 启动时，会根据 max-model-len 预分配整块 KV Cache 缓存
  #测试服务
  curl http://127.0.0.1:6606/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "Qwen3-14B",
  "messages": [{"role": "user", "content": "1+1=?"}],
  "temperature": 0.6,
  "top_p": 0.95,
  "max_tokens": 200,
  "stream": true
  }'
  ```
  ### 2. 安装evalscope
  ```
  pip install 'evalscope[app,perf]' -U
  ```
  ### 3. 评测脚本

```
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen3-14B',
    api_url='http://127.0.0.1:6606/v1/chat/completions',
    eval_type='openai_api',
    datasets=['ceval',],
    eval_batch_size=1,
    generation_config={
        'max_tokens': 200,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.7,  # 采样温度 (qwen 报告推荐值)
        'top_p': 0.8,  # top-p采样 (qwen 报告推荐值)
        'top_k': 20,  # top-k采样 (qwen 报告推荐值)
        'n': 1,  # 每个请求产生的回复数量
        'extra_body':{'chat_template_kwargs': {'enable_thinking': False}}  # 关闭思考模式
    },
    timeout=60000,  # 超时时间
    stream=True,  # 是否使用流式输出
    limit=10,  # 设置为1000条数据进行测试，但是好像不管用
)

run_task(task_cfg=task_cfg)
```
```
#取子集的方式
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen3-14B',
    api_url='http://127.0.0.1:6606/v1/chat/completions',
    eval_type='openai_api',
    datasets=['ceval'],
    dataset_args={
        "ceval": {
            "subset_list": ["business_administration", "marxism"]  # 只加载2门学科
        }
    },
    eval_batch_size=1,
    generation_config={
        'max_tokens': 1024,
        'temperature': 0.7,
        'top_p': 0.8,
        'top_k': 20,
        'n': 1,
        'extra_body':{'chat_template_kwargs': {'enable_thinking': False}}
    },
    timeout=60000,
    stream=True,
)

run_task(task_cfg=task_cfg)
```
<img width="829" height="335" alt="image" src="https://github.com/user-attachments/assets/017629b1-94b3-437a-b210-8cde0e734261" />



# 训练过程中遇到的问题
- 灾难性遗忘 ：在通用benchmark上测试分数下滑，日常闲聊也变得生硬
   方法：数据重放策略- 混入10%的通用数据
- 长尾分布 ：死记硬背，难以泛化（更改句式出现幻觉，只记得训练数据中的提问方式）
  - 分析长尾场景，利用gpt扩充数据，不同的问法，语气，加入一定的噪声（错别字，口语化表达）；
  - 过采样（将数据少的复制2-5次加入训练集）

  
 

