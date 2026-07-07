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
 

  # 数据集（medical-gpt)
  数据集最终混合配方；在最终送入训练的 train.json 中，你的数据结构应该是：
  - 55% 垂直领域单轮中文数据  https://huggingface.co/datasets/shibing624/medical/tree/main/finetune。  40k 
  - 25% 垂直领域多轮中文数据
  - 10% 通用英文高质量数据（单/多轮均可）
  - 10% 通用中文对话数据（用于保持通用聊天能力）share-gpt-chinese 共90k, 是中英双语的，从中抽取10k
    - 中英文7:3
    - 50%的多轮对话，选择对话轮次大于2的
    - 过滤低质量样本 （对话回复在30-500）之间
 

