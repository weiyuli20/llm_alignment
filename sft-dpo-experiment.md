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
 

  # 数据集
