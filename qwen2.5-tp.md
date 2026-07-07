# 阅读qwen2.5技术报告的一些笔记

qwen2.5

模型：
 -  dense:开源 
   - base:0.5B, 1.5B, 3B, 7B, 14B, 32B, and 72B
   - instruct:0.5B, 1.5B, 3B, 7B, 14B, 32B, and 72B
 - moe: 闭源。Qwen2.5-Turbo and Qwen2.5-Plus

模型架构：
- tokenizer: BBPE
- dense：transformer、RoPE、SwiGLU, RMSNorm with preNorm,GQA
- moe

上下文长度和支持生成的长度
- dense：
   - context：32768 32k
   - generation: 8192 8k
  
- moe : 1M

  

训练数据：
- 预训练：18 trillon  
  - focus on knowledge,coding, and mathematics
  - 数据配比：Through strategic down-sampling ofoverrepresented domains and up-sampling of high-value domains
  - Long-context Pre-training：
    - dense 两阶段: an initial phase with a 4,096-token context length, followed by an extension phase for longer sequences，extend the context length from 4,096 to 32,768 tokens during the final pre-training stage
    - For Qwen2.5-Turbo, we implement a progressive context length expansion strategy during training,advancing through four stages: 32,768 tokens, 65,536 tokens, 131,072 tokens, and ultimately 262,144 tokens
- 后训练: 共1 million
   - 后训练（sft)


训练方法：
- 预训练
- 后训练
   - sft
   - offline dpo
   - online grpo



特点:
 - including larger generation length (from 2K tokens to **8K** tokens), better support for structured input and output,(e.g., tables and JSON), and easier tool use.
