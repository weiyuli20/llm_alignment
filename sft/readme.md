系统学习sft

## 0. sft 理论
- 损失函数:交叉熵损失，最大化条件概率 P(响应|上下文)
- label 是由input右移一个token得到的。
- 
## 1.学习 trl SFTTrainer
 - SFTTrainer 支持的数据集格式
```
#Conversational
{"messages": [{"role": "user", "content": "What color is the sky?"},
              {"role": "assistant", "content": "It is blue."}]}

# Standard prompt-completion
{"prompt": "The sky is",
 "completion": " blue."}

# Conversational prompt-completion
{"prompt": [{"role": "user", "content": "What color is the sky?"}],
 "completion": [{"role": "assistant", "content": "It is blue."}]}
```

如果数据集格式不满足，需要自行转换
- SFTConfig 配置训练超参数
  
  常用配置项：
  - model_init_kwargs={"dtype":torch.bfloat16}, 这个字典支持 from_pretrained()方法的所有参数
  - packing=True  是指能将一些短句子拼成一个样本，减少padding
  - assistant_only_loss=True,仅计算assistant部分的损失，对数据集格式有严格要求，数据集格式为 Conversational
  - completion_only_loss=True, 仅计算completion部分的损失，数据集格式为 Conversational prompt-completion
  - learning_rate :学习率
  - output_dir:
  - 

- PEFTConfig 与peft库结合使用进行参数高效微调
  
  也可以不同PEFTConfig 而直接load一个PeftModel 传给SFTTrainer

- SFTTrainer，实例化trainer
  
  常用参数：
  - model : 待微调的模型
  - train_dataset : 训练数据集
  - args : SFTConfig


- trainer.train()开启训练
- 可以和 Liger Kernel 、Unsloth结合使用加速训练

## 练习使用SFTTrainer 进行微调， 结合Unsloth进行微调
## LoRA

## QLoRA

## 2. 手写sft训练

## 3.复现sft 灾难遗忘问题

## 4 sft llm  function call 能力