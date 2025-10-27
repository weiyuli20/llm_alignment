系统学习sft

tokenizer.pad_token = tokenizer.eos_token 用于训练兼容，解决部分开源模型在预训练时没有定义pad_token问题。

如果数据集满足SFTTrainer的要求，不用做预处理，可以直接传给SFTTrainer,否则，需要预处理数据集，转换成SFTTrainer支持的格式。

chat template  大模型的对话模板，如果数据集满足sfttrainer要求的格式，且tokenzier带有chat_template,会自动转换
setup_chat_format 用于配置自定义的cahttemplate 或者选择合适的chat template 设置给模型

处理后用于微调的数据集要么是只包含一个text 字符串，这个字符串已经是chat template格式， 要么是符合要求的对话数据集格式，sfttrainer内部会自动完成转换。

## 0. sft 理论
- 损失函数:交叉熵损失，最大化条件概率 P(响应|上下文)
- label 是由input右移一个token得到的。
- 
## 1.学习 trl SFTTrainer
SFTTrainer 是trl库中用于微调语言模型的一个类

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

处理后用于微调的数据集要么是只包含一个text 字符串，这个字符串已经是chat template格式， 要么是符合要求的对话数据集格式

``` python
def process_fun(example):
  MAX_LENGTH=384
  input_ids, attention_mask, labels = [], [], []
  instruction =tokenizer(f"<|im_start|>system\n现在你要扮演皇帝身边的女人-甄嬛<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
  response = tokenizer(f"{example['output']}", add_special_tokens=False)
  input_ids = instruction["input_ids"] + response["input_ids"]+[tokenizer.eos_token_id]
  attention_mask = instruction["attention_mask"] + response["attention_mask"] +[1]
  labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
  #超出最大长度进行截断处理
  if len(input_ids) > MAX_LENGTH:
    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

  return {
      "input_ids": input_ids,
      "attention_mask": attention_mask,
      "labels": labels
  }
```

``` python
def format_template(sample):
    import random
    user_input = sample["instruction"] +sample["input"]
    answer = sample["output"]

    return {"messages": [
        {"role":"system", "content": "现在你要扮演皇帝身边的女人-甄嬛"},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": answer}
    ]}

ds = ds.map(format_template,num_proc=16,remove_columns=ds.column_names)
```

- SFTConfig 配置训练超参数
  
  常用配置项：
  - model_init_kwargs={"dtype":torch.bfloat16}, 这个字典支持 from_pretrained()方法的所有参数
  - packing=True  是指能将一些短句子拼成一个样本，减少padding
  - assistant_only_loss=True,仅计算assistant部分的损失，对数据集格式有严格要求，数据集格式为 Conversational
  - completion_only_loss=True, 仅计算completion部分的损失，数据集格式为 Conversational prompt-completion
  - learning_rate :学习率
  - output_dir:
  - logging_steps: 10 ，多少步记录一次日志
  - num_train_epoch: 3 ， 训练轮数
  - gradient_checkpointing:True ，开启梯度检查点
  - save_steps: 100 

- PEFTConfig 与peft库结合使用进行参数高效微调
  
  也可以不用PEFTConfig 而直接load一个PeftModel 传给SFTTrainer

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

## evaluate llm
https://www.philschmid.de/evaluate-llm