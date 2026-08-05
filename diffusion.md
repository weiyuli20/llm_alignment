 diffusion

 核心模型unet

 推理时：unet的输出是带噪图片，输出是估计的噪声信息
 
 推理阶段的关键模块：scheduler, 它会接受（unet预测的噪声，x_t,t) 计算出一个噪声信息最少的x_(t-1)

 scheduler是数学算法，不涉及模型训练，它控制生成速度和质量 ， 可以尝试使用不同的scheduler 观察生成效果




**关于sft训练时学习率和batch_size的关系**
batch_size增大，梯度的噪声变小，梯度更稳定，可以使用更大学习率

当使用 梯度累计时： lr_new ≈ lr_old * gradient_accumulation_step

如果还是用原来的小学习率lr_old,收敛慢


调大学习率的一个等价做法是将 loss/ gradient_accumulation_step

```
grad_accum_steps = 4   # 你可以改成2/4/8做实验

for step, (images,) in enumerate(train_dataloader):

    # === forward ===
    noise = torch.randn_like(images)
    t = torch.randint(0, num_train_timesteps, (images.shape[0],), device=device)
    noisy_images = noise_scheduler.add_noise(images, noise, t)
    noise_pred = unet(noisy_images, t).sample
    loss = F.mse_loss(noise_pred, noise)

    # 关键：loss缩放，抵消梯度累加带来的幅度放大
    loss = loss / grad_accum_steps

    loss.backward()   # 梯度会累加，不step，不清零

    # 只有累积够步数，才更新权重、清空梯度
    if (step + 1) % grad_accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```
