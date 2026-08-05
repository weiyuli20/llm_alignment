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

** 条件生成 - loss_based guidance ** 

ddpm 无条件生成，如果我们想控制生成图片的一些属性，比如颜色（希望生成的图片整体看起来是绿色的）， 种类（只想生成小猫图片),风格..., 那就就有条件生成了


loss_bases guidance是一种在推理过程中实现条件约束生成的方法，不需要重新训练模型

通过定义目标损失函数，比如目标是绿色图片，可以定义一个损失函数计算预测和tagert之间的mse损失，然后对x计算梯度，根据梯度方向扰动x

注意：x.require_grad()有两个时机：
- after unet
- before unet :这种情况x经过unet会产生激活值，显存占用很多

** 条件生成 -clip_based guidance**

使用这种方法可以实现文生图，生成符合文本描述的图片，同样也是定义一个损失函数，这个函数计算图文embedding的距离作为loss



