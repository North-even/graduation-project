import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math

# 引入你之前的类定义 (为了代码简洁，假设你把类都放在 train.py 或者这里复制一遍)
# 必须包含: NpyDataset, UNet3D, DiceLoss, RSFLoss (修复版)
from train import NpyDataset, UNet3D, DiceLoss, RSFLoss

# ================= 微调配置 =================
DATA_ROOT = r"E:\Workspace\Projects\Data\BraTS\BraTS2020_TrainingData\Processed_Data"
# 加载你最好的那个模型
PRETRAINED_PATH = "unet3d_epoch_44.pth"
PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 4
# 【关键】：学习率要小！
LEARNING_RATE = 1e-4
# 微调不需要太多轮，10-20轮足矣
NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================

def main():
    print(f"--- 启动微调 (Fine-tuning): {DEVICE} ---")

    # 1. 准备数据
    train_ds = NpyDataset(DATA_ROOT, mode='train', patch_size=PATCH_SIZE)
    val_ds = NpyDataset(DATA_ROOT, mode='val', patch_size=PATCH_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 2. 初始化模型
    model = UNet3D(in_channels=4, out_channels=4).to(DEVICE)

    # 【关键】：加载预训练权重
    if os.path.exists(PRETRAINED_PATH):
        print(f"Loading checkpoint: {PRETRAINED_PATH}")
        model.load_state_dict(torch.load(PRETRAINED_PATH))
    else:
        print(f"错误: 找不到 {PRETRAINED_PATH}，无法微调！")
        return

    # 3. 优化器 (使用较小的学习率)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Loss 定义
    criterion_dice = DiceLoss(n_classes=4)
    # RSF 参数保持论文默认即可，sigma=2.0 或 3.0 对细节捕捉较好
    criterion_rsf = RSFLoss(lambda_1=1.0, lambda_2=1.0, mu=0.1, nu=0.001, sigma=2.0)

    # 5. 微调循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0

        for i, (img, seg) in enumerate(train_loader):
            img, seg = img.to(DEVICE), seg.to(DEVICE)

            optimizer.zero_grad()
            output = model(img)

            # --- 计算 Loss ---

            # A. 基础 Dice Loss (保持轮廓不崩)
            loss_dice = criterion_dice(output, seg)

            # B. 针对"增强肿瘤"的 RSF Loss (优化细节)
            # 我们针对 Channel 3 (增强肿瘤) 和 Flair图像 (Channel 0) 或 T1ce图像 (Channel 2)
            # 增强肿瘤在 T1ce (Channel 2) 上最亮，建议用 T1ce 做引导！
            # img shape: (B, 4, D, H, W) -> T1ce is index 2

            target_logit = output[:, 3:4, ...]  # (B, 1, ...) 增强肿瘤预测
            target_seg = ((seg == 1) | (seg == 3)).float().unsqueeze(1)
            input_img = img[:, 2:3, ...]  # (B, 1, ...) 使用 T1ce 模态作为灰度参考

            loss_rsf_val = criterion_rsf(target_logit, target_seg, input_img)

            # 【关键】：动态平衡
            # 这里的 0.005 是一个经验值。
            # 请在第一个 Step 观察打印结果：如果 RSF 是 100，Dice 是 0.4
            # 那么 0.005 * 100 = 0.5，这样两者量级相当，RSF 不会淹没 Dice。
            loss = loss_dice + 0.005 * loss_rsf_val

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 打印监控 (重点观察 RSF 的数值大小)
            if i % 20 == 0:
                print(
                    f"Ep {epoch + 1} | Step {i} | DiceL: {loss_dice.item():.4f} | RSFL: {loss_rsf_val.item():.2f} | Total: {loss.item():.4f}")

        # --- Validation ---
        # ... (保持原有的验证逻辑) ...
        # 保存微调后的模型
        torch.save(model.state_dict(), f"unet3d_finetune(TC+ET)_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    main()