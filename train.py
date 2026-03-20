import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math

# ================= 配置区域 =================
# 指向刚才生成的 Processed_Data 文件夹
DATA_ROOT = r"E:\Workspace\Projects\Data\BraTS\BraTS2020_TrainingData\Processed_Data"
PATCH_SIZE = (96, 96, 96)  # 训练切块大小
BATCH_SIZE = 4
LEARNING_RATE = 1e-3  # 初始学习率
NUM_EPOCHS = 50  # 训练轮数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================

# --- 1. 高性能 Dataset (读取 .npy) ---
class NpyDataset(Dataset):
    def __init__(self, root_dir, patch_size=(64, 64, 64), mode='train'):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.patient_folders = glob.glob(os.path.join(root_dir, "BraTS*"))

        # 简单划分训练集和验证集 (前80%训练，后20%验证)
        split_idx = int(len(self.patient_folders) * 0.8)
        if mode == 'train':
            self.patient_folders = self.patient_folders[:split_idx]
        else:
            self.patient_folders = self.patient_folders[split_idx:]

        print(f"[{mode}] 集加载完毕，共 {len(self.patient_folders)} 个样本")

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        folder = self.patient_folders[idx]
        # 读取处理好的 npy (速度极快)
        img = np.load(os.path.join(folder, "img.npy"))  # (4, D, H, W) 或 (4, H, W, D) 注意之前的保存顺序
        seg = np.load(os.path.join(folder, "seg.npy"))  # (D, H, W)

        # 确保形状和 patch 匹配
        img, seg = self.pad_if_needed(img, seg)
        img, seg = self.random_crop(img, seg)

        return torch.from_numpy(img).float(), torch.from_numpy(seg).long()

    def pad_if_needed(self, img, seg):
        # img shape: (4, x, y, z) based on preprocess logic
        _, x, y, z = img.shape
        px, py, pz = self.patch_size

        pad_x = max(0, px - x)
        pad_y = max(0, py - y)
        pad_z = max(0, pz - z)

        if pad_x > 0 or pad_y > 0 or pad_z > 0:
            img = np.pad(img, ((0, 0), (0, pad_x), (0, pad_y), (0, pad_z)), mode='constant')
            seg = np.pad(seg, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant')
        return img, seg

    def random_crop(self, img, seg):
        _, x, y, z = img.shape
        px, py, pz = self.patch_size

        # 随机起点
        sx = np.random.randint(0, x - px + 1) if x > px else 0
        sy = np.random.randint(0, y - py + 1) if y > py else 0
        sz = np.random.randint(0, z - pz + 1) if z > pz else 0

        return img[:, sx:sx + px, sy:sy + py, sz:sz + pz], seg[sx:sx + px, sy:sy + py, sz:sz + pz]


# --- 2. 标准 3D U-Net 模型 ---
class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_filters=16):
        super(UNet3D, self).__init__()

        # 编码器 (Encoder)
        self.enc1 = self.conv_block(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = self.conv_block(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = self.conv_block(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d(2)

        # 瓶颈层 (Bottleneck)
        self.bottleneck = self.conv_block(base_filters * 4, base_filters * 8)

        # 解码器 (Decoder)
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(base_filters * 8, base_filters * 4)  # concat后通道翻倍输入

        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(base_filters * 2, base_filters)

        # 输出层
        self.final = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.up3(b)
        # Skip Connection: 拼接 e3 和 d3
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


#--- Dice Loss 实现 ---
class DiceLoss(nn.Module):
    def __init__(self, n_classes=4, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, inputs, target):
        # inputs: (B, C, D, H, W) -> 未经过 softmax 的 logits
        # target: (B, D, H, W)    -> 标签索引 (0-3)

        # 1. 对预测值做 Softmax
        inputs = F.softmax(inputs, dim=1)

        # 2. 将 target 转为 One-Hot 编码
        # target_one_hot: (B, C, D, H, W)
        target_one_hot = F.one_hot(target, num_classes=self.n_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()

        # 3. 计算交集和并集 (只计算前景类 1,2,3，忽略背景 0)
        # 也可以计算全部类，但通常忽略背景效果更好
        # 这里为了简单，计算所有类的平均 Dice

        # Flatten: (B, C, ...) -> (B, C, N)
        inputs_flat = inputs.view(inputs.size(0), self.n_classes, -1)
        target_flat = target_one_hot.view(target_one_hot.size(0), self.n_classes, -1)

        intersection = (inputs_flat * target_flat).sum(dim=2)
        union = inputs_flat.sum(dim=2) + target_flat.sum(dim=2)

        # Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss = 1 - Dice
        # 对 Batch 和 Channel 取平均
        return 1 - dice.mean()

# --- 数据项+长度项+正则项
class RSFLoss(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, mu=1.0, nu=1.0, sigma=3.0, kernel_size=15):
        """
        参数来源于论文 Minimization of Region-Scalable Fitting Energy for Image Segmentation, Li et al. 2008
        lambda_1, lambda_2: 数据项权重 (Data Term)
        mu: 正则项权重 (Regularization Term / Re-initialization)
        nu: 长度项权重 (Length Term)
        sigma: 高斯核标准差 (控制局部感知的范围)
        """
        super(RSFLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.mu = mu
        self.nu = nu
        self.sigma = sigma

        # 创建高斯核用于计算局部均值 f1, f2
        self.kernel = self._create_gaussian_kernel(kernel_size, sigma)

    def _create_gaussian_kernel(self, kernel_size, sigma):
        # 创建 3D 高斯核
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # 高斯核生成:
        grid = torch.meshgrid([torch.arange(kernel_size), torch.arange(kernel_size), torch.arange(kernel_size)])
        grid = torch.stack(grid, dim=-1).float()
        d2 = torch.sum((grid - mean) ** 2, dim=-1)
        kernel_3d = torch.exp(-d2 / (2 * variance))
        kernel_3d = kernel_3d / kernel_3d.sum()

        return kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)

    def forward(self, pred, target, image):
        """
        pred: 网络输出 (Logits), 形状 (B, 1, D, H, W) <-- 确保外面已经选好了通道
        target: 标签 (B, 1, D, H, W)
        image: 原始 MRI 图像 (B, 1, D, H, W)
        """
        # === 修复点：直接使用 pred，不要再切片了 ===
        phi = pred

        # H_phi = Sigmoid(pred) 近似 Heaviside 函数
        H_phi = torch.sigmoid(phi)

        # 1. 计算局部拟合中心 f1 (前景) 和 f2 (背景)
        kernel = self.kernel.to(pred.device).type(pred.type())
        pad = self.kernel.shape[-1] // 2

        # I * H(phi)
        I_H = image * H_phi
        # I * (1 - H(phi))
        I_1mH = image * (1.0 - H_phi)

        # 计算分子分母的卷积
        conv_I_H = F.conv3d(I_H, kernel, padding=pad)
        conv_H = F.conv3d(H_phi, kernel, padding=pad)

        conv_I_1mH = F.conv3d(I_1mH, kernel, padding=pad)
        conv_1mH = F.conv3d((1.0 - H_phi), kernel, padding=pad)

        # f1 = (K * [I H]) / (K * H)
        f1 = conv_I_H / (conv_H + 1e-8)
        # f2 = (K * [I (1-H)]) / (K * (1-H))
        f2 = conv_I_1mH / (conv_1mH + 1e-8)

        # 2. 计算 RSF 数据项 (Data Term)
        sq_diff_1 = (image - f1) ** 2
        sq_diff_2 = (image - f2) ** 2

        term1 = F.conv3d(sq_diff_1, kernel, padding=pad)
        term2 = F.conv3d(sq_diff_2, kernel, padding=pad)

        loss_data = self.lambda_1 * torch.mean(term1 * H_phi) + \
                    self.lambda_2 * torch.mean(term2 * (1.0 - H_phi))

        # 3. 长度项 (Length Term)
        # 简单全变分近似
        dy = torch.abs(H_phi[:, :, :, 1:, :] - H_phi[:, :, :, :-1, :])
        dx = torch.abs(H_phi[:, :, :, :, 1:] - H_phi[:, :, :, :, :-1])
        dz = torch.abs(H_phi[:, :, 1:, :, :] - H_phi[:, :, :-1, :, :])

        loss_length = (torch.mean(dx) + torch.mean(dy) + torch.mean(dz))

        # 4. 正则项 (Regularization Term)
        # 计算 phi 的梯度
        phi_dx = phi[:, :, :, :, 1:] - phi[:, :, :, :, :-1]
        phi_dy = phi[:, :, :, 1:, :] - phi[:, :, :, :-1, :]
        phi_dz = phi[:, :, 1:, :, :] - phi[:, :, :-1, :, :]

        phi_dx = F.pad(phi_dx, (0, 1, 0, 0, 0, 0))
        phi_dy = F.pad(phi_dy, (0, 0, 0, 1, 0, 0))
        phi_dz = F.pad(phi_dz, (0, 0, 0, 0, 0, 1))

        grad_phi_norm = torch.sqrt(phi_dx ** 2 + phi_dy ** 2 + phi_dz ** 2 + 1e-8)

        loss_reg = torch.mean((grad_phi_norm - 1.0) ** 2)

        return loss_data + self.nu * loss_length + self.mu * loss_reg


# --- 3. 辅助函数: 计算 Dice Score (评估指标) ---
def calculate_dice(pred, target, num_classes=4):
    # pred: (B, C, D, H, W) -> argmax -> (B, D, H, W)
    pred_mask = torch.argmax(pred, dim=1)
    dice_scores = []

    # 对每个类别分别计算 (跳过背景0)
    for cls in range(1, num_classes):
        p = (pred_mask == cls).float()
        t = (target == cls).float()

        intersection = (p * t).sum()
        union = p.sum() + t.sum()

        if union == 0:
            dice_scores.append(1.0)  # 如果GT和Pred都没有该类别，算满分
        else:
            dice = (2. * intersection) / (union + 1e-8)
            dice_scores.append(dice.item())

    return np.mean(dice_scores)  # 返回所有前景类别的平均 Dice


# --- 4. 训练主循环 ---
def main():
    print(f"--- 启动训练: {DEVICE} ---")

    # 准备数据
    train_ds = NpyDataset(DATA_ROOT, mode='train', patch_size=PATCH_SIZE)
    val_ds = NpyDataset(DATA_ROOT, mode='val', patch_size=PATCH_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 初始化模型
    model = UNet3D(in_channels=4, out_channels=4).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=4)
    criterion_rsf = RSFLoss(lambda_1=1.0, lambda_2=1.0, mu=0.1, nu=0.001, sigma=2.0)

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0

        # --- Training ---
        for i, (img, seg) in enumerate(train_loader):
            img, seg = img.to(DEVICE), seg.to(DEVICE)

            output = model(img)  # Output logits: (B, 4, D, H, W)

            # 1. 计算传统的 Dice Loss (保证大体结构对)
            loss_dice = criterion_dice(output, seg)
            loss_ce = criterion_ce(output, seg)

            # 2. 计算 RSF Loss (提升细节)
            # RSF通常用于二分类（前景vs背景）。我们可以把问题简化为：分割"整个肿瘤"
            # 制作 Whole Tumor 的二值标签: 只要 seg > 0 都是前景
            seg_binary = (seg > 0).float().unsqueeze(1)  # (B, 1, D, H, W)

            # 取出 Logits 中对应前景的一个通道 (或者把 3个前景通道的 logits 聚合)
            # 这里简单起见，假设 output[:, 1:, ...] 的均值代表肿瘤强度
            # 或者我们只优化 '增强肿瘤' (Enhancing Tumor, Label 3 -> channel 3)
            # 更好的做法：训练一个专门的 Logit 通道代表 Whole Tumor

            # 演示：只针对 Channel 0 (背景) 和 Channel 1+2+3 (前景)
            # 我们用 output[:, 1, ...] 代表前景 Logits (假设只关心某个模态)
            # 注意：RSF需要 原图 img (取 Flair 通道 img[:, 0:1, ...])

            # 假设我们只优化 Flair 模态下的 增强肿瘤 (Channel 3)
            target_logit = output[:, 3:4, ...]  # 取出增强肿瘤的预测 Logits
            target_seg = (seg == 3).float().unsqueeze(1)
            input_img = img[:, 0:1, ...]  # 取 Flair 图像

            loss_rsf_val = criterion_rsf(target_logit, target_seg, input_img)

            # 3. 组合 Loss
            # 刚开始 lambda_rsf 给小一点，防止训练崩溃
            loss = loss_dice + 0.005 * loss_rsf_val

            # ... backward ...
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if i % 10 == 0:
                print(f"Ep {epoch + 1}/{NUM_EPOCHS} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation (每轮结束测一下) ---
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for img, seg in val_loader:
                img, seg = img.to(DEVICE), seg.to(DEVICE)
                output = model(img)
                val_dice += calculate_dice(output, seg)

        avg_val_dice = val_dice / len(val_loader)

        print(f"\n>>> Epoch {epoch + 1} 结束")
        print(f">>> Train Loss: {avg_train_loss:.4f}")
        print(f">>> Val Dice  : {avg_val_dice:.4f} (越高越好)\n")

        # 保存模型
        torch.save(model.state_dict(), f"unet3d_epoch_cs_{epoch + 1}.pth")


if __name__ == "__main__":
    main()