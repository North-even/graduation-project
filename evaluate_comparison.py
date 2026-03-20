import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import NpyDataset, UNet3D  # 导入你的模型和数据集类

# ================= 配置 =================
DATA_ROOT = r"E:\Workspace\Projects\Data\BraTS\BraTS2020_TrainingData\Processed_Data"
PATCH_SIZE = (96, 96, 96)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 两个模型的路径
BASELINE_MODEL = "unet3d_epoch_44.pth"  # 对照组
FINETUNED_MODEL = "unet3d_finetune(TC+ET)_epoch_20.pth"  # 实验组 (RSF)


# =======================================

def get_dice_score(pred, target, label_idx):
    # 计算特定类别的 Dice
    p = (pred == label_idx).float()
    t = (target == label_idx).float()
    intersection = (p * t).sum()
    union = p.sum() + t.sum()
    if union == 0: return 1.0  # 两者都没有该类别，算满分
    return (2.0 * intersection / (union + 1e-8)).item()


def evaluate_model(model_path, dataloader, name="Model"):
    print(f"--- 正在评估模型: {name} ({model_path}) ---")
    model = UNet3D(in_channels=4, out_channels=4).to(DEVICE)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"错误: 找不到模型 {model_path}")
        return None

    model.eval()

    # 存储所有病人的分数
    dice_WT = []  # Whole Tumor (1+2+3)
    dice_TC = []  # Tumor Core (1+3)
    dice_ET = []  # Enhancing Tumor (3)

    with torch.no_grad():
        for i, (img, seg) in enumerate(dataloader):
            img, seg = img.to(DEVICE), seg.to(DEVICE)

            # 预测
            output = model(img)
            pred = torch.argmax(output, dim=1)  # (B, D, H, W)

            # 遍历 batch 中的每个样本
            for b in range(img.shape[0]):
                p = pred[b]
                t = seg[b]

                # --- BraTS 标准区域计算 ---
                # 1. ET (Enhancing Tumor): 标签 3
                dice_et = get_dice_score(p, t, 3)

                # 2. TC (Tumor Core): 标签 1 或 3
                p_tc = ((p == 1) | (p == 3)).float()
                t_tc = ((t == 1) | (t == 3)).float()
                inter_tc = (p_tc * t_tc).sum()
                union_tc = p_tc.sum() + t_tc.sum()
                dice_tc = (2.0 * inter_tc / (union_tc + 1e-8)).item() if union_tc > 0 else 1.0

                # 3. WT (Whole Tumor): 标签 > 0
                p_wt = (p > 0).float()
                t_wt = (t > 0).float()
                inter_wt = (p_wt * t_wt).sum()
                union_wt = p_wt.sum() + t_wt.sum()
                dice_wt = (2.0 * inter_wt / (union_wt + 1e-8)).item() if union_wt > 0 else 1.0

                dice_ET.append(dice_et)
                dice_TC.append(dice_tc)
                dice_WT.append(dice_wt)

    # 计算平均分
    avg_ET = np.mean(dice_ET)
    avg_TC = np.mean(dice_TC)
    avg_WT = np.mean(dice_WT)
    avg_All = (avg_ET + avg_TC + avg_WT) / 3.0

    print(f"[{name}] 评估结果:")
    print(f"  > ET (增强肿瘤) Dice: {avg_ET:.4f}  <-- 重点关注！")
    print(f"  > TC (肿瘤核心) Dice: {avg_TC:.4f}")
    print(f"  > WT (整体肿瘤) Dice: {avg_WT:.4f}")
    print(f"  > 平均 Dice         : {avg_All:.4f}")
    print("-" * 30)

    return avg_ET, avg_TC, avg_WT, avg_All


def main():
    # 加载验证集
    # 注意: 这里 batch_size 还是设为 4 或 2，取决于显存
    val_ds = NpyDataset(DATA_ROOT, mode='val', patch_size=PATCH_SIZE)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    print(f"验证集样本数: {len(val_ds)}")

    # 评估 Baseline
    res_base = evaluate_model(BASELINE_MODEL, val_loader, name="Baseline (Dice Only)")

    # 评估 Ours
    res_ours = evaluate_model(FINETUNED_MODEL, val_loader, name="Ours (Dice + RSF)")

    # 打印对比总结
    if res_base and res_ours:
        print("\n====== 最终提升报告 (Improvement Report) ======")
        print(f"{'Metric':<10} | {'Baseline':<10} | {'Ours':<10} | {'Improvement'}")
        print("-" * 50)
        print(f"{'ET Dice':<10} | {res_base[0]:.4f}     | {res_ours[0]:.4f}     | {res_ours[0] - res_base[0]:+.4f}")
        print(f"{'TC Dice':<10} | {res_base[1]:.4f}     | {res_ours[1]:.4f}     | {res_ours[1] - res_base[1]:+.4f}")
        print(f"{'WT Dice':<10} | {res_base[2]:.4f}     | {res_ours[2]:.4f}     | {res_ours[2] - res_base[2]:+.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()