import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

# 引入你的基础类
from train import NpyDataset, UNet3D, calculate_dice

# ================= 配置区域 =================
DATA_ROOT = r"E:\Workspace\Projects\Data\BraTS\BraTS2020_TrainingData\Processed_Data"
PATCH_SIZE = (96, 96, 96)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 【关键】在这里填入你所有保存好的模型路径
# 格式：权重数值: "文件名"
MODEL_MAP = {
    0: "unet3d_epoch_44.pth",  # Baseline
    0.001: "unet3d_finetune_w0.001.pth",  # 弱约束
    0.005: "unet3d_finetune(TC+ET)_epoch_20.pth",  # 你的最佳模型 (注意文件名对不对)
    0.01: "unet3d_finetune_w0.01.pth",  # 强约束
    0.02: "unet3d_finetune_w0.02.pth"  # 过强约束
}


# ================= 评估函数 =================
def evaluate_model(model_path, val_loader):
    print(f"Loading: {model_path} ...")
    model = UNet3D(in_channels=4, out_channels=4).to(DEVICE)

    if not os.path.exists(model_path):
        print(f"⚠️ 警告: 找不到文件 {model_path}，跳过此权重！")
        return None, None, None

    model.load_state_dict(torch.load(model_path))
    model.eval()

    dice_ET, dice_TC, dice_WT = [], [], []

    with torch.no_grad():
        for img, seg in val_loader:
            img, seg = img.to(DEVICE), seg.to(DEVICE)
            output = model(img)
            pred = torch.argmax(output, dim=1)

            for b in range(img.shape[0]):
                p, t = pred[b], seg[b]
                # ET (Label 3)
                p_et, t_et = (p == 3).float(), (t == 3).float()
                dice_ET.append((2. * (p_et * t_et).sum() / (p_et.sum() + t_et.sum() + 1e-8)).item())
                # TC (Label 1+3)
                p_tc, t_tc = ((p == 1) | (p == 3)).float(), ((t == 1) | (t == 3)).float()
                dice_TC.append((2. * (p_tc * t_tc).sum() / (p_tc.sum() + t_tc.sum() + 1e-8)).item())
                # WT (Label > 0)
                p_wt, t_wt = (p > 0).float(), (t > 0).float()
                dice_WT.append((2. * (p_wt * t_wt).sum() / (p_wt.sum() + t_wt.sum() + 1e-8)).item())

    return np.mean(dice_ET), np.mean(dice_TC), np.mean(dice_WT)


# ================= 主程序 =================
def main():
    # 准备数据
    val_ds = NpyDataset(DATA_ROOT, mode='val', patch_size=PATCH_SIZE)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)  # 显存够可以改大 batch_size

    results = {'weight': [], 'ET': [], 'TC': [], 'WT': []}

    print(">>> 开始批量评估所有模型...")

    # 按权重大小排序执行
    sorted_weights = sorted(MODEL_MAP.keys())

    for w in sorted_weights:
        path = MODEL_MAP[w]
        print(f"\n--- 正在评估权重 lambda={w} ---")
        et, tc, wt = evaluate_model(path, val_loader)

        if et is not None:
            results['weight'].append(w)
            results['ET'].append(et)
            results['TC'].append(tc)
            results['WT'].append(wt)
            print(f"   [结果] ET: {et:.4f} | TC: {tc:.4f} | WT: {wt:.4f}")

    print("\n>>> 所有评估完成，正在绘图...")

    # --- 绘图部分 ---
    plt.figure(figsize=(10, 6), dpi=300)

    # 绘制三条线
    plt.plot(results['weight'], results['ET'], 'o-', linewidth=2, color='#d62728', label='ET (Enhancing Tumor)')  # 红色
    plt.plot(results['weight'], results['TC'], 's--', linewidth=1.5, color='#2ca02c', label='TC (Tumor Core)')  # 绿色
    plt.plot(results['weight'], results['WT'], '^-.', linewidth=1.5, color='#1f77b4', label='WT (Whole Tumor)')  # 蓝色

    # 标注最佳选择点 (0.005)
    target_w = 0.005
    if target_w in results['weight']:
        idx = results['weight'].index(target_w)
        plt.annotate(f'Selected: {results["ET"][idx]:.4f}',
                     xy=(target_w, results['ET'][idx]),
                     xytext=(target_w, results['ET'][idx] + 0.03),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=10, fontweight='bold')
        # 画一条虚线垂线指示
        plt.axvline(x=target_w, color='gray', linestyle=':', alpha=0.5)

    plt.title('Parameter Sensitivity Analysis of RSF Loss Weight ($\lambda$)', fontsize=14)
    plt.xlabel('RSF Loss Weight ($\lambda$)', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.xticks(results['weight'])  # 强制显示所有刻度

    save_path = 'sensitivity_result_2.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 最终折线图已保存为: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()