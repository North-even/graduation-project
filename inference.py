import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from train import UNet3D  # 确保 train.py 和 inference.py 在同一目录下

# ================= 配置 =================
# 指向 Processed_Data 文件夹
DATA_ROOT = r"E:\Workspace\Projects\Data\BraTS\BraTS2020_TrainingData\Processed_Data"
MODEL_PATH = "unet3d_epoch_cs_46.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =======================================

def load_model():
    print(f"加载模型: {MODEL_PATH}")
    # 这里的参数必须和你训练时定义的一模一样
    model = UNet3D(in_channels=4, out_channels=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


def pad_to_divisible(img, divisor=16):
    """
    将图像补全为 divisor 的倍数，防止 UNet 拼接报错
    img: (4, D, H, W)
    """
    _, d, h, w = img.shape

    # 计算需要补多少像素
    pad_d = (divisor - d % divisor) % divisor
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor

    # 如果不需要补，直接返回
    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return img, (d, h, w)

    # 执行 Padding (在 D, H, W 的末尾补0)
    # np.pad 参数格式: ((channel前,后), (D前,后), (H前,后), (W前,后))
    img_padded = np.pad(img, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

    print(f"  自动Padding: {d}x{h}x{w} -> {img_padded.shape[1]}x{img_padded.shape[2]}x{img_padded.shape[3]}")
    return img_padded, (d, h, w)


def predict_and_plot(model, patient_folder):
    patient_id = os.path.basename(patient_folder)
    print(f"正在预测病人: {patient_id}")

    # 1. 读取数据
    img = np.load(os.path.join(patient_folder, "img.npy"))  # (4, D, H, W)
    seg = np.load(os.path.join(patient_folder, "seg.npy"))  # (D, H, W)

    # 2. 自动补全尺寸 (修复报错的关键)
    img_padded, original_shape = pad_to_divisible(img, divisor=16)

    # 3. 预处理 (转 Tensor)
    input_tensor = torch.from_numpy(img_padded).unsqueeze(0).float().to(DEVICE)  # (1, 4, D, H, W)

    # 4. 推理
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask_padded = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (D_pad, H_pad, W_pad)

    # 5. 恢复原始尺寸 (把补的边切掉)
    d, h, w = original_shape
    pred_mask = pred_mask_padded[:d, :h, :w]

    # 6. 可视化
    # 找肿瘤面积最大的一层展示
    if np.sum(seg) > 0:
        slice_idx = np.argmax(np.sum(seg > 0, axis=(1, 2)))
    else:
        slice_idx = seg.shape[0] // 2

    print(f"展示切片层: {slice_idx}/{d}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # A. 原始图像 (Flair)
    axes[0].imshow(img[0, slice_idx, :, :], cmap='gray')
    axes[0].set_title(f"Input (Flair) - Slice {slice_idx}")

    # B. 金标准
    axes[1].imshow(seg[slice_idx, :, :], cmap='jet', interpolation='nearest', vmin=0, vmax=3)
    axes[1].set_title("Ground Truth")

    # C. 预测结果
    axes[2].imshow(pred_mask[slice_idx, :, :], cmap='jet', interpolation='nearest', vmin=0, vmax=3)
    axes[2].set_title("Model Prediction")

    plt.suptitle(f"Patient: {patient_id}")
    plt.tight_layout()
    plt.show()


def main():
    model = load_model()

    all_patients = glob.glob(os.path.join(DATA_ROOT, "BraTS*"))
    # 取最后20%作为测试
    val_patients = all_patients[int(len(all_patients) * 0.8):]

    if not val_patients:
        print("未找到病人数据")
        return

    # 循环找一个有肿瘤的病人展示 (防止正好随机到一个健康人，看不出效果)
    for patient_path in val_patients[:5]:
        seg = np.load(os.path.join(patient_path, "seg.npy"))
        if np.sum(seg) > 0:
            predict_and_plot(model, patient_path)
            break
    else:
        # 如果前5个都没肿瘤，就硬画第一个
        predict_and_plot(model, val_patients[0])


if __name__ == "__main__":
    main()