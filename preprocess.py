import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm

# ================= 配置区域 =================
# 原始数据路径
RAW_ROOT = r"E:\Workspace\Projects\Data\BraTS\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
# 处理后数据保存路径 (建议放在同级目录下新建一个 processed 文件夹)
SAVE_ROOT = r"E:\Workspace\Projects\Data\BraTS\BraTS2020_TrainingData\Processed_Data"


# ===========================================

def get_modalities_paths(patient_folder):
    """ 获取路径 """
    nii_files = glob.glob(os.path.join(patient_folder, "*.nii*"))
    paths = {}
    for file_path in nii_files:
        filename = os.path.basename(file_path).lower()
        if 'flair' in filename:
            paths['flair'] = file_path
        elif 't1ce' in filename:
            paths['t1ce'] = file_path
        elif 't1' in filename:
            paths['t1'] = file_path
        elif 't2' in filename:
            paths['t2'] = file_path
        elif 'seg' in filename:
            paths['seg'] = file_path
    return paths if len(paths) >= 5 else None


def process_patient(patient_folder, save_dir):
    patient_id = os.path.basename(patient_folder)
    paths = get_modalities_paths(patient_folder)
    if not paths: return

    # 1. 读取
    flair = nib.load(paths['flair']).get_fdata().astype(np.float32)
    t1 = nib.load(paths['t1']).get_fdata().astype(np.float32)
    t1ce = nib.load(paths['t1ce']).get_fdata().astype(np.float32)
    t2 = nib.load(paths['t2']).get_fdata().astype(np.float32)
    seg = nib.load(paths['seg']).get_fdata().astype(np.uint8)

    img = np.stack([flair, t1, t1ce, t2], axis=0)  # (4, H, W, D)
    seg[seg == 4] = 3  # 标签修正

    # 2. 裁剪 (去黑边)
    mask = np.any(img > 0, axis=0)
    coords = np.where(mask)
    if len(coords[0]) == 0: return  # 坏数据跳过

    x_min, x_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    z_min, z_max = coords[2].min(), coords[2].max() + 1

    img = img[:, x_min:x_max, y_min:y_max, z_min:z_max]
    seg = seg[x_min:x_max, y_min:y_max, z_min:z_max]

    # 3. 归一化 (Z-Score)
    # 注意：这里我们直接保存归一化后的数据，训练时就不用算了
    img_norm = np.zeros_like(img)
    for c in range(4):
        channel = img[c]
        mask = channel > 0
        if np.count_nonzero(mask) > 0:
            mean = channel[mask].mean()
            std = channel[mask].std()
            channel = (channel - mean) / (std + 1e-8)
            channel[~mask] = 0
        img_norm[c] = channel

    # 4. 保存为 .npy 格式
    # 保存两个文件：img.npy 和 seg.npy
    # 创建病人专属文件夹
    save_patient_dir = os.path.join(save_dir, patient_id)
    os.makedirs(save_patient_dir, exist_ok=True)

    np.save(os.path.join(save_patient_dir, "img.npy"), img_norm)
    np.save(os.path.join(save_patient_dir, "seg.npy"), seg)

    # 打印简略信息
    # print(f"Saved {patient_id}: shape {img.shape}")


def main():
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)

    patient_folders = glob.glob(os.path.join(RAW_ROOT, "BraTS*_Training_*"))
    print(f"开始处理 {len(patient_folders)} 个病人数据...")
    print(f"结果将保存在: {SAVE_ROOT}")

    # 使用 tqdm 显示进度条
    for folder in tqdm(patient_folders):
        process_patient(folder, SAVE_ROOT)

    print("\n所有数据处理完毕！")


if __name__ == "__main__":
    main()