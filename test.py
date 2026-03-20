import matplotlib.pyplot as plt
import numpy as np

# ================= 1. 数据录入 (已根据你的日志自动填充) =================

# --- Cold Start (Mixed Loss from Ep 0) ---
# 特征：起步低(0.08)，收敛慢，最高仅 0.56
cold_dice = [
    0.0882, 0.1982, 0.2470, 0.3337, 0.3692, 0.3452, 0.3703, 0.3835, 0.3827, 0.3902,
    0.3916, 0.3976, 0.4164, 0.4445, 0.4581, 0.3548, 0.4278, 0.4414, 0.4971, 0.4683,
    0.4506, 0.4519, 0.4929, 0.4740, 0.4731, 0.4309, 0.4272, 0.4202, 0.4326, 0.4608,
    0.4684, 0.4903, 0.4946, 0.4434, 0.4785, 0.4943, 0.5194, 0.5187, 0.5188, 0.5022,
    0.5338, 0.5103, 0.4634, 0.5554, 0.5353, 0.5682, 0.5512, 0.5366, 0.4813, 0.4816
]
cold_loss = [
    0.8225, 0.7255, 0.6548, 0.5641, 0.5062, 0.4799, 0.4874, 0.4768, 0.4652, 0.4642,
    0.4567, 0.4543, 0.4363, 0.4324, 0.4247, 0.4215, 0.4398, 0.4213, 0.4098, 0.4030,
    0.3945, 0.4026, 0.4111, 0.4009, 0.3907, 0.3796, 0.3873, 0.3964, 0.3819, 0.3849,
    0.3849, 0.3786, 0.3683, 0.3802, 0.3606, 0.3684, 0.3667, 0.3599, 0.3711, 0.3674,
    0.3588, 0.3661, 0.3551, 0.3569, 0.3708, 0.3695, 0.3619, 0.3759, 0.3751, 0.3686
]

# --- Warm Start Stage 1 (Pure Dice) ---
# 特征：起步高(0.39)，收敛快，最高达 0.72
warm_dice = [
    0.3945, 0.4339, 0.4438, 0.4568, 0.4418, 0.4283, 0.4806, 0.5214, 0.5818, 0.5828,
    0.5952, 0.5611, 0.5640, 0.5993, 0.6415, 0.6366, 0.5853, 0.6452, 0.6596, 0.6589,
    0.6422, 0.6393, 0.6561, 0.5865, 0.6275, 0.6779, 0.6445, 0.6970, 0.6814, 0.6917,
    0.7089, 0.7040, 0.6916, 0.6525, 0.7004, 0.6636, 0.6803, 0.6757, 0.6873, 0.6459,
    0.6876, 0.6995, 0.6765, 0.7226, 0.7119, 0.7041, 0.7058, 0.7046, 0.6855, 0.6867
]
warm_loss = [
    1.9234, 1.3403, 0.9746, 0.7882, 0.6851, 0.6052, 0.5720, 0.5274, 0.5064, 0.4862,
    0.4549, 0.4550, 0.4393, 0.4346, 0.4285, 0.4265, 0.3921, 0.3987, 0.4051, 0.3922,
    0.3671, 0.3884, 0.3796, 0.3817, 0.3713, 0.3898, 0.3779, 0.3727, 0.3662, 0.3466,
    0.3547, 0.3536, 0.3394, 0.3428, 0.3377, 0.3304, 0.3454, 0.3402, 0.3406, 0.3295,
    0.3546, 0.3347, 0.3366, 0.3314, 0.3258, 0.3208, 0.3162, 0.3126, 0.3312, 0.3190
]

# ================= 2. 绘图代码 =================

epochs = range(1, 51)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=300)

# --- 子图1: Loss 对比 (验证稳定性) ---
# 注意：Warm Start 的初始 Loss 很高(1.92)，这是因为 Loss 定义可能不同
# 但它的下降速度非常快。
ax1.plot(epochs, cold_loss, 'b--', label='Cold Start (Mixed Loss)', linewidth=1.5)
ax1.plot(epochs, warm_loss, 'r-', label='Warm Start Stage 1 (Pure Dice)', linewidth=1.5)
ax1.set_title('Training Loss Convergence: Cold vs. Warm Start', fontsize=12, fontweight='bold')
ax1.set_ylabel('Training Loss', fontsize=10)
ax1.set_xlabel('Epochs', fontsize=10)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# --- 子图2: Dice 对比 (验证性能) ---
# 這是最核心的证据！
ax2.plot(epochs, cold_dice, 'b-o', label='Cold Start (Mixed Loss)', markersize=3, linewidth=1)
ax2.plot(epochs, warm_dice, 'r-s', label='Warm Start Stage 1 (Pure Dice)', markersize=3, linewidth=1)

# 标注最高点
cold_max = max(cold_dice)
cold_epoch = cold_dice.index(cold_max) + 1
warm_max = max(warm_dice)
warm_epoch = warm_dice.index(warm_max) + 1

ax2.annotate(f'Cold Best: {cold_max:.4f}', xy=(cold_epoch, cold_max), xytext=(cold_epoch, cold_max+0.05),
             arrowprops=dict(facecolor='blue', shrink=0.05), fontsize=9, color='blue')
ax2.annotate(f'Warm Best: {warm_max:.4f}', xy=(warm_epoch, warm_max), xytext=(warm_epoch, warm_max-0.1),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=9, color='red')

ax2.set_title('Validation Dice Score: The Impact of Training Strategy', fontsize=12, fontweight='bold')
ax2.set_ylabel('Dice Score (Higher is Better)', fontsize=10)
ax2.set_xlabel('Epochs', fontsize=10)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('strategy_comparison.png')
print("✅ 图表已保存为 strategy_comparison.png")
print(f"Cold Start Best: Ep {cold_epoch} = {cold_max}")
print(f"Warm Start Best: Ep {warm_epoch} = {warm_max}")
plt.show()