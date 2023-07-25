import matplotlib.pyplot as plt
import numpy as np
import os

    # QKV_dimensions = [1024, 2048, 4096, 8192, 16384, 32768]

    # RSA_forward_time = [14.97, 22.19, 40.28, 72.97, 158.52, 391.87]
    # BurstAttention_forward_time = [15.90, 23.39, 39.16, 74.55, 163.09, 399.74]
    # Normal_forward_time = [9.17, 11.67, 18.40, 46.36, 157.73, np.nan]

    # RSA_backward_time = [23.41, 48.13, 90.55, 198.33, 445.55, 1057.94]
    # BurstAttention_backward_time = [11.05, 26.42, 52.29, 112.98, 271.00, 748.68]
    # Normal_backward_time = [1.73, 4.10, 18.11, 72.00, np.nan, np.nan]

    # RSA_memory = [1438, 1590, 2082, 3918, 11086, 39246]
    # BurstAttention_memory = [1438, 1518, 1742, 2510, 5454, 16720]
    # Normal_memory = [1536, 2040, 4024, 11832, np.nan, np.nan]
def plot_picture(QKV_dimensions, RSA_forward_time, BurstAttention_forward_time, Normal_forward_time, RSA_backward_time, BurstAttention_backward_time, Normal_backward_time, RSA_memory, BurstAttention_memory, Normal_memory, path="./png"):
    # 绘制前向计算时间的折线图

    plt.figure(figsize=(10, 5))
    plt.plot(QKV_dimensions, RSA_forward_time, marker='o', label='RSA forward time')
    plt.plot(QKV_dimensions, BurstAttention_forward_time, marker='o', label='BurstAttention forward time')
    plt.plot(QKV_dimensions, Normal_forward_time, marker='o', label='Normal forward time')
    plt.xlabel('QKV dimensions')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.title('Forward calculation time for different methods')
    plt.tight_layout()
    save_path = os.path.join(path,f'forward_time.png')
    plt.savefig(save_path)  # 保存图片

    # 绘制反向计算时间的折线图
    plt.figure(figsize=(10, 5))
    plt.plot(QKV_dimensions, RSA_backward_time, marker='o', label='RSA backward time')
    plt.plot(QKV_dimensions, BurstAttention_backward_time, marker='o', label='BurstAttention backward time')
    plt.plot(QKV_dimensions, Normal_backward_time, marker='o', label='Normal backward time')
    plt.xlabel('QKV dimensions')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.title('Forward+Backward calculation time for different methods')
    plt.tight_layout()
    save_path = os.path.join(path,f'forward_backward_time.png')
    plt.savefig(save_path)  # 保存图片

    # 绘制显存占用的折线图
    plt.figure(figsize=(10, 5))
    plt.plot(QKV_dimensions, RSA_memory, marker='o', label='RSA memory')
    plt.plot(QKV_dimensions, BurstAttention_memory, marker='o', label='BurstAttention memory')
    plt.plot(QKV_dimensions, Normal_memory, marker='o', label='Normal memory')
    plt.xlabel('QKV dimensions')
    plt.ylabel('Memory (MiB)')
    plt.legend()
    plt.title('Memory usage for different methods')
    plt.tight_layout()
    save_path = os.path.join(path,f'memory_usage.png')
    plt.savefig('./png/memory_usage.png')  # 保存图片
