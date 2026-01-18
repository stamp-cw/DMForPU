import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def get_mat_stats(file_path):
    """
    获取 .mat 文件的最大值减最小值统计信息
    """
    try:
        # 读取 .mat 文件
        data = loadmat(file_path)

        # 提取所有变量
        max_diff_all_files = []

        for key, value in data.items():
            # 排除掉Matlab内部的属性
            if key.startswith("__"):
                continue

            # 计算每个变量的最大值减最小值
            max_min_diff = np.max(value) - np.min(value)
            max_diff_all_files.append(max_min_diff)  # 收集最大值减最小值的结果

        return max_diff_all_files
    except Exception as e:
        print(f"无法读取文件 {file_path}: {e}")
        return None

def analyze_mat_folder(folder_path):
    """
    遍历文件夹，分析所有 .mat 文件
    """
    max_diff_all_files = []

    # 遍历文件夹下所有 .mat 文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mat"):
                file_path = os.path.join(root, file)
                print(f"分析文件: {file_path}")

                # 获取该文件的最大值减最小值统计信息
                max_diff = get_mat_stats(file_path)

                if max_diff:
                    max_diff_all_files.extend(max_diff)  # 收集所有文件的 max - min 差值

    return max_diff_all_files

def plot_histogram(max_diff_all_files):
    """
    绘制最大值减最小值的直方图
    """
    # 设置直方图的 bin 数量
    bin_size = 1
    min_diff = np.min(max_diff_all_files)
    max_diff = np.max(max_diff_all_files)

    # 使用 histogram 将数据按 bin_size 分 bin
    hist, bins = np.histogram(max_diff_all_files, bins=np.arange(min_diff, max_diff + bin_size, bin_size))

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='black', align='edge')
    plt.title("最大值减最小值分布")
    plt.xlabel("最大值减最小值")
    plt.ylabel("频率")
    plt.show()

def print_summary_stats(max_diff_all_files, total_files):
    """
    打印汇总统计信息
    """
    if max_diff_all_files:
        global_max_diff = np.max(max_diff_all_files)
        global_min_diff = np.min(max_diff_all_files)
        print(f"\n所有文件中最大值减最小值的范围: ({global_min_diff}, {global_max_diff})")

    # 汇总统计信息
    print("\n汇总统计信息:")
    print(f"总文件数: {total_files}")
    print(f"最大值减最小值的范围: ({global_min_diff}, {global_max_diff})")

if __name__ == "__main__":
    folder_path = input("请输入文件夹路径: ").strip()

    # 分析文件夹中的所有 .mat 文件
    max_diff_all_files = analyze_mat_folder(folder_path)

    # 打印汇总统计信息
    print_summary_stats(max_diff_all_files, total_files=len(max_diff_all_files))

    # 绘制最大值减最小值的分布直方图
    plot_histogram(max_diff_all_files)
