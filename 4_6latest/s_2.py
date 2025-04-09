import matplotlib.pyplot as plt
import csv
import numpy as np

# 使用 matplotlib 自带的样式
plt.style.use('ggplot')

def plot_performance():
    sizes, times = [], []
    
    with open('t2.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            sizes.append(int(row[0]))
            times.append(float(row[1]))

    plt.figure(figsize=(12, 7))
    plt.plot(sizes, times, 'o-', color='#2c7bb6', linewidth=3, markersize=8)
    
    plt.title('Matrix Multiplication Performance (ColMajor Optimized)', fontsize=14)
    plt.xlabel('Matrix Size (N x N)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    
    plt.xticks(np.arange(100, 1100, 100), rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('t2.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_performance()