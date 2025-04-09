import matplotlib.pyplot as plt
import csv

sizes = []
times = []

with open('t1.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots)  # 跳过标题行
    for row in plots:
        sizes.append(int(row[0]))
        times.append(float(row[1]))

plt.figure(figsize=(10, 6))
plt.plot(sizes, times, 'b-o', linewidth=2, markersize=8)
plt.title('Matrix Multiplication Performance')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Execution Time (seconds)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(sizes)
plt.tight_layout()
plt.savefig('performance_plot.png')
plt.show()