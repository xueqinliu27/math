
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_laplace
 
# 参数设置
k = 0.1  # 热扩散系数
t = 0.5  # 时间
x = np.linspace(0, 1, 100)  # 空间范围
u0 = np.exp(-((x - 0.5) ** 2) / 0.02)  # 初始条件
 
# 使用高斯拉普拉斯算子模拟热扩散
u_t = gaussian_laplace(u0, sigma=np.sqrt(k * t), mode='constant', cval=0.0)
 
plt.plot(x, u0, label='Initial')
plt.plot(x, u_t, label='After t = {}'.format(t))
plt.legend()
plt.grid()
plt.show()