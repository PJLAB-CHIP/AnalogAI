import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Generate synthetic data
x = np.linspace(-3, 3, 101)
y = np.linspace(-3, 3, 101)
X, Y = np.meshgrid(x, y)
# Z = np.sin(np.sqrt(X**2 + Y**2)) - np.cos(np.sqrt(X**2 + Y**2))

# mu = [0, 0]  # 均值
# sigma = [[0.1, 0], [0, 0.1]]  # 协方差矩阵

# Create the plot
fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot(111, projection='3d')
for _ in range(100):
    mu = [0, 0]  # 均值
    sigma = [[0.1, 0], [0, 0.1]]  # 协方差矩阵
    mu = np.random.uniform(-3, 3, 2)  # 随机生成均值，范围为[-1, 1]
    # sigma = np.random.uniform(0, 0.1, (2, 2))  # 随机生成方差，范围为[0.1, 0.5]
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    Z = np.exp(-0.5 * np.einsum('...k,kl,...l->...', pos - mu, np.linalg.inv(sigma), pos - mu))
    # Plot the surface with the 'coolwarm' colormap
    ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')

# Add a color bar to show the height mapping to color
# color_bar = fig.colorbar(surf, shrink=0.5, aspect=10)

# Set view angle for better perspective
ax.view_init(30, 225)
plt.savefig('./pictures/rugged_1.png')
plt.show()