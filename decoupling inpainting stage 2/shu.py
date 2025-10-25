import matplotlib.pyplot as plt
from  src.dataset import create_line_mask
# 参数设置
width, height = 512, 512
num_lines = 5
line_width_range = (5, 10)
angle_range = (0, 180)
gap_range = (20, 50)
randomize = True

# 生成掩码
mask = create_line_mask(width, height, num_lines, line_width_range, angle_range, gap_range, randomize)

# 可视化掩码
plt.imshow(mask, cmap='gray')
plt.title("Line Mask (Missing Area = 1)")
plt.show()