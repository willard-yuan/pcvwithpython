# -*- coding: utf-8 -*-
from pylab import *
from PIL import Image
from PCV.localdescriptors import harris

# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)

"""
Example of detecting Harris corner points (Figure 2-1 in the book).
"""

# 读入图像
im = array(Image.open('../data/empire.jpg').convert('L'))

# 检测harris角点
harrisim = harris.compute_harris_response(im)

# Harris响应函数
harrisim1 = 255 - harrisim

figure()
gray()

#画出Harris响应图
subplot(141)
imshow(harrisim1)
print harrisim1.shape
axis('off')
title(u'Harris响应图', fontproperties=font)

filtered_coords = harris.get_harris_points(harrisim, 6, threshold=0.01)
subplot(142)
imshow(im)
print im.shape
plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
axis('off')
title(u'阈值=0.01', fontproperties=font)

filtered_coords = harris.get_harris_points(harrisim, 6, threshold=0.05)
subplot(143)
imshow(im)
plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
axis('off')
title(u'阈值=0.05', fontproperties=font)

filtered_coords = harris.get_harris_points(harrisim, 6, threshold=0.1)
subplot(144)
imshow(im)
plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
axis('off')
title(u'阈值=0.1', fontproperties=font)

#原书采用的PCV中PCV harris模块
#harris.plot_harris_points(im, filtered_coords)

# plot only 200 strongest
# harris.plot_harris_points(im, filtered_coords[:200])

show()