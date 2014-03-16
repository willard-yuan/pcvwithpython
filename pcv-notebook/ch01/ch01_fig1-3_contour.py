 # -*- coding: utf-8 -*-
from PIL import Image
from pylab import *

im = array(Image.open('../data/empire.jpg').convert('L'))  # 打开图像，并转成灰度图像

figure()
gray()
contour(im, origin='image')
axis('equal')
axis('off')

figure()
hist(im.flatten(), 128)
show()
