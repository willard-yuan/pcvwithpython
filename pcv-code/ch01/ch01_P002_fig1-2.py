# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *

pil_im = Image.open('../data/empire.jpg')
figure()
gray()
subplot(121)
axis('off')
imshow(pil_im)

pil_im = Image.open('../data/empire.jpg').convert('L')
subplot(122)
axis('off')
imshow(pil_im)

show()