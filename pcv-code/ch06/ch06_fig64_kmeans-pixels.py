"""
Function: figure 6.4
    Clustering of pixels based on their color value using k-means.
Date: 2013-10-27
"""
# coding=utf-8
from scipy.cluster.vq import *
from scipy.misc import imresize
from pylab import *
import Image

steps = 100  # image is divided in steps*steps region
infile = '../data/empire.jpg'
im = array(Image.open(infile))
dx = im.shape[0] / steps
dy = im.shape[1] / steps
# compute color features for each region
features = []
for x in range(steps):
    for y in range(steps):
        R = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 0])
        G = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 1])
        B = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 2])
        features.append([R, G, B])
features = array(features, 'f')     # make into array
# cluster
centroids, variance = kmeans(features, 3)
code, distance = vq(features, centroids)
# create image with cluster labels
codeim = code.reshape(steps, steps)
codeim = imresize(codeim, im.shape[:2], 'nearest')
figure()
ax1 = subplot(121)
ax1.set_title('Image')
axis('off')
imshow(im)
ax2 = subplot(122)
ax2.set_title('Image after clustering')
axis('off')
imshow(codeim)
show()