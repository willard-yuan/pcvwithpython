from PIL import Image
from pylab import *
from PCV.localdescriptors import sift
#import imtools

imname = '../data/empire.jpg'
im1 = array(Image.open(imname).convert('L'))
sift.process_image(imname, 'empire.sift')
l1, d1 = sift.read_features_from_file('empire.sift')

figure()
gray()
sift.plot_features(im1, l1, circle=False)
show()
