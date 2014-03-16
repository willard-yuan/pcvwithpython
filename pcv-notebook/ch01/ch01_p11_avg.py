from pylab import *
from PCV.tools import imtools

avg = imtools.compute_average(['../data/avg/empirew560h800.jpg', '../data/avg/climbing_1_smallw560h800.jpg', '../data/avg/sunsetw560h800.jpg'])

imshow(avg)
axis('off')
show()
