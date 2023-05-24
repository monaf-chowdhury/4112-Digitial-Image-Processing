import numpy
import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image
from math import cos, sin


def getGrayColor(rgb):
    return rgb[0]


def setGrayColor(color):
    return [color, color, color]


img = Image.open('03.jpg')
img = numpy.asarray(img)

c = 1
y = float(input("input gamma :"))
# copy list not reference
pwl = deepcopy(img)


min = 99999999999999
max = 0

for i in range(len(img)):
    for j in range(len(img[i])):
        s = (c*img[i][j][0])**y
        if(s > max):
            max = s
        if(s < min):
            min = s

for i in range(len(img)):
    for j in range(len(img[i])):
        s = (c*img[i][j][0])**y
        s = (s-min)/(max-min)
        s = s*254
        pwl[i][j] = setGrayColor(s)

print('min:',min,'max:',max)

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.subplot(2, 2, 2)
plt.imshow(pwl)


plt.show()