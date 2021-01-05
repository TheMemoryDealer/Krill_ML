import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


for filename in os.listdir('../Krill images lateral'):
    img = cv2.imread('../Krill images lateral/{}'.format(filename))
    # print(filename)
    ht, wd, cc= img.shape

    # create new image of desired size and color (blue) for padding
    ww = 1700
    hh = 500
    color = (245,127,56)
    result = np.full((hh,ww,cc), color, dtype=np.uint8)

    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    # copy img image into center of result image
    result[yy:yy+ht, xx:xx+wd] = img

    # view result
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # save result
    cv2.imwrite('../Krill images lateral/{}'.format(filename), result)

