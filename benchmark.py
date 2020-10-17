# encoding=utf-8
# @Time : 2020/10/14 19:44
# @Author : qqyor
# @QQ : 603997262
# @File : performance_test.py
# @Project : Nox_pad
from functools import wraps
import time
def get_mask(width=88,height=88,radius=40):
    grid = np.meshgrid(range(width), range(height))
    return (grid[0]-width/2)**2+(grid[1]-height//2)**2<radius**2
def showtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result= func(*args, **kwargs)
        end_time=time.time()
        print()
        print(f'{func.__name__}耗时： {end_time - start_time} S')
        return result
    return wrapper

import cv2
from PIL import Image
import numpy as np
def cv2_or_PIl_read(file='01.png',times=100):
    start=time.time()
    for i in range(times):
        arr=cv2.imread(file)
    print('cv2 use time',time.time()-start)

    start = time.time()
    for i in range(times):
        arr=np.array(Image.open(file))
    print('PIL use time', time.time() - start)




# mask=get_mask()
# img1=cv2.imread('classes/1.png')
# img1[~mask]=0
# for row in range(5):
#     for col in range(6):
#         img2=cv2.imread(f'balls/{row}{col}.png')
#         img2[~mask] = 0
#         print(round(compare_ssim(img1, img2,full=True,multichannel=True)[0],2),end=' ')
#     print()
# # cv2_or_PIl_read()