# 使用matlab标定测试

import matlab
import matlab.engine
import numpy as np
import scipy.io
import os,sys,cv2
from PIL import Image
from matplotlib import pyplot as plt

import draw_polar
engine = matlab.engine.start_matlab()
# leftimages = engine.imageDatastore(,
#                                    'FileExtensions', {'.jpg', '.png'});
# rightimages = engine.imageDatastore(,
#                                     'FileExtensions',{'.jpg','.png'});
# print(type(leftimages))
# [imagePoints,boardSize] = engine.detectCheckboardPoints(leftimages.Files,rightimages.Files);
imgpath1 = r"D:\Codes\PyCharm\graduate_project\pics\left"
imgpath2 = r'D:\Codes\PyCharm\graduate_project\pics\right'
params = engine.stereoCalibrate(imgpath1,imgpath2,float(26))
mLS = np.array(params[2]).T
mRS = np.array(params[5]).T
R = np.array(params[0]).T
T = np.array(params[1])[0]
mLK = np.array(params[3])
mLP = np.array(params[4])
mRK = np.array(params[6])
mRP = np.array(params[7])
dLS = np.array([mLK[0][0],mLK[0][1],mLP[0][0],mLP[0][1],mLK[0][2]])
dRS = np.array([mRK[0][0],mRK[0][1],mRP[0][0],mRP[0][1],mRK[0][2]])
print(mLS)
print('--------')
print(mRS)
print('--------')
print(dLS)
print('--------')
print(dRS)
print('--------')
print(R)
print(type(R))
print(np.matrix(R))
print(type(np.matrix(R)))
print('--------')
print(T)
# 利用stereoRectify()计算立体校正的映射矩阵
rectify_scale = 1  # 设置为0的话，对图片进行剪裁，设置为1则保留所有原图像像素
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(mLS, dLS, mRS, dRS,
                                                  (640,480), R, T,
                                                  rectify_scale, (0, 0))
# 利用initUndistortRectifyMap函数计算畸变矫正和立体校正的映射变换，实现极线对齐。
Left_Stereo_Map = cv2.initUndistortRectifyMap(mLS, dLS, RL, PL,
                                              (640,480), cv2.CV_16SC2)

Right_Stereo_Map = cv2.initUndistortRectifyMap(mRS, dRS, RR, PR,
                                               (640,480), cv2.CV_16SC2)

frameR = cv2.imread(r'D:\Codes\PyCharm\graduate_project\pics\right\right2' + '.jpg', 0)
frameL = cv2.imread(r"D:\Codes\PyCharm\graduate_project\pics\left\left2" + '.jpg', 0)

draw_polar.draw_polar(frameL,frameR)

im_L = Image.fromarray(frameL)  # numpy 转 image类
im_R = Image.fromarray(frameR)  # numpy 转 image 类
width = im_L.size[0] * 2
height = im_L.size[1]
img_origin = Image.new('RGBA', (width, height))
img_origin.paste(im_L, box=(0, 0))
img_origin.paste(im_R, box=(640, 0))
plt.imshow(img_origin)
plt.show()
Left_rectified = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4,
                           cv2.BORDER_CONSTANT, 0)  # 使用remap函数完成映射
# im_L = Image.fromarray(Left_rectified)  # numpy 转 image类

Right_rectified = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4,
                            cv2.BORDER_CONSTANT, 0)
# im_R = Image.fromarray(Right_rectified)  # numpy 转 image 类

draw_polar.draw_polar(Left_rectified,Right_rectified)
# # 创建一个能同时并排放下两张图片的区域，后把两张图片依次粘贴进去
# width = im_L.size[0] * 2
# height = im_L.size[1]
#
# img_compare = Image.new('RGBA', (width, height))
# img_compare.paste(im_L, box=(0, 0))
# img_compare.paste(im_R, box=(640, 0))
# plt.imshow(img_origin)
# plt.show()
# # 在已经极线对齐的图片上均匀画线
# for i in range(1, 20):
#     len = 480 / 20
#     plt.axhline(y=i * len, color='r', linestyle='-')
# plt.imshow(img_compare)
# plt.savefig(r'result0' + '.jpg')
# plt.show()

