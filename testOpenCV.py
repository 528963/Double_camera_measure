# 使用OpenCV标定测试

import numpy as np
import cv2
import glob
imgpath1 = r"D:\Codes\PyCharm\graduate_project\pics\left"
imgpath2 = r'D:\Codes\PyCharm\graduate_project\pics\right'
imgLs = glob.glob(imgpath1 + '/*.jpg')
imgRs = glob.glob(imgpath2 + '/*.jpg')
if not (imgLs and imgRs):
    imgLs = glob.glob(imgpath1 + '/*.png')
    imgRs = glob.glob(imgpath2 + '/*.png')
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
width = 9
length = 6
boardlength = 26
objP = np.zeros(shape=(width * length, 3), dtype=np.float32)
for i in range(width * length):
    objP[i][0] = i % width
    objP[i][1] = int(i / width)
objPoint = objP * boardlength

objPoints = []
imgPointsL = []
imgPointsR = []
for imgL, imgR in zip(imgLs, imgRs):
    imgL = cv2.imread(imgL)
    imgR = cv2.imread(imgR)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    retL, cornersL = cv2.findChessboardCorners(grayL, (width, length), None)
    retR, cornersR = cv2.findChessboardCorners(grayR, (width, length), None)
    if (retL & retR) is True:
        objPoints.append(objPoint)
        cornersL2 = cv2.cornerSubPix(grayL, cornersL, (5, 5), (-1, -1), criteria)
        cornersR2 = cv2.cornerSubPix(grayR, cornersR, (5, 5), (-1, -1), criteria)
        imgPointsL.append(cornersL2)
        imgPointsR.append(cornersR2)
a = cv2.imread(imgLs[0]).shape
a = (a[1], a[0])
retL, cameraMatrixL, distMatrixL, RL, TL = cv2.calibrateCamera(objPoints, imgPointsL, a, None, None)
retR, cameraMatrixR, distMatrixR, RR, TR = cv2.calibrateCamera(objPoints, imgPointsR, a, None, None)

retS, mLS, dLS, mRS, dRS, R, T, E, F = cv2.stereoCalibrate(objPoints, imgPointsL, imgPointsR, cameraMatrixL,
                                                           distMatrixL, cameraMatrixR, distMatrixR, a,
                                                           criteria_stereo, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
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
print('--------')
print(T)