# 演示程序所用到的功能代码

import os
import numpy as np
import cv2
import glob


stereoCam = None # 双目相机配置信息存储

# 判断是否存在双目相机
def judge_camera():
    main = 'Judge_Camera.exe'
    rs = os.popen(main)
    a = rs.read()
    sa = a.split('\n')
    for line in sa:
        if len(line) >= 7:
            if line[:7] == '@camera':
                camera = line[7:].split(":")
                if camera[1] != 'Integrated Webcam':
                    return [camera[0],camera[1]]
    return 'wrong'


# 定义一个双目相机参数类
class stereoCamera(object):
    def __init__(self,cam_matrix_left,cam_matrix_right,distortion_l,distortion_r,R,T,photo_shape):
        self.cam_matrix_left = cam_matrix_left
        self.cam_matrix_right = cam_matrix_right
        self.distortion_l = distortion_l
        self.distortion_r = distortion_r
        self.R = R
        self.T = T
        self.photo_shape = photo_shape
        rectify_scale = 1
        RL,RR,PL,PR,self.Q,roiL,roiR = cv2.stereoRectify(self.cam_matrix_left,self.distortion_l,self.cam_matrix_right,self.distortion_r,
                                                    self.photo_shape,self.R,self.T,
                                                    rectify_scale,(0, 0))
        self.Left_Stereo_Map = cv2.initUndistortRectifyMap(self.cam_matrix_left,self.distortion_l,
                                                      RL,PL,self.photo_shape,cv2.CV_16SC2)
        self.Right_Stereo_Map = cv2.initUndistortRectifyMap(self.cam_matrix_right,self.distortion_r,
                                                       RR,PR,self.photo_shape,cv2.CV_16SC2)

# 使用opencv进行相机标定
def calibrate_opencv(width,length,boardlength,imgpath1,imgpath2):
    imgLs = glob.glob(imgpath1+'/*.jpg')
    imgRs = glob.glob(imgpath2+'/*.jpg')
    if not (imgLs and imgRs):
        imgLs = glob.glob(imgpath1 + '/*.png')
        imgRs = glob.glob(imgpath2 + '/*.png')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    width = int(width)
    length = int(length)
    boardlength = int(boardlength)
    objP = np.zeros(shape=(width * length, 3), dtype = np.float32)
    for i in range(width * length):
        objP[i][0] = i % width
        objP[i][1] = int(i / width)
    objPoint = objP * boardlength

    objPoints = []
    imgPointsL = []
    imgPointsR = []
    for imgL,imgR in zip(imgLs,imgRs):
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
    a = (a[1],a[0])
    retL, cameraMatrixL, distMatrixL, RL, TL = cv2.calibrateCamera(objPoints, imgPointsL, a, None, None)
    retR, cameraMatrixR, distMatrixR, RR, TR = cv2.calibrateCamera(objPoints, imgPointsR, a, None, None)

    retS, mLS, dLS, mRS, dRS, R, T, E, F = cv2.stereoCalibrate(objPoints, imgPointsL, imgPointsR, cameraMatrixL,
                                                               distMatrixL, cameraMatrixR, distMatrixR, a,
                                                               criteria_stereo, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    global stereoCam
    stereoCam = stereoCamera(mLS, mRS, dLS, dRS, R, T, a)
    print('done')

# 使用matlab进行相机标定
def calibrate_matlab(boardlength,imgpath1,imgpath2):
    imgpath1 = imgpath1.replace('\\','/')
    imgpath2 = imgpath2.replace('\\','/')
    import matlab
    import matlab.engine
    engine = matlab.engine.start_matlab()
    params = engine.stereoCalibrate(imgpath1,imgpath2,float(boardlength))
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
    a = (int(params[-1][0][1]),int(params[-1][0][0]))
    global stereoCam
    stereoCam = stereoCamera(mLS, mRS, dLS, dRS, R, T, a)
    print('done')

# 获得图片深度图像和三维坐标
def get_depthIMG(imgL,imgR,params):
    if imgL.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    SGBM_blockSize = params[1]
    SGBM_num = params[0]
    min_disp = params[5]
    num_disp = SGBM_num * 16
    uniquenessRatio = params[4]
    speckleRange = params[3]
    speckleWindowSize = params[2]
    disp12MaxDiff = 200
    preFilterCap = params[6]
    P1 = 8 * img_channels * SGBM_blockSize ** 2
    P2 = 4 * P1
    global stereoCam
    # 对图像进行重构
    imgl_rectified = cv2.remap(imgL, stereoCam.Left_Stereo_Map[0], stereoCam.Left_Stereo_Map[1], cv2.INTER_LINEAR)
    imgr_rectified = cv2.remap(imgR, stereoCam.Right_Stereo_Map[0], stereoCam.Right_Stereo_Map[1], cv2.INTER_LINEAR)
    # 将图像转为灰度图
    imgL = cv2.cvtColor(imgl_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgr_rectified, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,  # 最小的视差值
        numDisparities=num_disp,  # 视差范围
        blockSize=SGBM_blockSize,  # 匹配块大小（SADWindowSize）
        uniquenessRatio=uniquenessRatio,  # 视差唯一性百分比
        speckleRange=speckleRange,  # 视差变化阈值
        speckleWindowSize=speckleWindowSize,
        disp12MaxDiff=disp12MaxDiff,  # 左右视差图的最大容许差异
        preFilterCap=preFilterCap,
        P1=P1,  # 惩罚系数
        P2=P2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    # 得到视差图
    disparity = stereo.compute(imgL, imgR)
    # 转换为深度图
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., stereoCam.Q)
    return disp, threeD