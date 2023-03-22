# 相机相关参数

import cv2
import numpy as np

left_camera_matrix = np.array([[417.9438136,   0.,         301.15574194],
                             [  0.,         419.57047477, 237.25022416 ],
                             [  0.,           0.,           1.]])
right_distortion = np.array([-0.03759185,0.06944604,0.,0.,0.])

right_camera_matrix = np.array([[413.96532735,   0.,         298.72869061],
                                 [  0.,         415.17329915, 236.67635577],
                                 [  0.,           0.,           1.        ]])
left_distortion = np.array([-0.06908662,0.14306356,0.,0.,0.])


R = np.matrix([[ 9.99912336e-01,  5.09635644e-04, -1.32310476e-02],
 [-5.28283554e-04,  9.99998872e-01, -1.40594862e-03],
 [ 1.32303162e-02,  1.41281511e-03,  9.99911477e-01]])

T = np.array([-1.19067721e+02,-9.23268919e-03 ,-1.17703421e+00])
size = (640, 480) # 图像尺寸

# 进行立体更正
rectify_scale = 1
RL,RR,PL,PR,Q,roiL,roiR = cv2.stereoRectify(left_camera_matrix,left_distortion,right_camera_matrix,right_distortion,
                                            size,R,T,
                                            rectify_scale,(0, 0))
# 计算更正map
Left_Stereo_Map = cv2.initUndistortRectifyMap(left_camera_matrix,left_distortion,
                                                      RL,PL,size,cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(right_camera_matrix,right_distortion,
                                               RR,PR,size,cv2.CV_16SC2)
