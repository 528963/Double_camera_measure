# SGBM算法实验

import numpy as np
import cv2
import camera_configs
import time
cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.namedWindow("config")
cv2.namedWindow('depth')
cv2.resizeWindow("config",640,480)
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 600, 0)
cv2.createTrackbar("num", "config", 2, 60, lambda x: None)
cv2.createTrackbar("blockSize", "config", 3, 11, lambda x: None)
cv2.createTrackbar("SpeckleWindowSize", "config", 50, 200, lambda x: None)
cv2.createTrackbar("SpeckleRange", "config", 1, 255, lambda x: None)
cv2.createTrackbar("UniquenessRatio", "config", 1, 255, lambda x: None)
cv2.createTrackbar("MinDisparity", "config", 0, 255, lambda x: None)
cv2.createTrackbar("PreFilterCap", "config", 1, 65, lambda x: None)  # 注意调节的时候这个值必须是奇数
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# 添加点击事件，打印当前点的距离
def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print(threeD[y][x])
cv2.setMouseCallback("depth", callbackFunc, None)
while True:
    img1 = cv2.imread("./pics/imgs/bottle0.jpg")
    img2 = cv2.imread("./pics/imgs/bottle1.jpg")

    if img1.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3

    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(img1, camera_configs.Left_Stereo_Map[0], camera_configs.Left_Stereo_Map[1],
                               cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(img2, camera_configs.Right_Stereo_Map[0], camera_configs.Right_Stereo_Map[1],
                               cv2.INTER_LINEAR)
    # cv2.imshow("rect1", img1_rectified)
    # cv2.imshow("rect2", img2_rectified)

    # 将图片置为灰度图，为StereoSGBM作准备
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray1", imgL)
    # cv2.imshow("gray2", imgR)


    # # 两个trackbar用来调节不同的参数查看效果
    # num = cv2.getTrackbarPos("num", "depth")
    # blockSize = cv2.getTrackbarPos("blockSize", "depth")
    # 通过bar来获取到当前的参数
    num = cv2.getTrackbarPos("num", "config")
    SpeckleWindowSize = cv2.getTrackbarPos("SpeckleWindowSize", "config")
    SpeckleRange = cv2.getTrackbarPos("SpeckleRange", "config")
    blockSize = cv2.getTrackbarPos("blockSize", "config")
    UniquenessRatio = cv2.getTrackbarPos("UniquenessRatio", "config")
    MinDisparity = cv2.getTrackbarPos("MinDisparity", "config")
    PreFilterCap = cv2.getTrackbarPos("PreFilterCap", "config")
    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize < 5:
        blockSize = 5
    # SGBM_blockSize = blockSize  # 一个匹配块的大小,大于1的奇数
    # SGBM_num = num
    # min_disp = 0  # 最小的视差值，通常情况下为0
    # num_disp = SGBM_num * 16  # 192 - min_disp #视差范围，即最大视差值和最小视差值之差，必须是16的倍数。
    # # blockSize = blockSize #匹配块大小（SADWindowSize），必须是大于等于1的奇数，一般为3~11
    # uniquenessRatio = 6  # 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
    # speckleRange = 2  # 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
    # speckleWindowSize = 60  # 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50-200的范围内。
    disp12MaxDiff = 200  # 左右视差图的最大容许差异（超过将被清零），默认为 -1，即不执行左右视差检查。
    # P1 = 600  # 惩罚系数，一般：P1=8*通道数*SADWindowSize*SADWindowSize，P2=4*P1
    # P2 = 2400  # p1控制视差平滑度，p2值越大，差异越平滑
    SGBM_blockSize = blockSize
    SGBM_num = num
    num_disp = SGBM_num * 16
    speckleRange = SpeckleRange
    min_disp = MinDisparity
    uniquenessRatio = UniquenessRatio
    speckleWindowSize = SpeckleWindowSize
    preFilterCap = PreFilterCap
    P1 = 8 * img_channels * blockSize ** 2
    P2 = 4 * P1
    # 根据SGBM方法生成差异图
    start = time.clock()
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
    disparity = stereo.compute(imgL, imgR)

    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    bf_time = (time.clock() - start)
    print("bf_time:", '% 4f' % (bf_time*1000))
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., camera_configs.Q)

    cv2.imshow("left", img1_rectified)
    cv2.imshow("right", img2_rectified)
    cv2.imshow("depth", disp)

    key = cv2.waitKey(1)
# if key == ord("q"):
#     break
# elif key == ord("s"):
#     cv2.imwrite("./snapshot/SGBM_left.jpg", imgL)
#     cv2.imwrite("./snapshot/SGBM_right.jpg", imgR)
#     cv2.imwrite("./snapshot/SGBM_depth.jpg", disp)


cv2.destroyAllWindows()

# # 预处理
# def preprocess(img1, img2):
#     # 彩色图->灰度图
#     if (img1.ndim == 3):
#         img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
#     if (img2.ndim == 3):
#         img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
#     # 直方图均衡
#     img1 = cv2.equalizeHist(img1)
#     img2 = cv2.equalizeHist(img2)
#
#     return img1, img2
#
#
# def stereoMatchSGBM(left_img,right_img,down_scale=False):
#
#     if left_img.ndim == 2:
#         img_channels = 1
#     else:
#         img_channels = 3
#     blockSize = 3
#     minDisparity = 0
#     numDisparities = 128
#     P1 = 8 * img_channels * blockSize ** 2
#     P2 = 4 * P1
#     disp12MaxDiff = 1
#     preFilterCap = 63
#     uniquenessRatio = 15
#     speckleWindowSize = 100
#     speckleRange = 1
#     mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
#
#     left_matcher = cv2.StereoSGBM_create(
#         minDisparity= minDisparity,
#         numDisparities= numDisparities,
#         blockSize= blockSize,
#         disp12MaxDiff= disp12MaxDiff,
#         preFilterCap= preFilterCap,
#         P1= P1,
#         P2= P2,
#         uniquenessRatio= uniquenessRatio,
#         speckleWindowSize= speckleWindowSize,
#         speckleRange= speckleRange,
#         mode= mode
#     )
#
#     right_matcher = cv2.StereoSGBM_create(
#         minDisparity= -128,
#         numDisparities= numDisparities,
#         blockSize= blockSize,
#         disp12MaxDiff= disp12MaxDiff,
#         preFilterCap= preFilterCap,
#         P1= P1,
#         P2= P2,
#         uniquenessRatio= uniquenessRatio,
#         speckleWindowSize= speckleWindowSize,
#         speckleRange= speckleRange,
#         mode= mode
#     )
#
#     size = (left_img.shape[1], left_img.shape[0])
#     if down_scale == False:
#         disparity_left = left_matcher.compute(left_img, right_img)
#         disparity_right = right_matcher.compute(right_img,left_img)
#
#     else:
#         left_image_down = cv2.pyrDown(left_img)
#         right_image_down = cv2.pyrDown(right_img)
#         factor = left_img.shape[1] / left_image_down.shape[1]
#
#         disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
#         disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
#         disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
#         disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
#         disparity_left = factor * disparity_left
#         disparity_right = factor * disparity_right
#
#     trueDisp_left = disparity_left.astype(np.float32) / 16.
#     trueDisp_right = disparity_right.astype(np.float32) / 16.
#
#     return trueDisp_left, trueDisp_right
#
#
# def getDepthMapWithQ(disparityMap: np.ndarray, Q: np.ndarray) -> np.ndarray:
#     points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
#     depthMap = points_3d[:, :, 2]
#     reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
#     depthMap[reset_index] = 0
#
#     return depthMap.astype(np.float32)
#
#
# imgL = cv2.imread("./pics/left/left0.jpg")
# imgR = cv2.imread("./pics/right/right0.jpg")
# imgL,imgR = preprocess(imgL,imgR)
# disp,_ = stereoMatchSGBM(imgL,imgR,False)
# depthMap = getDepthMapWithQ(disp,camera_configs.Q)
# minDepth = np.min(depthMap)
# maxDepth = np.max(depthMap)
# print(minDepth, maxDepth)
# depthMapVis = (255.0 *(depthMap - minDepth)) / (maxDepth - minDepth)
# depthMapVis = depthMapVis.astype(np.uint8)
# cv2.imshow("DepthMap", depthMapVis)
# cv2.waitKey(0)

#
# # 创建用于显示深度的窗口和调节参数的bar
# cv2.namedWindow("depth")
# cv2.namedWindow("left")
# cv2.namedWindow("right")
# cv2.moveWindow("left", 0, 0)
# cv2.moveWindow("right", 600, 0)
#
# # 创建用于显示深度的窗口和调节参数的bar
# # cv2.namedWindow("depth")
# cv2.namedWindow("config", cv2.WINDOW_NORMAL)
# cv2.moveWindow("left", 0, 0)
# cv2.moveWindow("right", 600, 0)
#
# cv2.createTrackbar("num", "config", 0, 60, lambda x: None)
# cv2.createTrackbar("blockSize", "config", 30, 255, lambda x: None)
# cv2.createTrackbar("SpeckleWindowSize", "config", 1, 10, lambda x: None)
# cv2.createTrackbar("SpeckleRange", "config", 1, 255, lambda x: None)
# cv2.createTrackbar("UniquenessRatio", "config", 1, 255, lambda x: None)
# cv2.createTrackbar("TextureThreshold", "config", 1, 255, lambda x: None)
# cv2.createTrackbar("UniquenessRatio", "config", 1, 255, lambda x: None)
# cv2.createTrackbar("MinDisparity", "config", 0, 255, lambda x: None)
# cv2.createTrackbar("PreFilterCap", "config", 1, 65, lambda x: None)  # 注意调节的时候这个值必须是奇数
# cv2.createTrackbar("MaxDiff", "config", 1, 400, lambda x: None)
#
#
# # 添加点击事件，打印当前点的距离
# def callbackFunc(e, x, y, f, p):
#     if e == cv2.EVENT_LBUTTONDOWN:
#         print(threeD[y][x])
#         if abs(threeD[y][x][2]) < 3000:
#             print("当前距离:" + str(abs(threeD[y][x][2])))
#         else:
#             print("当前距离过大或请点击色块的位置")
#
#
# cv2.setMouseCallback("depth", callbackFunc, None)
#
# # 初始化计算FPS需要用到参数 注意千万不要用opencv自带fps的函数，那个函数得到的是摄像头最大的FPS
# frame_rate_calc = 1
# freq = cv2.getTickFrequency()
# font = cv2.FONT_HERSHEY_SIMPLEX
#
# imageCount = 1
#
# while True:
#     t1 = cv2.getTickCount()
#     # ret1, frame1 = cam1.read()
#     # ret1, frame2 = cam2.read()
#     img1 = cv2.imread("./pics/imgs/left0.jpg")
#     img2 = cv2.imread("./pics/imgs/right0.jpg")
#     # if not ret1:
#     #     print("camera is not connected!")
#     #     break
#
#     if img1.ndim == 2:
#         img1_channels = 1
#     else:
#         img1_channels = 3
#
#     # 这里的左右两个摄像头的图像是连在一起的，所以进行一下分割
#     # frame1 = frame[0:480, 0:640]
#     # frame2 = frame[0:480, 640:1280]
#
#     ####### 深度图测量开始 #######
#     # 立体匹配这里使用BM算法，
#
#     # 根据标定数据对图片进行重构消除图片的畸变
#     img1_rectified = cv2.remap(img1, camera_configs.Left_Stereo_Map[0], camera_configs.Left_Stereo_Map[1], cv2.INTER_LINEAR)
#     img2_rectified = cv2.remap(img2, camera_configs.Right_Stereo_Map[0], camera_configs.Right_Stereo_Map[1], cv2.INTER_LINEAR)
#
#     # 如有些版本 remap()的图是反的 这里对角翻转一下
#     # img1_rectified = cv2.flip(img1_rectified, -1)
#     # img2_rectified = cv2.flip(img2_rectified, -1)
#
#     # 将图片置为灰度图，为StereoBM作准备，BM算法只能计算单通道的图片，即灰度图
#     # 单通道就是黑白的，一个像素只有一个值如[123]，opencv默认的是BGR(注意不是RGB), 如[123,4,134]分别代表这个像素点的蓝绿红的值
#     imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
#     imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
#
#     out = np.hstack((img1_rectified, img2_rectified))
#     for i in range(0, out.shape[0], 30):
#         cv2.line(out, (0, i), (out.shape[1], i), (0, 255, 0), 1)
#     cv2.imshow("epipolar lines", out)
#
#     # 通过bar来获取到当前的参数
#     # BM算法对参数非常敏感，一定要耐心调整适合自己摄像头的参数，前两个参数影响大 后面的参数也要调节
#     num = cv2.getTrackbarPos("num", "config")
#     SpeckleWindowSize = cv2.getTrackbarPos("SpeckleWindowSize", "config")
#     SpeckleRange = cv2.getTrackbarPos("SpeckleRange", "config")
#     blockSize = cv2.getTrackbarPos("blockSize", "config")
#     UniquenessRatio = cv2.getTrackbarPos("UniquenessRatio", "config")
#     TextureThreshold = cv2.getTrackbarPos("TextureThreshold", "config")
#     MinDisparity = cv2.getTrackbarPos("MinDisparity", "config")
#     PreFilterCap = cv2.getTrackbarPos("PreFilterCap", "config")
#     MaxDiff = cv2.getTrackbarPos("MaxDiff", "config")
#     if blockSize % 2 == 0:
#         blockSize += 1
#     if blockSize < 5:
#         blockSize = 5
#
#     # 根据BM算法生成深度图的矩阵，也可以使用SGBM，SGBM算法的速度比BM慢，但是比BM的精度高
#     stereo = cv2.StereoSGBM_create(
#         minDisparity=MinDisparity,  # 最小的视差值
#         numDisparities=num,  # 视差范围
#         blockSize=blockSize,  # 匹配块大小（SADWindowSize）
#         uniquenessRatio=UniquenessRatio,  # 视差唯一性百分比
#         speckleRange=SpeckleRange,  # 视差变化阈值
#         speckleWindowSize=SpeckleWindowSize,
#         # disp12MaxDiff=200,  # 左右视差图的最大容许差异
#         P1=8 * img1_channels * blockSize ** 2,  # 惩罚系数
#         P2=32 * img1_channels * blockSize ** 2,
#     )
#     # stereo.setROI1(camera_configs.roiL)
#     # stereo.setROI2(camera_configs.roiR)
#     stereo.setPreFilterCap(PreFilterCap)
#     stereo.setMinDisparity(MinDisparity)
#     # stereo.setTextureThreshold(TextureThreshold)
#     stereo.setUniquenessRatio(UniquenessRatio)
#     stereo.setSpeckleWindowSize(SpeckleWindowSize)
#     stereo.setSpeckleRange(SpeckleRange)
#     stereo.setDisp12MaxDiff(MaxDiff)
#
#     # 对深度进行计算，获取深度矩阵
#     disparity = stereo.compute(imgL, imgR)
#     # 按照深度矩阵生产深度图
#     disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#
#     # 将深度图扩展至三维空间中，其z方向的值则为当前的距离
#     threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., camera_configs.Q)
#     # # 将深度图转为伪色图，这一步对深度测量没有关系，只是好看而已
#     # fakeColorDepth = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
#     #
#     # cv2.putText(frame1, "FPS: {0:.2f}".format(frame_rate_calc), (30, 50), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
#
#     # 按下S可以保存图片
#     interrupt = cv2.waitKey(10)
#     if interrupt & 0xFF == 27:  # 按下ESC退出程序
#         break
#     if interrupt & 0xFF == ord('s'):
#         # cv2.imwrite('images/left' + '.jpg', frame1)
#         # cv2.imwrite('images/right' + '.jpg', frame2)
#         # cv2.imwrite('images/img1_rectified' + '.jpg', img1_rectified)  # 畸变，注意观察正反
#         # cv2.imwrite('images/img2_rectified' + '.jpg', img2_rectified)
#         cv2.imwrite('images/depth' + '.jpg', disp)
#         # cv2.imwrite('images/fakeColor' + '.jpg', fakeColorDepth)
#         # cv2.imwrite('mages/epipolar' + '.jpg', out)
#
#     ####### 任务1：测距结束 #######
#
#     # 显示
#     # cv2.imshow("frame", frame) # 原始输出，用于检测左右
#     # cv2.imshow("frame1", frame1)  # 左边原始输出
#     # cv2.imshow("frame2", frame2)  # 右边原始输出
#     # cv2.imshow("img1_rectified", img1_rectified)  # 左边矫正后输出
#     # cv2.imshow("img2_rectified", img2_rectified)  # 右边边矫正后输出
#     cv2.imshow("depth", disp)  # 输出深度图及调整的bar
#     # cv2.imshow("fakeColor", fakeColorDepth)  # 输出深度图的伪色图，这个图没有用只是好看
#
#     # 需要对深度图进行滤波将下面几行开启即可 开启后FPS会降低
#     img_medianBlur = cv2.medianBlur(disp, 25)
#     img_medianBlur_fakeColorDepth = cv2.applyColorMap(img_medianBlur, cv2.COLORMAP_JET)
#     img_GaussianBlur = cv2.GaussianBlur(disp, (7, 7), 0)
#     img_Blur = cv2.blur(disp, (5, 5))
#     cv2.imshow("img_GaussianBlur", img_GaussianBlur)  # 右边原始输出
#     cv2.imshow("img_medianBlur_fakeColorDepth", img_medianBlur_fakeColorDepth)  # 右边原始输出
#     cv2.imshow("img_Blur", img_Blur)  # 右边原始输出
#     cv2.imshow("img_medianBlur", img_medianBlur)  # 右边原始输出
#
#     t2 = cv2.getTickCount()
#     time1 = (t2 - t1) / freq
#     frame_rate_calc = 1 / time1
#
# # cam1.release()
# cv2.destroyAllWindows()
