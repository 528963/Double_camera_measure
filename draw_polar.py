# 画出图像极线

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

def draw_polar(img1,img2):
    # img2 = cv2.imread(r'D:\Codes\PyCharm\graduate_project\pics\right\right2' + '.jpg', 0)
    # img1 = cv2.imread(r"D:\Codes\PyCharm\graduate_project\pics\left\left2" + '.jpg', 0)

    sift = cv2.SIFT_create()

    # 找到图像关键点
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    def drawlines(img1,img2,lines,pts1,pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r,c = img1.shape
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
            img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
        return img1,img2

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)


    img1= Image.fromarray(img1)  # numpy 转 image类
    # 创建一个能同时并排放下两张图片的区域，后把两张图片依次粘贴进去
    width = img1.size[0] * 2
    height = img1.size[1]

    img5= Image.fromarray(img5)  # numpy 转 image类
    img3= Image.fromarray(img3)  # numpy 转 image类
    img_compare = Image.new('RGBA', (width, height))
    img_compare.paste(img5, box=(0, 0))
    img_compare.paste(img3, box=(640, 0))
    # plt.subplot(121),plt.imshow(img5)
    # plt.subplot(122),plt.imshow(img3)
    plt.imshow(img_compare)
    plt.show()