# 标定和拍摄界面

import os,sys
from PyQt5.QtWidgets import QFileDialog
import window.CamStandard as SubWin
from PyQt5 import QtWidgets,QtGui,QtCore
from PyQt5.QtWidgets import QMainWindow
import cv2
import numpy as np
from efficientdet import Efficientdet
from PIL import Image
import tools

class SubMain(QMainWindow,SubWin.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.set_ui()
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 10
        self.slot_init()
        self.PIC_NUM = 0

    def set_ui(self):
        # QMainWindow.__init__(self)
        # SubWin.Ui_MainWindow.__init__(self)
        self.setupUi(self)

    def slot_init(self):
        self.btnOpenCam.clicked.connect(self.button_start_clicked)
        self.btnCatchCam.clicked.connect(self.button_shoot_clicked)
        self.btnClose.clicked.connect(self.button_close_clicked)
        self.timer_camera.timeout.connect(self.cam_shoot)
        self.btnSelectLeftImage.clicked.connect(self.button_sleft_clicked)
        self.btnSelectRightImage.clicked.connect(self.button_sright_clicked)
        self.buttonGroup.buttonClicked.connect(self.button_group_clicked)
        self.btnStartCalib.clicked.connect(self.button_calib_clicked)

    def button_calib_clicked(self):
        self.width = self.ChessBoardWidth.text()
        self.length = self.ChessBoardLength.text()
        self.boardlength = self.RectWidth.text()
        if len(self.width) and len(self.length) and len(self.boardlength) and len(self.length) and len(self.rightpath):
            # a = (int(self.cap.get(3))//2,int(self.cap.get(4)))
            if self.method == 'OpenCV':
                tools.calibrate_opencv(self.width,self.length,self.boardlength,self.leftpath,self.rightpath)
            elif self.method == 'MATLAB':
                tools.calibrate_matlab(self.boardlength,self.leftpath,self.rightpath)
            msg = QtWidgets.QMessageBox.information(self, '标定成功', '标定已经完成！', buttons=QtWidgets.QMessageBox.Ok)
        else:
            msg = QtWidgets.QMessageBox.information(self, '错误', '请确定参数是否全部填写正确！', buttons=QtWidgets.QMessageBox.Ok)

    def button_group_clicked(self):
        self.method = self.buttonGroup.checkedButton().text()
        # print(type(self.method))

    def button_sright_clicked(self):
        filedirpath = QFileDialog.getExistingDirectory(self,'选择右相机图片路径')
        self.rightpath = filedirpath
        self.right_path.setText(self.rightpath)


    def button_sleft_clicked(self):
        filedirpath = QFileDialog.getExistingDirectory(self,'选择左相机图片路径')
        self.leftpath = filedirpath
        self.left_path.setText(self.leftpath)

    def button_shoot_clicked(self):
        if os.path.exists('./pics') == False:
            msg = QtWidgets.QMessageBox.information(self, '操作成功', '未发现指定文件夹，已成功创建！', buttons=QtWidgets.QMessageBox.Ok)
            os.mkdir('./pics')
        else:
            ret,self.frame = self.cap.read()
            # pics = cv2.resize(self.frame, (1280,480))
            pics = self.frame
            pics_left = pics[:, 0:640, :]
            pics_right = pics[:, 640:1280, :]
            pics_path_left = './pics/left/left' + str(self.PIC_NUM) + '.jpg'
            pics_path_right = './pics/right/right' + str(self.PIC_NUM) + '.jpg'
            cv2.imwrite(pics_path_left,pics_left)
            cv2.imwrite(pics_path_right,pics_right)
            msg = QtWidgets.QMessageBox.information(self,'操作成功','已经成功拍摄！',buttons=QtWidgets.QMessageBox.Ok)
            self.PIC_NUM = self.PIC_NUM + 1


    def button_start_clicked(self):
        if self.timer_camera.isActive() == False:
            camera = tools.judge_camera()
            if camera[0] >= '0' and camera[0] <= '9':
                self.CAM_NUM = int(camera[0])
            else:
                self.CAM_NUM = 10
            flag = self.cap.open(self.CAM_NUM)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)
                self.btnOpenCam.setText('停止拍摄')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.left_cam.clear()
            self.right_cam.clear()
            self.btnOpenCam.setText('开始拍摄')

    def button_close_clicked(self):
        if self.timer_camera.isActive() == True:
            self.timer_camera.stop()
            self.cap.release()
            self.left_cam.clear()
            self.right_cam.clear()
        self.close()


    def cam_shoot(self):
        try:
            ret, self.frame = self.cap.read()
            # show = cv2.resize(self.frame,(1280,480))
            show = self.frame
            left_show = show[:, 0:640, :]
            right_show = show[:, 640:1280, :]
            left_show = cv2.cvtColor(left_show, cv2.COLOR_BGR2RGB)
            right_show = cv2.cvtColor(right_show, cv2.COLOR_BGR2RGB)
            left_showImage = QtGui.QImage(left_show.data, left_show.shape[1], left_show.shape[0],
                                          QtGui.QImage.Format_RGB888)
            right_showImage = QtGui.QImage(right_show.data, right_show.shape[1], right_show.shape[0],
                                           QtGui.QImage.Format_RGB888)
            self.left_cam.setPixmap(QtGui.QPixmap.fromImage(left_showImage))
            self.right_cam.setPixmap(QtGui.QPixmap.fromImage(right_showImage))
        except:
            msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            self.timer_camera.stop()
            self.left_cam.clear()
            self.right_cam.clear()
