# 主界面

import sys,cv2
from threading import Thread
import window.Main as Main
from subWin import SubMain
import window.configs as configs
from PyQt5 import QtWidgets,QtGui,QtCore
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import pyqtSignal
import numpy as np
from efficientdet import Efficientdet
from PIL import Image
import tools

class MainProcess(QMainWindow,Main.Ui_MainWindow):
    change_signal = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.timer_camera = QtCore.QTimer()
        self.timer_detect = QtCore.QTimer()
        self.timer_depths = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 10
        self.efficientdet = Efficientdet()
        self.set_ui()
        # 一定要在主窗口类的初始化函数中对子窗口进行实例化，如果在其他函数中实例化子窗口
        # 可能会出现子窗口闪退的问题
        self.slot_init()
        self.ChildDialog = Configs()
        self.ChildDialog.param_signal.connect(self.show_depths)

    def set_ui(self):
        QMainWindow.__init__(self)
        Main.Ui_MainWindow.__init__(self)
        self.setupUi(self)

    # 绑定信号
    def slot_init(self):
        self.btnScanCam.clicked.connect(self.button_scan_camera)

        self.btnOpen.clicked.connect(self.button_open_camera_clicked)
        self.timer_camera.timeout.connect(self.show_camera)

        self.btnDetect.clicked.connect(self.button_detect_clicked)
        self.timer_detect.timeout.connect(self.show_detect)

        self.btnStandard.clicked.connect(self.change_signal)
        self.change_signal.connect(self.page_change)

        self.btnMeasure.clicked.connect(self.button_depths_clicked)
        self.timer_depths.timeout.connect(self.show_depths)
    # 切换页面信号绑定
    def slot_btn_change(self):
        self.change_signal.emit()

    def page_change(self):
        self.sub = SubMain()
        if self.timer_camera.isActive() == True:
            self.timer_camera.stop()
        elif self.timer_detect.isActive() == True:
            self.timer_detect.stop()
        if self.cap.isOpened():
            self.cap.release()
        self.left_image.clear()
        self.right_image.clear()
        # 父窗口生成子窗口后置顶子窗口，限制操作父窗口
        self.sub.setWindowModality(QtCore.Qt.ApplicationModal)
        self.sub.show()

    def button_scan_camera(self):
        camera = tools.judge_camera()
        if camera[0] >= '0' and camera[0] <= '9':
            self.CAM_NUM = int(camera[0])
            msg = QtWidgets.QMessageBox.warning(self, 'ok',"找到双目相机"+camera[1]+"请点击'打开相机'打开双目相机",buttons=QtWidgets.QMessageBox.Ok)
        else:
            self.CAM_NUM = 10
            msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)


    def button_depths_clicked(self):
        if self.timer_depths.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.ChildDialog.show()
                self.timer_depths.start(30)
                self.btnMeasure.setText('停止测距')
        else:
            self.timer_depths.stop()
            self.cap.release()
            self.ChildDialog.close()
            self.right_image.clear()
            self.left_image.clear()
            self.btnMeasure.setText('双目测距')

    def show_depths(self,params=[8,11,60,32,1,0,1]):
        try:
            flag, self.frame = self.cap.read()
            frame = self.frame
            l_frame = frame[:, 0:640, :]
            r_frame = frame[:, 640:1280, :]
            dist,threeD = tools.get_depthIMG(l_frame,r_frame,params)
            l_frame = cv2.cvtColor(l_frame, cv2.COLOR_BGR2RGB)
            l_frame = Image.fromarray(np.uint8(l_frame))
            dist = cv2.cvtColor(dist,cv2.COLOR_GRAY2RGB)
            l_frame = np.array(self.efficientdet.detect_image(l_frame,False,threeD))
            image = dist
            right_showImage = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3,
                                            QtGui.QImage.Format_RGB888)

            self.right_image.setPixmap(QtGui.QPixmap(right_showImage))
            l_frame = cv2.cvtColor(l_frame, cv2.COLOR_BGR2RGB)
            left_showImage = QtGui.QImage(l_frame.data, l_frame.shape[1], l_frame.shape[0],
                                          QtGui.QImage.Format_RGB888)
            self.left_image.setPixmap(QtGui.QPixmap.fromImage(left_showImage))
        except:
            msg = QtWidgets.QMessageBox.warning(self, 'warning', "有错误发生！", buttons=QtWidgets.QMessageBox.Ok)
            self.timer_depths.stop()
            self.btnMeasure.setText('双目测距')
            self.CAM_NUM = 10
            self.left_image.clear()
            self.right_image.clear()

    def button_detect_clicked(self):
        if self.timer_detect.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                if self.timer_camera.isActive() == True:
                    self.timer_camera.stop()
                    self.btnOpen.setText('打开相机')
                self.timer_detect.start(30)
                self.btnDetect.setText('停止检测')
                self.timer_camera.stop()
        else:
            self.timer_detect.stop()
            self.cap.release()
            self.left_image.clear()
            self.right_image.clear()
            self.btnDetect.setText('目标检测')

    def show_detect(self):
        try:
            # 读取某一帧
            flag, self.frame = self.cap.read()
            frame = self.frame
            l_frame = frame[:, 0:640, :]
            r_frame = frame[:, 640:1280, :]
            # 格式转变，BGRtoRGB
            l_frame = cv2.cvtColor(l_frame, cv2.COLOR_BGR2RGB)
            r_frame = cv2.cvtColor(r_frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            l_frame = Image.fromarray(np.uint8(l_frame))
            r_frame = Image.fromarray(np.uint8(r_frame))
            # 进行检测
            l_frame = np.array(self.efficientdet.detect_image(l_frame))
            r_frame = np.array(self.efficientdet.detect_image(r_frame))
            left_showImage = QtGui.QImage(l_frame.data, l_frame.shape[1], l_frame.shape[0],
                                          QtGui.QImage.Format_RGB888)
            right_showImage = QtGui.QImage(r_frame.data, r_frame.shape[1], r_frame.shape[0],
                                           QtGui.QImage.Format_RGB888)
            self.left_image.setPixmap(QtGui.QPixmap.fromImage(left_showImage))
            self.right_image.setPixmap(QtGui.QPixmap.fromImage(right_showImage))
        except:
            msg = QtWidgets.QMessageBox.warning(self, 'warning', "有错误发生！",buttons=QtWidgets.QMessageBox.Ok)
            self.timer_detect.stop()
            self.btnDetect.setText('目标检测')
            self.CAM_NUM = 10
            self.left_image.clear()
            self.right_image.clear()

    def button_open_camera_clicked(self):
        # self.button_scan_camera()
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # print(self.timer_detect.isActive())
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                if self.timer_detect.isActive() == True:
                    self.timer_detect.stop()
                    self.btnDetect.setText('目标检测')
                self.timer_camera.start(30)
                self.btnOpen.setText('关闭相机')
                self.timer_detect.stop()
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.left_image.clear()
            self.right_image.clear()
            self.btnOpen.setText('打开相机')

    def show_camera(self):
        try:
            flag, self.frame = self.cap.read()
            show = self.frame
            left_show = show[:, 0:640, :]
            right_show = show[:, 640:1280, :]
            left_show = cv2.cvtColor(left_show, cv2.COLOR_BGR2RGB)
            right_show = cv2.cvtColor(right_show, cv2.COLOR_BGR2RGB)

            left_showImage = QtGui.QImage(left_show.data, left_show.shape[1], left_show.shape[0], QtGui.QImage.Format_RGB888)
            right_showImage = QtGui.QImage(right_show.data, right_show.shape[1], right_show.shape[0], QtGui.QImage.Format_RGB888)
            self.left_image.setPixmap(QtGui.QPixmap.fromImage(left_showImage))
            self.right_image.setPixmap(QtGui.QPixmap.fromImage(right_showImage))
        except:
            msg = QtWidgets.QMessageBox.warning(self, 'warning', "有错误发生！",buttons=QtWidgets.QMessageBox.Ok)
            self.btnOpen.setText('打开双目相机')
            self.CAM_NUM = 10
            self.timer_camera.stop()
            self.left_image.clear()
            self.right_image.clear()

class Configs(QMainWindow,configs.Ui_MainWindow):
    param_signal = QtCore.pyqtSignal(list)
    def __init__(self):
        super().__init__()
        self.params = [1,5,50,1,1,0,1]
        self.timer_params = QtCore.QTimer()
        self.setUI()
        self.config_slot()

    def setUI(self):
        self.setupUi(self)
        self.numSlider.setMinimum(1)
        self.numSlider.setMaximum(13)
        self.numSlider.setSingleStep(1)

        self.bsSlider.setMinimum(5)
        self.bsSlider.setMaximum(31)
        self.bsSlider.setSingleStep(2)

        self.swsSlider.setMinimum(50)
        self.swsSlider.setMaximum(200)
        self.swsSlider.setSingleStep(1)

        self.srSlider.setMinimum(1)
        self.srSlider.setMaximum(255)
        self.srSlider.setSingleStep(1)

        self.urSlider.setMinimum(1)
        self.urSlider.setMaximum(255)
        self.urSlider.setSingleStep(1)

        self.mdSlider.setMinimum(0)
        self.mdSlider.setMaximum(255)
        self.mdSlider.setSingleStep(1)

        self.pfcSlider.setMaximum(65)
        self.pfcSlider.setMinimum(1)
        self.pfcSlider.setSingleStep(2)
        self.retranslateUi(self)

    def config_slot(self):
        self.numSlider.valueChanged.connect(self.num)
        self.bsSlider.valueChanged.connect(self.block)
        self.swsSlider.valueChanged.connect(self.speckleWindowSize)
        self.srSlider.valueChanged.connect(self.speckleRange)
        self.urSlider.valueChanged.connect(self.uniquenessRatio)
        self.mdSlider.valueChanged.connect(self.minDisparity)
        self.pfcSlider.valueChanged.connect(self.preFilterCap)

        self.transBtn.clicked.connect(self.button_trans_clicked)
        self.timer_params.timeout.connect(self.send_params)


    def num(self):
        self.numVal.setNum(self.numSlider.value())
        self.params[0] = int(self.numSlider.value())
    def block(self):
        self.bsVal.setNum(self.bsSlider.value())
        self.params[1] = self.bsSlider.value()
    def speckleWindowSize(self):
        self.swsVal.setNum(self.swsSlider.value())
        self.params[2] = self.swsSlider.value()
    def speckleRange(self):
        self.srVal.setNum(self.srSlider.value())
        self.params[3] = self.srSlider.value()
    def uniquenessRatio(self):
        self.urVal.setNum(self.urSlider.value())
        self.params[4] = self.urSlider.value()
    def minDisparity(self):
        self.mdVal.setNum(self.mdSlider.value())
        self.params[5] = self.mdSlider.value()
    def preFilterCap(self):
        self.pfcVal.setNum(self.pfcSlider.value())
        self.params[6] = self.pfcSlider.value()

    def button_trans_clicked(self):
        if self.timer_params.isActive() == False:
            self.timer_params.start(30)
            self.transBtn.setText("停止传送")
        else:
            self.timer_params.stop()
            self.transBtn.setText("开始传送")

    def send_params(self):
        params = self.params
        def threadFunc():
            self.param_signal.emit(params)
        thread = Thread(target=threadFunc)
        thread.start()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = MainProcess()
    ui.show()
    sys.exit(app.exec_())