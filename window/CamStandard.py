# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CamStandard.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1329, 740)
        MainWindow.setMouseTracking(True)
        MainWindow.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(31, 21, 91, 16))
        self.label.setObjectName("label")
        self.left_cam = QtWidgets.QLabel(self.centralwidget)
        self.left_cam.setGeometry(QtCore.QRect(10, 230, 640, 480))
        self.left_cam.setStyleSheet("QLabel{\n"
"    background-color:white;\n"
"}")
        self.left_cam.setObjectName("left_cam")
        self.right_cam = QtWidgets.QLabel(self.centralwidget)
        self.right_cam.setGeometry(QtCore.QRect(670, 230, 640, 480))
        self.right_cam.setStyleSheet("QLabel{\n"
"    background-color:white;\n"
"}")
        self.right_cam.setObjectName("right_cam")
        self.ChessBoardWidth = QtWidgets.QLineEdit(self.centralwidget)
        self.ChessBoardWidth.setGeometry(QtCore.QRect(180, 20, 101, 21))
        self.ChessBoardWidth.setObjectName("ChessBoardWidth")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(289, 21, 16, 16))
        self.label_2.setObjectName("label_2")
        self.ChessBoardLength = QtWidgets.QLineEdit(self.centralwidget)
        self.ChessBoardLength.setGeometry(QtCore.QRect(310, 20, 101, 21))
        self.ChessBoardLength.setObjectName("ChessBoardLength")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(420, 20, 16, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(30, 60, 111, 16))
        self.label_4.setObjectName("label_4")
        self.RectWidth = QtWidgets.QLineEdit(self.centralwidget)
        self.RectWidth.setGeometry(QtCore.QRect(180, 60, 101, 21))
        self.RectWidth.setObjectName("RectWidth")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(290, 60, 72, 15))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(30, 110, 72, 15))
        self.label_6.setObjectName("label_6")
        self.Calib01 = QtWidgets.QRadioButton(self.centralwidget)
        self.Calib01.setGeometry(QtCore.QRect(180, 110, 115, 19))
        self.Calib01.setObjectName("Calib01")
        self.buttonGroup = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.Calib01)
        self.Calib02 = QtWidgets.QRadioButton(self.centralwidget)
        self.Calib02.setGeometry(QtCore.QRect(310, 110, 115, 19))
        self.Calib02.setObjectName("Calib02")
        self.buttonGroup.addButton(self.Calib02)
        self.btnSelectLeftImage = QtWidgets.QPushButton(self.centralwidget)
        self.btnSelectLeftImage.setGeometry(QtCore.QRect(950, 30, 171, 28))
        self.btnSelectLeftImage.setObjectName("btnSelectLeftImage")
        self.btnSelectRightImage = QtWidgets.QPushButton(self.centralwidget)
        self.btnSelectRightImage.setGeometry(QtCore.QRect(950, 80, 171, 28))
        self.btnSelectRightImage.setObjectName("btnSelectRightImage")
        self.btnStartCalib = QtWidgets.QPushButton(self.centralwidget)
        self.btnStartCalib.setGeometry(QtCore.QRect(30, 160, 93, 28))
        self.btnStartCalib.setObjectName("btnStartCalib")
        self.btnOpenCam = QtWidgets.QPushButton(self.centralwidget)
        self.btnOpenCam.setGeometry(QtCore.QRect(130, 160, 93, 28))
        self.btnOpenCam.setObjectName("btnOpenCam")
        self.btnCatchCam = QtWidgets.QPushButton(self.centralwidget)
        self.btnCatchCam.setGeometry(QtCore.QRect(230, 160, 93, 28))
        self.btnCatchCam.setObjectName("btnCatchCam")
        self.btnClose = QtWidgets.QPushButton(self.centralwidget)
        self.btnClose.setGeometry(QtCore.QRect(330, 160, 111, 28))
        self.btnClose.setObjectName("btnClose")
        self.left_path = QtWidgets.QLineEdit(self.centralwidget)
        self.left_path.setEnabled(True)
        self.left_path.setGeometry(QtCore.QRect(510, 30, 421, 28))
        self.left_path.setMouseTracking(False)
        self.left_path.setObjectName("left_path")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(453, 20, 20, 121))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.right_path = QtWidgets.QLineEdit(self.centralwidget)
        self.right_path.setGeometry(QtCore.QRect(510, 80, 421, 28))
        self.right_path.setObjectName("right_path")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "棋盘角点数："))
        self.left_cam.setText(_translate("MainWindow", "左相机图像"))
        self.right_cam.setText(_translate("MainWindow", "右相机图像"))
        self.label_2.setText(_translate("MainWindow", "宽"))
        self.label_3.setText(_translate("MainWindow", "长"))
        self.label_4.setText(_translate("MainWindow", "棋盘方块边长："))
        self.label_5.setText(_translate("MainWindow", "mm"))
        self.label_6.setText(_translate("MainWindow", "标定方式："))
        self.Calib01.setText(_translate("MainWindow", "OpenCV"))
        self.Calib02.setText(_translate("MainWindow", "MATLAB"))
        self.btnSelectLeftImage.setText(_translate("MainWindow", "选择左相机图片文件夹"))
        self.btnSelectRightImage.setText(_translate("MainWindow", "选择右相机图片文件夹"))
        self.btnStartCalib.setText(_translate("MainWindow", "开始标定"))
        self.btnOpenCam.setText(_translate("MainWindow", "打开相机"))
        self.btnCatchCam.setText(_translate("MainWindow", "拍摄相片"))
        self.btnClose.setText(_translate("MainWindow", "关闭标定窗口"))
