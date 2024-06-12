from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import *

import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMovie

import EmotionRecongnition
from models.cnn import mini_XCEPTION, tiny_XCEPTION, big_XCEPTION, simple_CNN
from real_time_video_me import Emotion_Rec
from os import getcwd
import numpy as np
import cv2
import time
from base64 import b64decode
from os import remove
from slice_png import img as bgImg
import image1_rc


class Ui_MainWindow(object):
    # choose = "默认模型：_mini_XCEPTION.102-0.66.hdf5"

    def __init__(self, MainWindow):

        super().__init__()

        self.path = getcwd()
        self.timer_camera = QtCore.QTimer() # 定时器

        self.setupUi(MainWindow)
        self.retranslateUi(MainWindow)
        self.slot_init() # 槽函数设置

        # 设置界面动画
        gif = QMovie(':/newPrefix/images_test/scan.gif')
        self.label_face.setMovie(gif)
        gif.start()


        self.cap = cv2.VideoCapture() # 屏幕画面对象
        self.CAM_NUM = 0 # 摄像头标号
        self.model_path = None # 模型路径
        # self.__flag_work = 0



    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(1200, 950)#大小
        MainWindow.setMinimumSize(QtCore.QSize(1200, 950))#最小化
        MainWindow.setMaximumSize(QtCore.QSize(1200, 950))#最大化数值765
        MainWindow.setToolTip("")#工具提示
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/result.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)#图标
        MainWindow.setAutoFillBackground(True)
        '''1line:背景；；
        3line:'''
        MainWindow.setStyleSheet("#MainWindow{border-image: url(./images_test/background.PNG);}\n"   
                                 "\n"
                                 "#QInputDialog{border-image: url(./images_test/light.png);}\n"
                                 "\n"
                                 "QMenuBar{border-color:transparent;}\n"
                                 "QToolButton[objectName=pushButton_doIt]{\n"
                                 "border:5px;}\n"
                                 "\n"
                                 "QToolButton[objectName=pushButton_doIt]:hover {\n"
                                 "image:url(:/newPrefix/images_test/run_hover.png);}\n"
                                 "\n"
                                 "QToolButton[objectName=pushButton_doIt]:pressed {\n"
                                 "image:url(:/newPrefix/images_test/run_pressed.png);}\n"
                                 "\n"
                                 "QScrollBar:vertical{\n"
                                 "background:transparent;\n"
                                 "padding:2px;\n"
                                 "border-radius:8px;\n"
                                 "max-width:14px;}\n"
                                 "\n"
                                 "QScrollBar::handle:vertical{\n"
                                 "background:#9acd32;\n"
                                 "min-height:50px;\n"
                                 "border-radius:8px;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::handle:vertical:hover{\n"
                                 "background:#9eb764;}\n"
                                 "\n"
                                 "QScrollBar::handle:vertical:pressed{\n"
                                 "background:#9eb764;\n"
                                 "}\n"
                                 "QScrollBar::add-page:vertical{\n"
                                 "background:none;\n"
                                 "}\n"
                                 "                               \n"
                                 "QScrollBar::sub-page:vertical{\n"
                                 "background:none;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::add-line:vertical{\n"
                                 "background:none;}\n"
                                 "                                 \n"
                                 "QScrollBar::sub-line:vertical{\n"
                                 "background:none;\n"
                                 "}\n"
                                 "QScrollArea{\n"
                                 "border:0px;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar:horizontal{\n"
                                 "background:transparent;\n"
                                 "padding:0px;\n"
                                 "border-radius:6px;\n"
                                 "max-height:4px;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::handle:horizontal{\n"
                                 "background:#9acd32;\n"
                                 "min-width:50px;\n"
                                 "border-radius:6px;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::handle:horizontal:hover{\n"
                                 "background:#9eb764;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::handle:horizontal:pressed{\n"
                                 "background:#9eb764;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::add-page:horizontal{\n"
                                 "background:none;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::sub-page:horizontal{\n"
                                 "background:none;\n"
                                 "}\n"
                                 "QScrollBar::add-line:horizontal{\n"
                                 "background:none;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::sub-line:horizontal{\n"
                                 "background:none;\n"
                                 "}\n"
                                 "QToolButton::hover{\n"
                                 "border:0px;\n"
                                 "} ")
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        ##############################################################################################
        ##表情识别标题字样
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        self.label_title.setGeometry(QtCore.QRect(440, 50, 320, 50))#窗口位置和大小
        self.label_title.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("隶书")#字体
        font.setPointSize(22)#字号
        # font.setItalic(True)#斜体
        self.label_title.setFont(font)
        self.label_title.setStyleSheet("color: rgb(40,148,255);")#白字（（（表情识别系统
        self.label_title.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)#左部中间；；顶部
        self.label_title.setObjectName("label_title")

        ###################################################################################################
        #页面跳转
        self.toolButton_jump = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_jump.setGeometry(QtCore.QRect(20, 490, 40, 40))
        self.toolButton_jump.setMinimumSize(QtCore.QSize(40, 40))
        self.toolButton_jump.setMaximumSize(QtCore.QSize(40, 40))
        self.toolButton_jump.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_jump.setAutoFillBackground(False)
        self.toolButton_jump.setStyleSheet("background-color: transparent;")
        self.toolButton_jump.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./images_test/showmodel.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_jump.setIcon(icon)
        self.toolButton_jump.setIconSize(QtCore.QSize(40, 40))
        self.toolButton_jump.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton_jump.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_jump.setAutoRaise(False)
        self.toolButton_jump.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_jump.setObjectName("toolButton_jump")

        ###############################################################################################
        # 跳转_文本框
        self.textButton_jump = QtWidgets.QTextEdit(self.centralwidget)
        self.textButton_jump.setGeometry(QtCore.QRect(80, 490, 220, 38))
        self.textButton_jump.setMinimumSize(QtCore.QSize(220, 38))
        self.textButton_jump.setMaximumSize(QtCore.QSize(220, 38))
        self.textButton_jump.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textButton_jump.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        font = QtGui.QFont()
        font.setFamily("SimSun")
        font.setPointSize(12)
        self.textButton_jump.setFont(font)
        self.textButton_jump.setStyleSheet("background-color: transparent;\n"
                                          "border-color: rgb(255, 255, 255);\n"
                                          "color: rgb(40,148,255);")
        # 白框白字透明背景
        self.textButton_jump.setReadOnly(True)
        self.textButton_jump.setObjectName("textButton_jump")

        ##############################################################################################
        ##应该是作者
        self.label_author = QtWidgets.QLabel(self.centralwidget)
        self.label_author.setGeometry(QtCore.QRect(250, 110, 132, 30))#位置+大小
        self.label_author.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(16)
        self.label_author.setFont(font)
        self.label_author.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_author.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)#左中置顶
        self.label_author.setObjectName("label_author")
        ##########################################################################################
        #用时--文字
        self.label_useTime = QtWidgets.QLabel(self.centralwidget)
        self.label_useTime.setGeometry(QtCore.QRect(150, 720, 100, 40))#横向长度，纵向长度
        font = QtGui.QFont()
        font.setFamily("华文仿宋")
        font.setPointSize(16)
        self.label_useTime.setFont(font)
        self.label_useTime.setObjectName("label_useTime")
        ###########################################################################################
        #识别结果--文字
        self.label_scanResult = QtWidgets.QLabel(self.centralwidget)
        self.label_scanResult.setGeometry(QtCore.QRect(900, 720, 140, 40))
        font = QtGui.QFont()
        font.setFamily("华文仿宋")
        font.setPointSize(16)
        self.label_scanResult.setFont(font)
        self.label_scanResult.setObjectName("label_scanResult")
        ###############################################################################################
        #用时的图标
        self.label_picTime = QtWidgets.QLabel(self.centralwidget)
        self.label_picTime.setGeometry(QtCore.QRect(80, 720, 40, 40))
        self.label_picTime.setStyleSheet("border-image: url(:/newPrefix/images_test/net_speed.png);")
        self.label_picTime.setText("")#text的显示是在图标上方（不用）
        self.label_picTime.setObjectName("label_picTime")
        ###############################################################################################
        #结果的图标
        self.label_picResult = QtWidgets.QLabel(self.centralwidget)
        self.label_picResult.setGeometry(QtCore.QRect(830, 720, 40, 40))
        self.label_picResult.setStyleSheet("border-image: url(./images_test/result.png);")#路由直接改为绝对路径
        self.label_picResult.setText("")
        self.label_picResult.setObjectName("label_picResult")
        #################################################################################################
        #用时上方的横线
        # self.line = QtWidgets.QFrame(self.centralwidget)
        # self.line.setGeometry(QtCore.QRect(440, 160, 321, 21))
        # self.line.setFrameShape(QtWidgets.QFrame.HLine)
        # self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.line.setObjectName("line")
        ##############################################################################################
        #人脸----动图
        self.label_face = QtWidgets.QLabel(self.centralwidget)
        self.label_face.setGeometry(QtCore.QRect(320, 200, 560, 370))
        self.label_face.setMinimumSize(QtCore.QSize(560, 370))
        self.label_face.setMaximumSize(QtCore.QSize(560, 370))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(16)
        self.label_face.setFont(font)
        self.label_face.setLayoutDirection(QtCore.Qt.LeftToRight)#从左到右
        self.label_face.setStyleSheet("border-image: url(:/newPrefix/images_test/scan.gif);")
        self.label_face.setAlignment(QtCore.Qt.AlignCenter)#左中
        self.label_face.setObjectName("label_face")
        ###############################################################################################
        #模型选择_文本框
        self.textEdit_model = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_model.setGeometry(QtCore.QRect(80, 240, 220, 38))
        self.textEdit_model.setMinimumSize(QtCore.QSize(220, 38))
        self.textEdit_model.setMaximumSize(QtCore.QSize(220, 38))
        self.textEdit_model.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textEdit_model.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        font = QtGui.QFont()
        font.setFamily("SimSun")
        font.setPointSize(12)
        self.textEdit_model.setFont(font)
        self.textEdit_model.setStyleSheet("background-color: transparent;\n"
                                          "border-color: rgb(255, 255, 255);\n"
                                          "color: rgb(40,148,255);")
        #白框白字透明背景
        self.textEdit_model.setReadOnly(True)
        self.textEdit_model.setObjectName("textEdit_model")
        ###################################################################################################
        #选择图片_图标加按钮
        self.toolButton_file = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_file.setGeometry(QtCore.QRect(1140, 490, 40, 40))
        self.toolButton_file.setMinimumSize(QtCore.QSize(40, 40))
        self.toolButton_file.setMaximumSize(QtCore.QSize(40, 40))
        self.toolButton_file.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_file.setAutoFillBackground(False)
        self.toolButton_file.setStyleSheet("background-color: transparent;")
        self.toolButton_file.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./images_test/recovery1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_file.setIcon(icon)
        self.toolButton_file.setIconSize(QtCore.QSize(40, 40))
        self.toolButton_file.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton_file.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_file.setAutoRaise(False)
        self.toolButton_file.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_file.setObjectName("toolButton_camera_2")
        #############################################################################################################
        #摄像头_文本框
        self.textEdit_camera = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_camera.setGeometry(QtCore.QRect(900, 240, 220, 38))
        self.textEdit_camera.setMinimumSize(QtCore.QSize(220, 38))
        self.textEdit_camera.setMaximumSize(QtCore.QSize(220, 38))
        self.textEdit_camera.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textEdit_camera.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        font = QtGui.QFont()
        font.setFamily("SimSun")
        font.setPointSize(12)
        self.textEdit_camera.setFont(font)
        self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                           "border-color: rgb(255, 255, 255);\n"
                                           "color: rgb(40,148,255);")
        self.textEdit_camera.setReadOnly(True)
        self.textEdit_camera.setObjectName("textEdit_camera")
        ###################################################################################################
        #选择图片_文本框
        self.textEdit_pic = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_pic.setGeometry(QtCore.QRect(900, 490, 220, 38))
        self.textEdit_pic.setMinimumSize(QtCore.QSize(220, 38))
        self.textEdit_pic.setMaximumSize(QtCore.QSize(220, 38))
        self.textEdit_pic.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textEdit_pic.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        font = QtGui.QFont()
        font.setFamily("SimSun")
        font.setPointSize(12)
        self.textEdit_pic.setFont(font)
        self.textEdit_pic.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.textEdit_pic.setStyleSheet("background-color: transparent;\n"
                                        "border-color: rgb(255, 255, 255);\n"
                                        "color: rgb(40,148,255);")
        self.textEdit_pic.setReadOnly(True)
        self.textEdit_pic.setObjectName("textEdit_pic")
        ####################################################################################################
        #摄像头_图标加按钮
        self.toolButton_camera = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_camera.setGeometry(QtCore.QRect(1140, 240, 40, 40))
        self.toolButton_camera.setMinimumSize(QtCore.QSize(40, 40))
        self.toolButton_camera.setMaximumSize(QtCore.QSize(40, 40))
        self.toolButton_camera.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_camera.setAutoFillBackground(False)
        self.toolButton_camera.setStyleSheet("background-color: transparent;")
        self.toolButton_camera.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("./images_test/g2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_camera.setIcon(icon1)
        self.toolButton_camera.setIconSize(QtCore.QSize(50, 39))
        self.toolButton_camera.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton_camera.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_camera.setAutoRaise(False)
        self.toolButton_camera.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_camera.setObjectName("toolButton_camera")
        ##################################################################################################
        #模型选择图标加按钮
        self.toolButton_model = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_model.setGeometry(QtCore.QRect(20, 240, 40, 40))
        self.toolButton_model.setMinimumSize(QtCore.QSize(40, 40))
        self.toolButton_model.setMaximumSize(QtCore.QSize(40, 40))
        self.toolButton_model.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_model.setAutoFillBackground(False)
        self.toolButton_model.setStyleSheet("background-color: transparent;")
        self.toolButton_model.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("./images_test/fx.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_model.setIcon(icon2)
        self.toolButton_model.setIconSize(QtCore.QSize(40, 40))
        self.toolButton_model.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton_model.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_model.setAutoRaise(False)
        self.toolButton_model.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_model.setObjectName("toolButton_model")
        ###################################################################################################
        #用时--数字
        self.label_time = QtWidgets.QLabel(self.centralwidget)
        self.label_time.setGeometry(QtCore.QRect(240, 720, 100, 40))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_time.setStyleSheet("color: rgb(40,148,255);")
        self.label_time.setFont(font)
        self.label_time.setObjectName("label_time")
        ################################################################################################
        #结果--英文
        self.label_result = QtWidgets.QLabel(self.centralwidget)
        self.label_result.setGeometry(QtCore.QRect(1040, 720, 200, 40))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_result.setFont(font)
        self.label_result.setStyleSheet("color: rgb(40,148,255);")#100 189 189
        self.label_result.setObjectName("label_result")
        ################################################################################################
        #结果输出（条目
        self.label_outputResult = QtWidgets.QLabel(self.centralwidget)
        self.label_outputResult.setGeometry(QtCore.QRect(405, 620, 390, 325))#4——250
        self.label_outputResult.setText("")
        self.label_outputResult.setStyleSheet("border-image: url(./images_test/ini.png);")
        self.label_outputResult.setObjectName("label_outputResult")

        # # 竖线1
        # self.line_3 = QtWidgets.QFrame(self.centralwidget)
        # self.line_3.setGeometry(QtCore.QRect(449,379, 1, 252))
        # self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        # self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.line_3.setObjectName("line_2")


        ###################################################################################################
        MainWindow.setCentralWidget(self.centralwidget)
        self.actionGoogle_Translate = QtWidgets.QAction(MainWindow)
        self.actionGoogle_Translate.setObjectName("actionGoogle_Translate")
        self.actionHTML_type = QtWidgets.QAction(MainWindow)
        self.actionHTML_type.setObjectName("actionHTML_type")
        self.actionsoftware_version = QtWidgets.QAction(MainWindow)
        self.actionsoftware_version.setObjectName("actionsoftware_version")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)#信息传输

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Emotion Recognition & Analysis System"))#情绪识别与分析
        self.label_title.setText(_translate("MainWindow", "<html><head/><body><p><b>ERAS</p></body></html>"))#情绪识别系统
        self.label_useTime.setText(_translate("MainWindow", "<html><head/><body><p><b>Time：</p></body></html>"))#用时
        self.label_scanResult.setText(_translate("MainWindow", "<html><head/><body><p><b>Results：<br/></p></body></html>"))#识别结果
        self.label_face.setText(
            _translate("MainWindow", "<html><head/><body><p align=\"center\"><br/></p></body></html>"))
        self.textButton_jump.setHtml(_translate("MainWindow",
                                               "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                               "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                               "p, li { white-space: pre-wrap; }\n"
                                               "</style></head><body style=\" font-family:\'华文仿宋\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                                               "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Adobe Devanagari\';\"><b>View Network</span></p></body></html>"))#查看网络结构

        self.textEdit_model.setHtml(_translate("MainWindow",
                                               "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                               "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                               "p, li { white-space: pre-wrap; }\n"
                                               "</style></head><body style=\" font-family:\'华文仿宋\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                                               "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Adobe Devanagari\';\"><b>Select Model</span></p></body></html>"))#选择模型
        self.textEdit_camera.setHtml(_translate("MainWindow",
                                                "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                                "p, li { white-space: pre-wrap; }\n"
                                                "</style></head><body style=\" font-family:\'华文仿宋\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                                                "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Adobe Devanagari\';\"><b>Real-time</span></p></body></html>"))#实时摄像未开启
        self.textEdit_pic.setHtml(_translate("MainWindow",
                                             "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                             "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                             "p, li { white-space: pre-wrap; }\n"
                                             "</style></head><body style=\" font-family:\'华文仿宋\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                                             "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Adobe Devanagari\';\"><b>Select Image</span></p></body></html>"))#选择图片
        self.label_time.setText(_translate("MainWindow", "0 s"))
        self.label_result.setText(_translate("MainWindow", "None"))
        self.actionGoogle_Translate.setText(_translate("MainWindow", "Google Translate"))
        self.actionHTML_type.setText(_translate("MainWindow", "HTML type"))
        self.actionsoftware_version.setText(_translate("MainWindow", "software version"))


    def slot_init(self): # 定义槽函数,,信息传输---模型接口
        self.toolButton_camera.clicked.connect(self.button_open_camera_click)
        self.toolButton_model.clicked.connect(self.choose_model)
        self.timer_camera.timeout.connect(self.show_camera)
        self.toolButton_file.clicked.connect(self.choose_pic)
        self.toolButton_jump.clicked.connect(self.jump)

    ##########################################################
    #界面跳转函数
    def jump(self):
        child = DialogOfYouNeed()
        child.exec_()



    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False: # 检查定时状态
            flag = self.cap.open(self.CAM_NUM) # 检查相机状态
            if flag == False: # 相机打开失败提示
                msg = QtWidgets.QMessageBox.warning(self.centralwidget, u"Warning",
                                                    u"Please check if the camera and computer are connected correctly! ",#请检测相机与电脑是否连接正确！
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)

            else:
                # 准备运行识别程序
                self.textEdit_pic.setText('File not selected')#文件未选中
                QtWidgets.QApplication.processEvents()
                self.textEdit_camera.setText('Real-Time ON.')#实时摄像已开启
                self.label_face.setText('Recognition system being activated...\n\nleading')#正在启动识别系统...
                # 新建对象
                self.emotion_model = Emotion_Rec(self.model_path)
                QtWidgets.QApplication.processEvents()
                # 打开定时器
                self.timer_camera.start(30)
        else:
            # 定时器未开启，界面回复初始状态
            self.timer_camera.stop()
            self.cap.release()
            self.label_face.clear()
            self.textEdit_camera.setText('Real-Time OFF.')#实时摄像已关闭
            self.textEdit_pic.setText('File not selected')#文件未选中
            gif = QMovie(':/newPrefix/images_test/scan.gif')
            self.label_face.setMovie(gif)
            gif.start()
            self.label_outputResult.clear()
            self.label_outputResult.setStyleSheet("border-image: url(./images_test/ini.png);")

            self.label_result.setText('None')
            self.label_time.setText('0 s')


    def show_camera(self):
        # 定时器槽函数，每隔一段时间执行
        flag, self.image = self.cap.read() # 获取画面
        self.image=cv2.flip(self.image, 1) # 左右翻转

        tmp = open('slice.png', 'wb')
        tmp.write(b64decode(bgImg))
        tmp.close()
        canvas = cv2.imread('slice.png') # 用于数据显示的背景图片
        remove('slice.png')

        time_start = time.time() # 计时
        # 使用模型预测
        result = self.emotion_model.run(self.image, canvas, self.label_face, self.label_outputResult)
        time_end = time.time()
        # 在界面显示结果
        self.label_result.setText(result)
        self.label_time.setText(str(round((time_end-time_start),3))+' s')



    def choose_pic(self):
        # 界面处理
        self.timer_camera.stop()
        self.cap.release()
        self.label_face.clear()
        self.label_result.setText('None')
        self.label_time.setText('0 s')
        self.textEdit_camera.setText('Real-Time OFF.')#实时摄像已关闭
        self.label_outputResult.clear()
        self.label_outputResult.setStyleSheet("border-image: url(./images_test/ini.png);")

        # 使用文件选择对话框选择图片
        fileName_choose, filetype = QFileDialog.getOpenFileName(
                                self.centralwidget, "选取图片文件",
                                self.path,  # 起始路径
                                "图片(*.jpg;*.jpeg;*.png)") # 文件类型
        self.path = fileName_choose # 保存路径
        if fileName_choose != '':
            self.textEdit_pic.setText(fileName_choose+'File selected')#文件已选中
            self.label_face.setText('Recognition system being activated...\n\nleading')#在启动识别系统...
            QtWidgets.QApplication.processEvents()
            # 生成模型对象
            self.emotion_model = Emotion_Rec(self.model_path)
            # 读取背景图
            tmp = open('slice.png', 'wb')
            tmp.write(b64decode(bgImg))
            tmp.close()
            canvas = cv2.imread('slice.png')
            remove('slice.png')

            image = self.cv_imread(fileName_choose) # 读取选择的图片
            # 计时并开始模型预测
            QtWidgets.QApplication.processEvents()
            time_start = time.time()
            result = self.emotion_model.run(image, canvas, self.label_face, self.label_outputResult)
            print(result)
            if result==None:
                result="None"
            time_end = time.time()
            # 显示结果
            self.label_result.setText(result)
            self.label_time.setText(str(round((time_end - time_start), 3)) + ' s')

        else:
            # 选择取消，恢复界面状态
            self.textEdit_pic.setText('File not selected')#文件未选中
            gif = QMovie(':/newPrefix/images_test/scan.gif')
            self.label_face.setMovie(gif)
            gif.start()
            self.label_outputResult.clear() # 清除画面
            self.label_outputResult.setStyleSheet("border-image: url(./images_test/ini.png);")
            self.label_result.setText('None')
            self.label_time.setText('0 s')


    def cv_imread(self,filePath):
        # 读取图片
        cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
        ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        ## cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        return cv_img


    def choose_model(self):
        # 选择训练好的模型文件
        global choose
        choose="默认模型：_mini_XCEPTION.102-0.66.hdf5" #choose="默认模型：_mini_XCEPTION.102-0.66.hdf5"
        self.timer_camera.stop()
        self.cap.release()
        self.label_face.clear()
        self.label_result.setText('None')
        self.label_time.setText('0 s')
        self.textEdit_camera.setText('Real-Time OFF.')#实时摄像已关闭
        self.label_outputResult.clear()
        self.label_outputResult.setStyleSheet("border-image: url(./images_test/ini.png);")

        # 调用文件选择对话框
        fileName_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget,
                                                                "选取图片文件", getcwd(), # 起始路径
                                                                "Model File (*.hdf5)")  # 文件类型
        # 显示提示信息
        if fileName_choose != '':
            self.model_path = fileName_choose
            self.textEdit_model.setText(fileName_choose)
        else:
            self.textEdit_model.setText('Default Model')#使用默认模型

        gif = QMovie(':/newPrefix/images_test/scan.gif')
        self.label_face.setMovie(gif)
        gif.start()
        choose=self.getchoose()



    def getchoose(self):
        get_choose=self.textEdit_model.toPlainText()
        print(get_choose)
        print(type(get_choose))
        return get_choose



# class ChildDialogUi(QDialog):
#     def __init__(self):
#         super().__init__()
#
#         self.setWindowTitle("网络模型显示")
#         self.setGeometry(100, 100, 800, 600)
#         self.setWindowFlags(Qt.WindowCloseButtonHint)
#         layout = QHBoxLayout()
#         choosemodel="默认模型：_mini_XCEPTION.102-0.66.hdf5"
#         if Ui_MainWindow.choose!="":choosemodel=Ui_MainWindow.choose
#         print(choosemodel)
#         self.label = QLabel(self)
#         # self.label.setText(choosemodel)
#         # self.label.setGeometry(QtCore.QRect(10, 10,10,10))
#         layout.addWidget(self.label)
#         self.setLayout(layout)


        #
        # if get_choose == '':
        #     print("默认模型：_mini_XCEPTION.102-0.66.hdf5")
        #     choose = "默认模型：_mini_XCEPTION.102-0.66.hdf5"
        #     # model = mini_XCEPTION((64, 64, 1), 7)
        #     # with open('modelsummary.txt', 'w') as f:
        #     #     model.summary(print_fn=lambda x: f.write(x + '\n'))
        #
        # elif "_tiny_XCEPTION" in get_choose:
        #     print(get_choose.find("_tiny_XCEPTION"))
        #     print("_tiny_XCEPTION_model")
        #     choose = "_tiny_XCEPTION_model"
        #     # model = tiny_XCEPTION((64, 64, 1), 7)
        #     # with open('modelsummary.txt', 'w') as f:
        #     #     model.summary(print_fn=lambda x: f.write(x + '\n'))
        #
        # elif "_mini_XCEPTION" in get_choose:
        #     print(get_choose.find("_mini_XCEPTION"))
        #     print("_mini_XCEPTION_model")
        #     choose = "_mini_XCEPTION_model"
        #     # model = tiny_XCEPTION((64, 64, 1), 7)
        #     # with open('modelsummary.txt', 'w') as f:
        #     #     model.summary(print_fn=lambda x: f.write(x + '\n'))
        #
        # elif "_big_XCEPTION" in get_choose:
        #     print("_big_XCEPTION_model")
        #     choose = "_big_XCEPTION_model"
        #     # print(choose)
        #     # model = big_XCEPTION((64, 64, 1), 7)
        #     # with open('modelsummary.txt', 'w') as f:
        #     #     model.summary(print_fn=lambda x: f.write(x + '\n'))
        #
        # elif "_simple_CNN" in get_choose:
        #     print("_simple_CNN_model")
        #     get_choose = "_simple_CNN_model"
        #     # model = simple_CNN((48, 48, 1), 7)
        #     # with open('modelsummary.txt', 'w') as f:
        #     #     model.summary(print_fn=lambda x: f.write(x + '\n'))
        #
        # else:
        #     print("wrong")
        # print("######")
        # print(get_choose)
        # print("######")
class myWindow(QDialog):  # 不可用QMainWindow,因为QLabel继承自QWidget
    def __init__(self):
        super(myWindow, self).__init__()
        self.resize(600, 12000)  # 设定窗口大小(根据自己显示图片的大小，可更改)
        # if choose!="":
        #     getchoose=choose
        # else :
        #     getchoose="默认模型：_mini_XCEPTION.102-0.66.hdf5"
        try:
            getchoose=choose
        except:
            getchoose = "默认模型：_mini_XCEPTION.102-0.66.hdf5"
        print("函数")
        print(getchoose)
        print("##########")
        choosemodel="mini_XCEPTION.102-0.66"#mini_XCEPTION.102-0.66

        if getchoose=="默认模型：_mini_XCEPTION.102-0.66.hdf5":
            print(getchoose=="默认模型：_mini_XCEPTION.102-0.66.hdf5")
            print(choosemodel)
            print("默认")

        elif "_mini_XCEPTION" in getchoose:
            print(getchoose.find("_mini_XCEPTION"))
            print("_mini_XCEPTION_model")
            choosemodel=choosemodel

        elif "_tiny_XCEPTION" in getchoose:
            print(getchoose.find("_tiny_XCEPTION"))
            print("_tiny_XCEPTION_model")
            choosemodel = "tiny_XCEPTION.147-accuracy0.62"


        elif "_big_XCEPTION" in getchoose:
            print("_big_XCEPTION_model")
            choosemodel = "big_XCEPTION.41-accuracy0.71"

        elif "_simple_CNN" in getchoose:
            print("_simple_CNN_model")
            choosemodel = "simple_CNN.183-accuracy0.57"

        elif "_simpler_CNN" in getchoose:
            print("_simple_CNN_model")
            choosemodel = "simpler_CNN.191-accuracy0.59"

        else :
            choosemodel="wrong"
            print("wrong")
        self.setWindowTitle(choosemodel)  # 设定窗口名称
        self.imgPixmap = QPixmap('.\pic_model\_'+choosemodel+'.hdf5.png')  # 载入图片mini_XCEPTION.102-0.66.hdf5
        self.scaledImg = self.imgPixmap.scaled(self.size())  # 初始化缩放图
        self.singleOffset = QPoint(0, 0)  # 初始化偏移值
        self.isLeftPressed = bool(False)  # 图片被点住(鼠标左键)标志位
        self.isImgLabelArea = bool(True)  # 鼠标进入label图片显示区域

    '''重载绘图: 动态绘图'''

    def paintEvent(self, event):
        self.imgPainter = QPainter()  # 用于动态绘制图片
        self.imgFramePainter = QPainter()  # 用于动态绘制图片外线框
        self.imgPainter.begin(self)  # 无begin和end,则将一直循环更新
        self.imgPainter.drawPixmap(self.singleOffset, self.scaledImg)  # 从图像文件提取Pixmap并显示在指定位置
        self.imgFramePainter.setPen(QColor(168, 34, 3))  # 不设置则为默认黑色   # 设置绘图颜色/大小/样式
        self.imgFramePainter.drawRect(10, 10, 480, 480)  # 为图片绘外线狂(向外延展1)
        self.imgPainter.end()  # 无begin和end,则将一直循环更新

    # =============================================================================
    # 图片移动: 首先,确定图片被点选(鼠标左键按下)且未左键释放;
    #          其次,确定鼠标移动;
    #          最后,更新偏移值,移动图片.
    # =============================================================================
    '''重载一下鼠标按下事件(单击)'''

    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:  # 左键按下
            print("鼠标左键单击")  # 响应测试语句
            self.isLeftPressed = True;  # 左键按下(图片被点住),置Ture
            self.preMousePosition = event.pos()  # 获取鼠标当前位置
        elif event.buttons() == QtCore.Qt.RightButton:  # 右键按下
            print("鼠标右键单击")  # 响应测试语句
        elif event.buttons() == QtCore.Qt.MidButton:  # 中键按下
            print("鼠标中键单击")  # 响应测试语句
        elif event.buttons() == QtCore.Qt.LeftButton | QtCore.Qt.RightButton:  # 左右键同时按下
            print("鼠标左右键同时单击")  # 响应测试语句
        elif event.buttons() == QtCore.Qt.LeftButton | QtCore.Qt.MidButton:  # 左中键同时按下
            print("鼠标左中键同时单击")  # 响应测试语句
        elif event.buttons() == QtCore.Qt.MidButton | QtCore.Qt.RightButton:  # 右中键同时按下
            print("鼠标右中键同时单击")  # 响应测试语句
        elif event.buttons() == QtCore.Qt.LeftButton | QtCore.Qt.MidButton \
                | QtCore.Qt.RightButton:  # 左中右键同时按下
            print("鼠标左中右键同时单击")  # 响应测试语句

    '''重载一下滚轮滚动事件'''

    def wheelEvent(self, event):
        #        if event.delta() > 0:                                                 # 滚轮上滚,PyQt4
        # This function has been deprecated, use pixelDelta() or angleDelta() instead.
        angle = event.angleDelta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        angleX = angle.x()  # 水平滚过的距离(此处用不上)
        angleY = angle.y()  # 竖直滚过的距离
        if angleY > 0:  # 滚轮上滚
            print("鼠标中键上滚")  # 响应测试语句
            self.scaledImg = self.imgPixmap.scaled(self.scaledImg.width() + 5,
                                                   self.scaledImg.height() + 5)
            newWidth = event.x() - (self.scaledImg.width() * (event.x() - self.singleOffset.x())) \
                       / (self.scaledImg.width() - 5)
            newHeight = event.y() - (self.scaledImg.height() * (event.y() - self.singleOffset.y())) \
                        / (self.scaledImg.height() - 5)
            self.singleOffset = QPoint(newWidth, newHeight)  # 更新偏移量
            self.repaint()  # 重绘
        else:  # 滚轮下滚
            print("鼠标中键下滚")  # 响应测试语句
            self.scaledImg = self.imgPixmap.scaled(self.scaledImg.width() - 5,
                                                   self.scaledImg.height() - 5)
            newWidth = event.x() - (self.scaledImg.width() * (event.x() - self.singleOffset.x())) \
                       / (self.scaledImg.width() + 5)
            newHeight = event.y() - (self.scaledImg.height() * (event.y() - self.singleOffset.y())) \
                        / (self.scaledImg.height() + 5)
            self.singleOffset = QPoint(newWidth, newHeight)  # 更新偏移量
            self.repaint()  # 重绘

    '''重载一下鼠标键公开事件'''

    def mouseReleaseEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:  # 左键释放
            self.isLeftPressed = False;  # 左键释放(图片被点住),置False
            print("鼠标左键松开")  # 响应测试语句
        elif event.button() == Qt.RightButton:  # 右键释放
            self.singleOffset = QPoint(0, 0)  # 置为初值
            self.scaledImg = self.imgPixmap.scaled(self.size())  # 置为初值
            self.repaint()  # 重绘
            print("鼠标右键松开")  # 响应测试语句

    '''重载一下鼠标移动事件'''

    def mouseMoveEvent(self, event):
        if self.isLeftPressed:  # 左键按下
            print("鼠标左键按下，移动鼠标")  # 响应测试语句
            self.endMousePosition = event.pos() - self.preMousePosition  # 鼠标当前位置-先前位置=单次偏移量
            self.singleOffset = self.singleOffset + self.endMousePosition  # 更新偏移量
            self.preMousePosition = event.pos()  # 更新当前鼠标在窗口上的位置，下次移动用
            self.repaint()  # 重绘


#    '''重载一下鼠标双击事件'''
#    def mouseDoubieCiickEvent(self, event):
#        if event.buttons() == QtCore.Qt.LeftButton:                           # 左键按下
#            self.setText ("双击鼠标左键的功能: 自己定义")
#
#
#    '''重载一下鼠标进入控件事件'''
#    def enterEvent(self, event):
#
#
#    '''重载一下鼠标离开控件事件'''
#    def leaveEvent(self, event):
#

#    '''重载一下鼠标进入控件事件'''
#    def enterEvent(self, event):
#
#
#    '''重载一下鼠标离开控件事件'''
#    def leaveEvent(self, event):
#


class DialogOfYouNeed(myWindow):
    def __init__(self):
        super().__init__()

