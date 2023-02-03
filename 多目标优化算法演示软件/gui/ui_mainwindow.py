# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1000, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.startBtn = QPushButton(self.centralwidget)
        self.startBtn.setObjectName(u"startBtn")
        self.startBtn.setGeometry(QRect(740, 120, 91, 51))
        font = QFont()
        font.setFamily(u"\u5fae\u8f6f\u96c5\u9ed1")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.startBtn.setFont(font)
        self.verticalLayoutWidget = QWidget(self.centralwidget)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(20, 90, 111, 401))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.quesBtn = QPushButton(self.verticalLayoutWidget)
        self.quesBtn.setObjectName(u"quesBtn")
        font1 = QFont()
        font1.setFamily(u"\u9ed1\u4f53")
        font1.setPointSize(12)
        self.quesBtn.setFont(font1)

        self.verticalLayout.addWidget(self.quesBtn)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.popuBtn = QPushButton(self.verticalLayoutWidget)
        self.popuBtn.setObjectName(u"popuBtn")
        self.popuBtn.setFont(font1)

        self.verticalLayout.addWidget(self.popuBtn)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_2)

        self.algoBtn = QPushButton(self.verticalLayoutWidget)
        self.algoBtn.setObjectName(u"algoBtn")
        self.algoBtn.setFont(font1)

        self.verticalLayout.addWidget(self.algoBtn)

        self.quesText = QTextBrowser(self.centralwidget)
        self.quesText.setObjectName(u"quesText")
        self.quesText.setGeometry(QRect(140, 20, 291, 181))
        font2 = QFont()
        font2.setFamily(u"\u5b8b\u4f53")
        font2.setPointSize(8)
        self.quesText.setFont(font2)
        self.quesText_2 = QTextBrowser(self.centralwidget)
        self.quesText_2.setObjectName(u"quesText_2")
        self.quesText_2.setGeometry(QRect(140, 220, 291, 81))
        self.quesText_2.setFont(font2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1000, 26))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u591a\u76ee\u6807\u7b97\u6cd5\u6f14\u793a\u8f6f\u4ef6", None))
        self.startBtn.setText(QCoreApplication.translate("MainWindow", u"\u5f00\u59cb", None))
        self.quesBtn.setText(QCoreApplication.translate("MainWindow", u"\u8bbe\u7f6e\u95ee\u9898", None))
        self.popuBtn.setText(QCoreApplication.translate("MainWindow", u"\u8bbe\u7f6e\u79cd\u7fa4", None))
        self.algoBtn.setText(QCoreApplication.translate("MainWindow", u"\u8bbe\u7f6e\u7b97\u6cd5", None))
        self.quesText.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'\u5b8b\u4f53'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\u95ee\u9898\u63cf\u8ff0\uff1a</p></body></html>", None))
        self.quesText_2.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'\u5b8b\u4f53'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\u79cd\u7fa4\u63cf\u8ff0\uff1a</p></body></html>", None))
    # retranslateUi

