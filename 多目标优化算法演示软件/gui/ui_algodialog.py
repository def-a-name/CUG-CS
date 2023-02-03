# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'algodialog.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_AlgoDialog(object):
    def setupUi(self, AlgoDialog):
        if not AlgoDialog.objectName():
            AlgoDialog.setObjectName(u"AlgoDialog")
        AlgoDialog.resize(399, 300)
        self.buttonBox = QDialogButtonBox(AlgoDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setGeometry(QRect(170, 240, 201, 32))
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        self.select2 = QComboBox(AlgoDialog)
        self.select2.setObjectName(u"select2")
        self.select2.setGeometry(QRect(140, 50, 241, 31))
        self.select1 = QComboBox(AlgoDialog)
        self.select1.setObjectName(u"select1")
        self.select1.setGeometry(QRect(30, 50, 91, 31))
        self.label_2 = QLabel(AlgoDialog)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(220, 20, 71, 21))
        font = QFont()
        font.setFamily(u"\u9ed1\u4f53")
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_1 = QLabel(AlgoDialog)
        self.label_1.setObjectName(u"label_1")
        self.label_1.setGeometry(QRect(40, 20, 81, 21))
        self.label_1.setFont(font)
        self.algoTxt = QLabel(AlgoDialog)
        self.algoTxt.setObjectName(u"algoTxt")
        self.algoTxt.setGeometry(QRect(210, 90, 141, 16))
        self.gridLayoutWidget = QWidget(AlgoDialog)
        self.gridLayoutWidget.setObjectName(u"gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(10, 130, 387, 81))
        self.gridLayout = QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.label_3 = QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName(u"label_3")
        font1 = QFont()
        font1.setFamily(u"\u5b8b\u4f53")
        font1.setPointSize(8)
        self.label_3.setFont(font1)

        self.gridLayout.addWidget(self.label_3, 0, 2, 1, 1)

        self.label = QLabel(self.gridLayoutWidget)
        self.label.setObjectName(u"label")
        self.label.setFont(font1)

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.lineEdit = QLineEdit(self.gridLayoutWidget)
        self.lineEdit.setObjectName(u"lineEdit")

        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 1)

        self.lineEdit_2 = QLineEdit(self.gridLayoutWidget)
        self.lineEdit_2.setObjectName(u"lineEdit_2")
        font2 = QFont()
        font2.setPointSize(8)
        self.lineEdit_2.setFont(font2)

        self.gridLayout.addWidget(self.lineEdit_2, 0, 3, 1, 1)

        self.label_4 = QLabel(self.gridLayoutWidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font1)

        self.gridLayout.addWidget(self.label_4, 1, 0, 1, 1)

        self.lineEdit_3 = QLineEdit(self.gridLayoutWidget)
        self.lineEdit_3.setObjectName(u"lineEdit_3")
        self.lineEdit_3.setFont(font2)

        self.gridLayout.addWidget(self.lineEdit_3, 1, 1, 1, 1)

        self.label_5 = QLabel(self.gridLayoutWidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setFont(font1)

        self.gridLayout.addWidget(self.label_5, 1, 2, 1, 1)

        self.lineEdit_4 = QLineEdit(self.gridLayoutWidget)
        self.lineEdit_4.setObjectName(u"lineEdit_4")
        self.lineEdit_4.setFont(font2)

        self.gridLayout.addWidget(self.lineEdit_4, 1, 3, 1, 1)


        self.retranslateUi(AlgoDialog)
        self.buttonBox.accepted.connect(AlgoDialog.accept)
        self.buttonBox.rejected.connect(AlgoDialog.reject)

        QMetaObject.connectSlotsByName(AlgoDialog)
    # setupUi

    def retranslateUi(self, AlgoDialog):
        AlgoDialog.setWindowTitle(QCoreApplication.translate("AlgoDialog", u"\u8bbe\u7f6e\u7b97\u6cd5", None))
#if QT_CONFIG(whatsthis)
        self.select2.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(whatsthis)
        self.select1.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.label_2.setText(QCoreApplication.translate("AlgoDialog", u"\u7b97\u6cd5\u540d\u79f0", None))
        self.label_1.setText(QCoreApplication.translate("AlgoDialog", u"\u7b97\u6cd5\u7c7b\u578b", None))
        self.algoTxt.setText(QCoreApplication.translate("AlgoDialog", u"geatpy\u5e93\u7b97\u6cd5", None))
        self.label_3.setText(QCoreApplication.translate("AlgoDialog", u"\u6700\u5927\u65f6\u95f4(s):", None))
        self.label.setText(QCoreApplication.translate("AlgoDialog", u"\u6700\u5927\u8fdb\u5316\u4ee3\u6570:", None))
        self.lineEdit.setText(QCoreApplication.translate("AlgoDialog", u"100", None))
        self.lineEdit_2.setText(QCoreApplication.translate("AlgoDialog", u"None", None))
        self.label_4.setText(QCoreApplication.translate("AlgoDialog", u"\u6700\u5927\u8bc4\u4ef7\u6b21\u6570\n"
"(\u8fdb\u5316\u4ee3\u6570*\u79cd\u7fa4\u89c4\u6a21):", None))
        self.lineEdit_3.setText(QCoreApplication.translate("AlgoDialog", u"None", None))
        self.label_5.setText(QCoreApplication.translate("AlgoDialog", u"\u6700\u4f18\u4e2a\u4f53\u6700\u5927\u6570\u76ee:", None))
        self.lineEdit_4.setText(QCoreApplication.translate("AlgoDialog", u"None", None))
    # retranslateUi

