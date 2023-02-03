# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'popudialog.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_PopuDialog(object):
    def setupUi(self, PopuDialog):
        if not PopuDialog.objectName():
            PopuDialog.setObjectName(u"PopuDialog")
        PopuDialog.resize(400, 300)
        self.buttonBox = QDialogButtonBox(PopuDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setGeometry(QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        self.label_1 = QLabel(PopuDialog)
        self.label_1.setObjectName(u"label_1")
        self.label_1.setGeometry(QRect(80, 40, 81, 21))
        font = QFont()
        font.setFamily(u"\u9ed1\u4f53")
        font.setPointSize(10)
        self.label_1.setFont(font)
        self.select1 = QComboBox(PopuDialog)
        self.select1.setObjectName(u"select1")
        self.select1.setGeometry(QRect(40, 70, 141, 31))
        self.label_2 = QLabel(PopuDialog)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(220, 40, 121, 21))
        self.label_2.setFont(font)
        self.lineEdit = QLineEdit(PopuDialog)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QRect(220, 70, 111, 31))
        self.encodeTxt = QLabel(PopuDialog)
        self.encodeTxt.setObjectName(u"encodeTxt")
        self.encodeTxt.setGeometry(QRect(40, 110, 131, 31))

        self.retranslateUi(PopuDialog)
        self.buttonBox.accepted.connect(PopuDialog.accept)
        self.buttonBox.rejected.connect(PopuDialog.reject)

        QMetaObject.connectSlotsByName(PopuDialog)
    # setupUi

    def retranslateUi(self, PopuDialog):
        PopuDialog.setWindowTitle(QCoreApplication.translate("PopuDialog", u"\u521d\u59cb\u5316\u79cd\u7fa4", None))
        self.label_1.setText(QCoreApplication.translate("PopuDialog", u"\u7f16\u7801\u65b9\u5f0f", None))
#if QT_CONFIG(whatsthis)
        self.select1.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.label_2.setText(QCoreApplication.translate("PopuDialog", u"\u79cd\u7fa4\u4e2a\u4f53\u6570\u76ee", None))
        self.lineEdit.setText(QCoreApplication.translate("PopuDialog", u"30", None))
        self.encodeTxt.setText(QCoreApplication.translate("PopuDialog", u"RI: \u5b9e\u6570\u7f16\u7801", None))
    # retranslateUi

