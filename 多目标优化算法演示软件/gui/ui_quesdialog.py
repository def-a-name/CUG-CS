# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'quesdialog.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_QuesDialog(object):
    def setupUi(self, QuesDialog):
        if not QuesDialog.objectName():
            QuesDialog.setObjectName(u"QuesDialog")
        QuesDialog.resize(400, 300)
        self.buttonBox = QDialogButtonBox(QuesDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setGeometry(QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        self.select1 = QComboBox(QuesDialog)
        self.select1.setObjectName(u"select1")
        self.select1.setGeometry(QRect(30, 70, 141, 31))
        self.select2 = QComboBox(QuesDialog)
        self.select2.setObjectName(u"select2")
        self.select2.setGeometry(QRect(210, 70, 141, 31))
        self.label = QLabel(QuesDialog)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(70, 40, 81, 21))
        font = QFont()
        font.setFamily(u"\u9ed1\u4f53")
        font.setPointSize(10)
        self.label.setFont(font)
        self.label_2 = QLabel(QuesDialog)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(240, 40, 71, 21))
        self.label_2.setFont(font)
        self.horizontalLayoutWidget = QWidget(QuesDialog)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(80, 150, 281, 41))
        self.tspLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.tspLayout.setObjectName(u"tspLayout")
        self.tspLayout.setContentsMargins(0, 0, 0, 0)
        self.label_3 = QLabel(self.horizontalLayoutWidget)
        self.label_3.setObjectName(u"label_3")

        self.tspLayout.addWidget(self.label_3)

        self.tspSelect1 = QComboBox(self.horizontalLayoutWidget)
        self.tspSelect1.setObjectName(u"tspSelect1")

        self.tspLayout.addWidget(self.tspSelect1)

        self.label_4 = QLabel(self.horizontalLayoutWidget)
        self.label_4.setObjectName(u"label_4")
        font1 = QFont()
        font1.setFamily(u"\u5fae\u8f6f\u96c5\u9ed1")
        font1.setPointSize(10)
        font1.setBold(True)
        font1.setWeight(75)
        self.label_4.setFont(font1)

        self.tspLayout.addWidget(self.label_4)

        self.tspSelect2 = QComboBox(self.horizontalLayoutWidget)
        self.tspSelect2.setObjectName(u"tspSelect2")

        self.tspLayout.addWidget(self.tspSelect2)


        self.retranslateUi(QuesDialog)
        self.buttonBox.accepted.connect(QuesDialog.accept)
        self.buttonBox.rejected.connect(QuesDialog.reject)

        QMetaObject.connectSlotsByName(QuesDialog)
    # setupUi

    def retranslateUi(self, QuesDialog):
        QuesDialog.setWindowTitle(QCoreApplication.translate("QuesDialog", u"\u8bbe\u7f6e\u95ee\u9898", None))
#if QT_CONFIG(whatsthis)
        self.select1.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(whatsthis)
        self.select2.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.label.setText(QCoreApplication.translate("QuesDialog", u"\u95ee\u9898\u7c7b\u578b", None))
        self.label_2.setText(QCoreApplication.translate("QuesDialog", u"\u95ee\u9898\u540d\u79f0", None))
        self.label_3.setText(QCoreApplication.translate("QuesDialog", u"TSP\u95ee\u9898\u96c6", None))
        self.label_4.setText(QCoreApplication.translate("QuesDialog", u"+", None))
    # retranslateUi

