from PySide2.QtWidgets import QWidget,QApplication,QMainWindow,QPushButton
from PySide2.QtWidgets import QPlainTextEdit,QMessageBox,QDialog,QFileDialog,QStackedWidget
from PySide2.QtUiTools import QUiLoader
from PySide2 import QtSvg, QtGui
from manager import *
from mythread import *
from gui.ui_quesdialog import Ui_QuesDialog
from gui.ui_popudialog import Ui_PopuDialog
from gui.ui_algodialog import Ui_AlgoDialog

import sys
import os
import time
import win32api
import warnings
warnings.filterwarnings("ignore")

qlist1 = ["连续多目标", "离散多目标", "RCM约束多目标", "自定义"]
qlist2 = [["BNH","OSY","SRN","CON","ZDT1","ZDT2","ZDT3","ZDT4","ZDT6","DTLZ1"],
        ["MP1","KNAPSACK1","KNAPSACK2","ZDT5","MOTSP"],
        ["RCM01","RCM05","RCM08","RCM10","RCM23","RCM24","RCM26","RCM29","RCM31"],
        []]
qlist3 = ["kroA100", "kroB100", "kroC100", "kroD100"]
alist1 = ["NSGA-II", "NSGA-III", "MOEA/D", "自定义"]
alist2 = [["moea_NSGA2_templet","moea_NSGA2_DE_templet","moea_NSGA2_archive_templet","my_NSGA2"],
        ["moea_NSGA3_templet","moea_NSGA3_DE_templet"],
        ["moea_MOEAD_templet","moea_MOEAD_DE_templet","moea_MOEAD_archive_templet"],
        []]

class QuesDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_QuesDialog()
        self.ui.setupUi(self)
        
        self.list1 = qlist1
        self.list2 = qlist2
        self.list3 = qlist3
        self.ui.select1.addItems(self.list1)
        self.ui.select2.addItems(self.list2[0])
        self.ui.tspSelect1.addItems(self.list3)
        self.ui.tspSelect2.addItems(self.list3)
        self.ui.label_3.hide()
        self.ui.label_4.hide()
        self.ui.tspSelect1.hide()
        self.ui.tspSelect2.hide()

        self.ui.select1.currentIndexChanged.connect(self.box1Change)
        self.ui.select2.currentIndexChanged.connect(self.box2Change)

    def box1Change(self, curIndex):
    	self.ui.select2.clear()
    	self.ui.select2.addItems(self.list2[curIndex])

    def box2Change(self):
        curText = self.ui.select2.currentText()
        if(curText == "MOTSP"):
            self.ui.label_3.show()
            self.ui.label_4.show()
            self.ui.tspSelect1.show()
            self.ui.tspSelect2.show()
        else:
            self.ui.label_3.hide()
            self.ui.label_4.hide()
            self.ui.tspSelect1.hide()
            self.ui.tspSelect2.hide()

class PopuDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_PopuDialog()
        self.ui.setupUi(self)
        
        self.list = ["RI", "BG", "P"]
        self.ui.select1.addItems(self.list)

        self.ui.select1.currentIndexChanged.connect(self.boxChange)

    def boxChange(self, curIndex):
        if(curIndex == 0):
            self.ui.encodeTxt.setText("RI: 实数编码")
        elif(curIndex == 1):
            self.ui.encodeTxt.setText("BG: 二进制编码")
        elif(curIndex == 2):
            self.ui.encodeTxt.setText("P: 排列编码")

class AlgoDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_AlgoDialog()
        self.ui.setupUi(self)
        
        self.list1 = alist1
        self.list2 = alist2
        self.ui.select1.addItems(self.list1)
        self.ui.select2.addItems(self.list2[0])

        self.ui.select1.currentIndexChanged.connect(self.box1Change)
        self.ui.select2.currentIndexChanged.connect(self.box2Change)

    def box1Change(self, curIndex):
        self.ui.select2.clear()
        self.ui.select2.addItems(self.list2[curIndex])

    def box2Change(self):
        curText = self.ui.select2.currentText()
        if(curText[:4] == "moea"):
            self.ui.algoTxt.setText("geatpy库算法")
        else:
            self.ui.algoTxt.setText("自定义算法")

class SvgLabel(QLabel):
    def __init__(self, imgName):
        super(SvgLabel, self).__init__()
        self.imgName = imgName

    def paintEvent(self, event):
        renderer =  QtSvg.QSvgRenderer(self.imgName)
        painter = QtGui.QPainter(self)
        renderer.render(painter)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = QUiLoader().load('gui/mainwindow.ui')
        self.m = Manager()
        self.ui.move(500, 300)

        self.ui.resBtn.setEnabled(False)
        self.ui.gdBtn.setEnabled(False)
        self.ui.igdBtn.setEnabled(False)
        self.ui.hvBtn.setEnabled(False)
        self.ui.spaBtn.setEnabled(False)
        self.ui.chromBtn.setEnabled(False)
        self.ui.objVBtn.setEnabled(False)
        self.ui.fitVBtn.setEnabled(False)
        self.ui.phenBtn.setEnabled(False)
        self.ui.cvBtn.setEnabled(False)
        self.ui.qcodeBtn.setEnabled(False)
        self.ui.acodeBtn.setEnabled(False)

        self.ui.quesBtn.clicked.connect(self.openQuesDialog)
        self.ui.popuBtn.clicked.connect(self.openPopuDialog)
        self.ui.algoBtn.clicked.connect(self.openAlgoDialog)
        self.ui.startBtn.clicked.connect(self.run)
        self.ui.dirBtn.clicked.connect(self.openSaveDialog)
        self.ui.resBtn.clicked.connect(self.showResGraph)
        self.ui.gdBtn.clicked.connect(self.showGDGraph)
        self.ui.igdBtn.clicked.connect(self.showIGDGraph)
        self.ui.hvBtn.clicked.connect(self.showHVGraph)
        self.ui.spaBtn.clicked.connect(self.showSpacingGraph)
        self.ui.chromBtn.clicked.connect(self.showChrom)
        self.ui.objVBtn.clicked.connect(self.showObjV)
        self.ui.fitVBtn.clicked.connect(self.showFitV)
        self.ui.phenBtn.clicked.connect(self.showPhen)
        self.ui.cvBtn.clicked.connect(self.showCV)
        self.ui.qcodeBtn.clicked.connect(self.showQuesCode)
        self.ui.acodeBtn.clicked.connect(self.showAlgoCode)
        self.ui.newqBtn.clicked.connect(self.newQuesCode)
        self.ui.newaBtn.clicked.connect(self.newAlgoCode)
        self.ui.setNewqBtn.clicked.connect(self.addQuesCode)
        self.ui.setNewaBtn.clicked.connect(self.addAlgoCode)
        self.ui.dirEdit.textChanged.connect(self.updateSavedir)
        self.ui.logEdit.textChanged.connect(self.updateLogtras)
        self.m.signal.connect(self.handleMSignal)

    def openQuesDialog(self):
        dialog = QuesDialog()
        if(dialog.exec_()):
            prob = dialog.ui.select2.currentText()
            flag = False
            if(prob == "MOTSP"):
                data1 = dialog.ui.tspSelect1.currentText()
                data2 = dialog.ui.tspSelect2.currentText()
                flag = self.m.setProblem(prob, data1, data2)
            else:
                flag = self.m.setProblem(prob)
            if(flag):
                self.ui.quesText.setText("问题描述：")
                self.ui.quesText.append(self.m.problemText())
                self.ui.qcodeBtn.setEnabled(True)

    def openPopuDialog(self):
        dialog = PopuDialog()
        if(dialog.exec_()):
            encode = dialog.ui.select1.currentText()
            popsize = int(dialog.ui.lineEdit.text())
            flag = self.m.setPopulation(encode, popsize)
            if(flag):
                self.ui.popuText.setText("种群描述：")
                self.ui.popuText.append(self.m.populationText())

    def openAlgoDialog(self):
        dialog = AlgoDialog()
        if(dialog.exec_()):
            try:
                algo = dialog.ui.select2.currentText()
                maxgen = int(dialog.ui.lineEdit.text())
                maxtime = maxeval = maxsize = None
                if(dialog.ui.lineEdit_2.text() != "None"):
                    maxtime = int(dialog.ui.lineEdit_2.text())
                if(dialog.ui.lineEdit_3.text() != "None"):
                    maxeval = int(dialog.ui.lineEdit_3.text())
                if(dialog.ui.lineEdit_4.text() != "None"):
                    maxsize = int(dialog.ui.lineEdit_4.text())
                logtras = int(self.ui.logEdit.text())
                drawings = 1
                flag = self.m.setAlgorithm(algo, maxgen, maxtime, maxeval, maxsize, logtras, drawings)
                if(flag):
                    self.ui.algoText.setText("算法描述：")
                    self.ui.algoText.append(self.m.algorithmText())
                    # self.ui.logText.append("算法运行日志：")
                    self.ui.acodeBtn.setEnabled(True)
            except ValueError as ve:
                QMessageBox.critical(self, "算法设置错误", "参数设置错误")

    def openSaveDialog(self):
        fileName, _ = QFileDialog.getSaveFileName(self, "结果保存位置", self.m.savedir)
        if(fileName != ""):
            self.ui.dirEdit.setText(fileName)

    def showResGraph(self):
        self.labelRes = SvgLabel(self.m.savedir + "\\Pareto Front Plot.svg")
        self.labelRes.setGeometry(self.ui.x()+100,self.ui.y()-80,800,600)
        self.labelRes.show()

    def showGDGraph(self):
        self.labelGD = SvgLabel(self.m.savedir + "\\gd.svg")       
        self.labelGD.setGeometry(self.ui.x(),self.ui.y()-80,400,300)
        self.labelGD.show()

    def showIGDGraph(self):
        self.labelIGD = SvgLabel(self.m.savedir + "\\igd.svg")       
        self.labelIGD.setGeometry(self.ui.x()+self.ui.width()-400,self.ui.y()-80,400,300)
        self.labelIGD.show()

    def showHVGraph(self):
        self.labelHV = SvgLabel(self.m.savedir + "\\hv.svg")       
        self.labelHV.setGeometry(self.ui.x(),self.ui.y()+self.ui.height()-300-80,400,300)
        self.labelHV.show()

    def showSpacingGraph(self):
        self.labelSpa = SvgLabel(self.m.savedir + "\\spacing.svg")       
        self.labelSpa.setGeometry(self.ui.x()+self.ui.width()-400,self.ui.y()+self.ui.height()-300-80,400,300)
        self.labelSpa.show()

    def showChrom(self):
        try:
            os.startfile(self.m.savedir + "\\Chrom.csv")
        except FileNotFoundError as ne:
            QMessageBox.critical(self, "文件打开错误", "最终代基因型矩阵未保存")

    def showObjV(self):
        try:
            os.startfile(self.m.savedir + "\\ObjV.csv")
        except FileNotFoundError as ne:
            QMessageBox.critical(self, "文件打开错误", "最终代目标函数矩阵未保存")

    def showFitV(self):
        try:
            os.startfile(self.m.savedir + "\\FitnV.csv")
        except FileNotFoundError as ne:
            QMessageBox.critical(self, "文件打开错误", "最终代适应值函数矩阵未保存")

    def showPhen(self):
        try:
            os.startfile(self.m.savedir + "\\Phen.csv")
        except FileNotFoundError as ne:
            QMessageBox.critical(self, "文件打开错误", "最终代决策变量矩阵未保存")

    def showCV(self):
        try:
            os.startfile(self.m.savedir + "\\CV.csv")
        except FileNotFoundError as ne:
            QMessageBox.critical(self, "文件打开错误", "最终代违反约束矩阵未保存，请检查是否为带约束问题")

    def showQuesCode(self):
        curdir = os.path.abspath(os.path.dirname(__file__))
        if(self.m.problem.name in qlist2[0]):
            win32api.ShellExecute(0, 'open', 'notepad.exe', 'problem/MyProblemCont.py', '', 1);
        elif(self.m.problem.name in qlist2[1]):
            win32api.ShellExecute(0, 'open', 'notepad.exe', 'problem/MyProblemComb.py', '', 1);
        elif(self.m.problem.name in qlist2[2]):
            win32api.ShellExecute(0, 'open', 'notepad.exe', 'problem/MyProblemRCM.py', '', 1);
        elif(self.m.problem.name in qlist2[3]):
            win32api.ShellExecute(0, 'open', 'notepad.exe', 'problem/MyProblemCust.py', '', 1);
        else:
             QMessageBox.critical(self, "文件打开错误", "找不到对应的问题文件")

    def showAlgoCode(self):
        algoName = self.m.algorithm.name
        if(algoName in alist2[0]):
            win32api.ShellExecute(0, 'open', 'notepad.exe', 'algorithm/nsga2/%s.py'%(algoName), '', 1);
        elif(algoName in alist2[1]):
            win32api.ShellExecute(0, 'open', 'notepad.exe', 'algorithm/nsga3/%s.py'%(algoName), '', 1);
        elif(algoName in alist2[2]):
            win32api.ShellExecute(0, 'open', 'notepad.exe', 'algorithm/moead/%s.py'%(algoName), '', 1);
        elif(algoName in alist2[3]):
            win32api.ShellExecute(0, 'open', 'notepad.exe', 'algorithm/algorithm_custom.py', '', 1);
        else:
             QMessageBox.critical(self, "文件打开错误", "找不到对应的算法文件")

    def newQuesCode(self):
        win32api.ShellExecute(0, 'open', 'notepad.exe', 'problem/MyProblemCust.py', '', 1);

    def newAlgoCode(self):
        win32api.ShellExecute(0, 'open', 'notepad.exe', 'algorithm/algorithm_custom.py', '', 1);

    def addQuesCode(self):
        if(self.ui.newqEdit.text() == ""):
            QMessageBox.critical(self, "文件添加错误", "自定义问题名为空")
        else:
            if(self.ui.newqEdit.text() not in qlist2[-1]):
                qlist2[-1].append(self.ui.newqEdit.text())
            QMessageBox.information(self, "文件添加成功", "自定义问题已添加")

    def addAlgoCode(self):
        if(self.ui.newaEdit.text() == ""):
            QMessageBox.critical(self, "文件添加错误", "自定义算法名为空")
        else:
            if(self.ui.newaEdit.text() not in alist2[-1]):
                alist2[-1].append(self.ui.newaEdit.text())
            __import__('algorithm.algorithm_custom', fromlist = ['*'])
            QMessageBox.information(self, "文件添加成功", "自定义算法已添加")

    def updateSavedir(self, newdir):
        self.m.savedir = newdir

    def updateLogtras(self, newLogtras):
        if(self.m.algorithm != None):
            try:
                self.m.algorithm.logTras = int(newLogtras)
            except ValueError as ve:
                QMessageBox.critical(self, "算法设置错误", "日志间隔格式错误")

    def handleMSignal(self, sigVal):
        if(sigVal == 0):    # Manager.runAlgorithm()算法未定义
            QMessageBox.critical(self, "算法运行错误", "算法未设置完成")
        elif(sigVal == 1):  # Manager.runAlgorithm()找到可行解
            # QMessageBox.information(self, "算法运行成功", "算法运行完毕，非支配种群信息已保存")
            self.ui.resBtn.setEnabled(True)
            self.ui.gdBtn.setEnabled(True)
            self.ui.igdBtn.setEnabled(True)
            self.ui.hvBtn.setEnabled(True)
            self.ui.spaBtn.setEnabled(True)
            self.ui.chromBtn.setEnabled(True)
            self.ui.objVBtn.setEnabled(True)
            self.ui.fitVBtn.setEnabled(True)
            self.ui.phenBtn.setEnabled(True)
            self.ui.cvBtn.setEnabled(True)
        elif(sigVal == 2):  # Manager结果文件更新
            self.ui.dirEdit.setText(self.m.savedir)
            self.ui.resBtn.setEnabled(False)
            self.ui.gdBtn.setEnabled(False)
            self.ui.igdBtn.setEnabled(False)
            self.ui.hvBtn.setEnabled(False)
            self.ui.spaBtn.setEnabled(False)
            self.ui.chromBtn.setEnabled(False)
            self.ui.objVBtn.setEnabled(False)
            self.ui.fitVBtn.setEnabled(False)
            self.ui.phenBtn.setEnabled(False)
            self.ui.cvBtn.setEnabled(False)
        elif(sigVal == 3):  # Manager日志文件更新
            self.ui.logText.append(self.m.logText)

    def run(self):
        self.ui.startBtn.setEnabled(False)
        # self.tlist.append(my_Thread_run(self.m))
        # self.tlist[-1].finished.connect(self.m.showResult)
        # self.tlist[-1].finished.connect(self.resetStartBtn)
        # self.tlist[-1].start()
        self.t = my_Thread_run(self.m)
        self.t.finished.connect(self.m.showResult)
        self.t.finished.connect(self.resetStartBtn)
        self.t.start()
        
    def resetStartBtn(self):
        self.ui.startBtn.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.ui.show()
    
    sys.exit(app.exec_())