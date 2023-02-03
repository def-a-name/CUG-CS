import sys
import os
import datetime
import importlib
import numpy as np
import geatpy as ea

from algorithm.moead.moea_MOEAD_archive_templet import moea_MOEAD_archive_templet
from algorithm.moead.moea_MOEAD_DE_templet import moea_MOEAD_DE_templet
from algorithm.moead.moea_MOEAD_templet import moea_MOEAD_templet

from algorithm.nsga2.moea_NSGA2_archive_templet import moea_NSGA2_archive_templet
from algorithm.nsga2.moea_NSGA2_DE_templet import moea_NSGA2_DE_templet
from algorithm.nsga2.moea_NSGA2_templet import moea_NSGA2_templet
from algorithm.nsga2.my_NSGA2 import my_NSGA2

from algorithm.nsga3.moea_NSGA3_DE_templet import moea_NSGA3_DE_templet
from algorithm.nsga3.moea_NSGA3_templet import moea_NSGA3_templet

from problem.MyProblemComb import *
from problem.MyProblemCont import *
from problem.MyProblemRCM import *

from PySide2.QtWidgets import QApplication,QWidget,QMessageBox,QLabel
from PySide2.QtCore import Signal
from PySide2 import QtSvg, QtGui

class Manager(QWidget):
	"""算法与问题管理类"""
	signal = Signal(int)
	def __init__(self):
		super(Manager, self).__init__()
		self.problem = None
		self.pCustomLib = None
		self.population = None
		self.algorithm = None
		self.aCustomLib = None
		self.savedir = ""
		self.logText = ""

	def setProblem(self, name, *problemArgs):
		className = ""
		if(len(problemArgs) == 0):
			className = name + "()"
		elif(len(problemArgs) == 2):
			className = name + "('%s','%s')"%(problemArgs[0], problemArgs[1])
		self.pCustomLib = importlib.import_module("problem.MyProblemCust")
		try:
			self.problem = eval(className)
			if(self.algorithm != None):
				self.algorithm.problem = self.problem
				self.updateDir()
			QMessageBox.information(self, "问题成功", "问题:%s 已导入" % (name))
			# print(self.problem)
			return True
		except NameError as ne:
			try:
				className = "self.pCustomLib." + className
				self.problem = eval(className)
				if(self.algorithm != None):
					self.algorithm.problem = self.problem
					self.updateDir()
				QMessageBox.information(self, "问题成功", "问题:%s 已导入" % (name))
				# print(self.problem)
				return True
			except AttributeError as ae:
				QMessageBox.critical(self, "问题错误", "问题:%s 不存在" % (name))
				self.problem = None
		return False
	
	def confirmProblem(self, name):
		## 自定义的算法要导入带惩罚函数的问题类
		## geatpy库的算法则导入不带惩罚函数的问题类
		## motsp问题有数据集参数，单独讨论
		if(self.problem.name == "MOTSP"):
			return

		try:
			if(name[:4] == "moea"):
				problemTmp = eval(self.problem.name + "(punFlag=0)")
				self.problem = problemTmp
			else:
				problemTmp = eval(self.problem.name + "(punFlag=1)")
				self.problem = problemTmp
		except NameError as ne:
			if(name[:4] == "moea"):
				problemTmp = eval("self.pCustomLib." + self.problem.name + "(punFlag=0)")
				self.problem = problemTmp
			else:
				problemTmp = eval("self.pCustomLib." + self.problem.name + "(punFlag=1)")
				self.problem = problemTmp

	def problemText(self):
		if(self.problem != None):
			text = ("问题名: " + str(self.problem.name) + "\n"
					+ "目标函数: " + str(self.problem.M) + "\n"
					+ "求目标最大值/最小值\n(1最小值，-1最大值): " + str(self.problem.maxormins) + "\n"
					+ "决策变量个数: " + str(self.problem.Dim) + "\n"
					+ "决策变量类型\n(0实数，1整数): " + str(self.problem.varTypes) + "\n"
					+ "决策变量下界: " + str(self.problem.lb) + "\n"
					+ "决策变量上界: " + str(self.problem.ub) + "\n"
					+ "下界和上界是否包含边界值\n(0不包含，1包含): " + str(self.problem.borders) + "\n")
			return text

	def setPopulation(self, encode, popsize):
		Encoding = encode	# 编码方式
		NIND = popsize	# 种群规模
		if(self.problem == None):
			QMessageBox.critical(self, "种群错误", "问题未定义")
		else:
			try:
				Field = ea.crtfld(Encoding, self.problem.varTypes, self.problem.ranges,
								self.problem.borders) # 创建区域描述器
				#实例化种群对象（此时种群还没被真正初始化，仅仅是生成一个种群对象）
				self.population = ea.Population(Encoding, Field, NIND)
				if(self.algorithm != None):
					self.algorithm.population = self.population
					self.updateDir()
				QMessageBox.information(self, "种群成功", "种群初始化完成")
				# print(self.population)
				return True
			except RuntimeError as re:
				QMessageBox.critical(self, "种群错误", "种群初始化失败，请检查参数")
				self.population = None
		return False
	
	def populationText(self):
		if(self.population != None):
			text = ("编码方式: " + str(self.population.Encoding) + "\n"
					+ "种群规模: " + str(self.population.sizes) + "\n")
			return text

	def setAlgorithm(self, name, maxgen, maxtime=None, maxeval=None, maxsize=None, logtras=10, drawing=1):
		if(self.problem == None or self.population == None):
			if(self.problem == None):
				QMessageBox.critical(self, "算法错误", "问题未定义")
			if(self.population == None):
				QMessageBox.critical(self, "算法错误", "种群未初始化")
		else:
			algoName = name + '''(self.problem, self.population, 
				                 MAXGEN=maxgen,
				                 MAXTIME=maxtime,
				                 MAXEVALS=maxeval,
				                 MAXSIZE=maxsize,
				                 logTras=logtras,
				                 verbose=True,
				                 outFunc=None,
				                 drawing=drawing,
				                 dirName=None)'''
			self.confirmProblem(name)
			self.aCustomLib = importlib.import_module("algorithm.algorithm_custom")
			try:
				# algo = eval(algoName)
				# self.algorithm = copy.deepcopy(algo)
				if(self.algorithm != None):
					del self.algorithm
				self.algorithm = eval(algoName)
				self.updateDir()
				self.algorithm.signal.strSignal.connect(self.handleAlgoSignal)	# 处理算法发送的日志信号
				QMessageBox.information(self, "算法设置成功", "算法:%s 已加载" % (name))
				# print(self.algorithm)
				return True
			except NameError as ne:
				try:
					algoName = "self.aCustomLib." + algoName
					if(self.algorithm != None):
						del self.algorithm
					self.algorithm = eval(algoName)
					self.updateDir()
					self.algorithm.signal.strSignal.connect(self.handleAlgoSignal)	# 处理算法发送的日志信号
					QMessageBox.information(self, "算法设置成功", "算法:%s 已加载" % (name))
					# print(self.algorithm)
					return True
				except AttributeError as ae:
					QMessageBox.critical(self, "算法设置错误", "算法:%s 不存在" % (name))
					self.algorithm = None
			except RuntimeError as re:
				QMessageBox.critical(self, "算法设置错误", str(re))
				self.algorithm = None
		return False

	def updateDir(self):
		if(self.problem != None and self.algorithm != None):
			curdir = os.path.abspath(os.path.dirname(__file__))
			self.savedir = (curdir+"\\result\\"
							+self.algorithm.name+"_"+self.problem.name+"_"
							+datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S"))
			self.savedir = self.savedir.replace("/", "")	# 避免MOEA/D出现路径错误
			self.algorithm.dirName = self.savedir
			self.signal.emit(2)		# 更新结果文件路径

	def algorithmText(self):
		if(self.algorithm != None):
			text = ("算法名: " + str(self.algorithm.__class__.__name__) + "\n"
					+ "最大迭代次数: " + str(self.algorithm.MAXGEN) + "\n"
					+ "最大时间(s): " + str(self.algorithm.MAXTIME) + "\n"
					+ "最大评价次数: " + str(self.algorithm.MAXEVALS) + "\n"
					+ "Pareto最优解最大数目: " + str(self.algorithm.MAXSIZE) + "\n")
			return text

	def handleAlgoSignal(self, logstr):
		if(self.logText != logstr):
			self.logText = logstr
			# print(logstr)
			self.signal.emit(3)

	def runAlgorithm(self):
		if(self.algorithm == None):
			#在新的线程里运行算法，用信号反馈错误
			self.signal.emit(0)
		else:
			NDSet, population = self.algorithm.run()
			if(NDSet.sizes != 0):
				self.signal.emit(1)
				return True
		return False

	def showResult(self, sigVal):
		if sigVal==1 and (self.algorithm != None) and (self.algorithm.plotter != None):
			self.algorithm.plotter.show()
			self.plotTrcGraph("gd")
			self.plotTrcGraph("igd")
			self.plotTrcGraph("hv")
			self.plotTrcGraph("spacing")

	def plotTrcGraph(self, metricName):
		metric = [[metricName]]
		Metrics = np.array([self.algorithm.log[metric[i][0]] for i in range(len(metric))]).T
		# 绘制指标追踪分析图
		# print(Metrics)
		ea.trcplot(Metrics, labels=metric, titles=metric, save_path=self.savedir+"\\")

		
if __name__ == '__main__':
	app = QApplication(sys.argv)
	
	# m = Manager()
	# m.setProblem("BNH")
	# m.setPopulation("RI", 30)
	# m.setAlgorithm("moea_NSGA2_templet", 100)
	# m.runAlgorithm()
	# m.showResult()
	# m.plotTrcGraph("gd")
	# m.plotTrcGraph("igd")
	# m.plotTrcGraph("hv")
	# m.plotTrcGraph("spacing")

	# m = Manager()
	# m.setProblem('MOTSP', "kroA100", "kroD100")
	# m.setPopulation('RI', 30)
	# m.setAlgorithm("my_NSGA2", 100)
	# m.runAlgorithm()
	# m.showResult()

	sys.exit(app.exec_())
	
	# help(ea.trcplot)
