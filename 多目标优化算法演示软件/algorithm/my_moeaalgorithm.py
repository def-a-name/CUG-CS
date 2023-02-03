# -*- coding: utf-8 -*-
import time
import warnings
import numpy as np
import geatpy as ea
from PySide2.QtCore import QObject, Signal


class mySignal(QObject):
    strSignal = Signal(str)

class my_MoeaAlgorithm(ea.MoeaAlgorithm):  # 多目标优化算法类的父类
    signal = mySignal()
    """
    描述:
        此为多目标进化优化算法类的父类，所有多目标优化算法类均继承自该父类。

    对比于父类该类新增的变量和函数:

        drawing        : int - 绘图方式的参数，
                               0表示不绘图；
                               1表示绘制最终结果图；
                               2表示实时绘制目标空间动态图；
                               3表示实时绘制决策空间动态图。
        draw()         : 绘图函数。

    """

    def __init__(self,
                 problem,
                 population,
                 MAXGEN = None,
                 MAXTIME = None,
                 MAXEVALS = None,
                 MAXSIZE = None,
                 logTras = None,
                 verbose = None,
                 outFunc = None,
                 drawing = None,
                 dirName = None,
                 **kwargs):

        """
        描述: 
            在该构造函数里只初始化静态参数以及对动态参数进行定义。
        
        """

        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, dirName)  # 先调用父类构造函数
        self.drawing = 1 if drawing is None else drawing
        # 以下为用户不需要设置的属性
        self.plotter = None  # 存储绘图对象

    # def __str__(self):
    #     info = {}
    #     info['Algorithm Name'] = self.name
    #     info['Algorithm MAXGEN'] = self.MAXGEN
    #     info['Algorithm MAXTIME'] = self.MAXTIME
    #     info['Algorithm MAXEVALS'] = self.MAXEVALS
    #     return str(info)

    # def initialization(self):

    #     """
    #     描述: 
    #         该函数用于在进化前对算法类的一些动态参数进行初始化操作。
    #         该函数需要在执行算法类的run()函数的一开始被调用，同时开始计时，
    #         以确保所有这些参数能够被正确初始化。
        
    #     """

    #     self.ax = None  # 初始化ax
    #     self.passTime = 0  # 初始化passTime
    #     self.log = None  # 初始化log
    #     self.currentGen = 0  # 初始为第0代
    #     self.evalsNum = 0  # 初始化评价次数为0
    #     self.log = {'gen': [], 'eval': []} if self.logTras != 0 else None  # 初始化log
    #     self.timeSlot = time.time()  # 开始计时

    # def logging(self, pop):

    #     """
    #     描述:
    #         用于在进化过程中记录日志。该函数在stat()函数里面被调用。
    #         如果需要在日志中记录其他数据，需要在自定义算法类中重写该函数。

    #     输入参数:
    #         pop : class <Population> - 种群对象。

    #     输出参数:
    #         无输出参数。

    #     """

    #     self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算logging的耗时
    #     if len(self.log['gen']) == 0:  # 初始化log的各个键值
    #         self.log['gd'] = []
    #         self.log['igd'] = []
    #         self.log['hv'] = []
    #         self.log['spacing'] = []
    #     self.log['gen'].append(self.currentGen)
    #     self.log['eval'].append(self.evalsNum)  # 记录评价次数
    #     [levels, _] = ea.ndsortDED(pop.ObjV, needLevel=1, CV=pop.CV, maxormins=self.problem.maxormins)  # 非支配分层
    #     NDSet = pop[np.where(levels == 1)[0]]  # 只保留种群中的非支配个体，形成一个非支配种群
    #     if self.problem.ReferObjV is not None:
    #         self.log['gd'].append(ea.indicator.GD(NDSet.ObjV, self.problem.ReferObjV))  # 计算GD指标
    #         self.log['igd'].append(ea.indicator.IGD(NDSet.ObjV, self.problem.ReferObjV))  # 计算IGD指标
    #         self.log['hv'].append(ea.indicator.HV(NDSet.ObjV), )  # 计算HV指标
    #     else:
    #         self.log['gd'].append(None)
    #         self.log['igd'].append(None)
    #         self.log['hv'].append(ea.indicator.HV(NDSet.ObjV))  # 计算HV指标
    #     self.log['spacing'].append(ea.indicator.Spacing(NDSet.ObjV))  # 计算Spacing指标
    #     self.timeSlot = time.time()  # 更新时间戳

    def draw(self, pop, EndFlag=False):

        """
        描述:
            该函数用于在进化过程中进行绘图。该函数在stat()以及finishing函数里面被调用。

        输入参数:
            pop     : class <Population> - 种群对象。
            
            EndFlag : bool - 表示是否是最后一次调用该函数。

        输出参数:
            无输出参数。

        """

        if not EndFlag:
            self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算画图的耗时
            # 绘制动画
            if self.drawing == 2:
                # 绘制目标空间动态图
                if pop.ObjV.shape[1] == 2 or pop.ObjV.shape[1] == 3:
                    if self.plotter is None:
                        figureName = 'Pareto Front Plot'
                        self.plotter = ea.PointScatter(self.problem.M,grid=True,legend=True,title=figureName,saveName=self.dirName+"\\"+figureName)
                    self.plotter.refresh()
                    self.plotter.add(pop.ObjV, color='red', label='MOEA PF at ' + str(self.currentGen) + ' Generation')
                else:
                    if self.plotter is None:
                        figureName = 'Parallel Coordinate Plot'
                        self.plotter = ea.ParCoordPlotter(self.problem.M, grid=True, legend=True, title=figureName,saveName=self.dirName+"\\"+figureName)
                    self.plotter.refresh()
                    self.plotter.add(pop.ObjV, color='red', label='MOEA Objective Value at ' + str(self.currentGen) + ' Generation')
                self.plotter.draw()
            elif self.drawing == 3:
                # 绘制决策空间动态图
                if self.plotter is None:
                    figureName = 'Variables Value Plot'
                    self.plotter = ea.ParCoordPlotter(self.problem.Dim, grid=True, legend=True, title=figureName,saveName=self.dirName+"\\"+figureName)
                self.plotter.refresh()
                self.plotter.add(pop.Phen, marker='o', color='blue', label='Variables Value at ' + str(self.currentGen) + ' Generation')
                self.plotter.draw()
            self.timeSlot = time.time()  # 更新时间戳
        else:
            # 绘制最终结果图
            if self.drawing != 0:
                if self.plotter is not None:  # 若绘制了动画，则保存并关闭动画
                    self.plotter.createAnimation()
                    self.plotter.close()
                if pop.ObjV.shape[1] == 2 or pop.ObjV.shape[1] == 3:
                    figureName = 'Pareto Front Plot'
                    self.plotter = ea.PointScatter(self.problem.M, grid=True, legend=True, title=figureName, saveName=self.dirName+"\\"+figureName)
                    self.plotter.add(self.problem.ReferObjV, color='gray', alpha=0.1, label='True PF')
                    self.plotter.add(pop.ObjV, color='red', label='MOEA PF')
                    self.plotter.draw()
                else:
                    figureName = 'Parallel Coordinate Plot'
                    self.plotter = ea.ParCoordPlotter(self.problem.M, grid=True, legend=True, title=figureName, saveName=self.dirName+"\\"+figureName)
                    self.plotter.add(self.problem.TinyReferObjV, color='gray', alpha=0.5, label='True Objective Value')
                    self.plotter.add(pop.ObjV, color='red', label='MOEA Objective Value')
                    self.plotter.draw()

    def display(self):
        """
        描述:
            该函数打印日志log中每个键值的最后一条数据。假如log中只有一条数据或没有数据，则会打印表头。
            该函数将会在子类中被覆盖，以便进行更多其他的输出展示。
        修改:
            将输出改为返回字符串
        """
        self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算display()的耗时
        headers = []
        widths = []
        values = []
        for key in self.log.keys():
            # 设置单元格宽度
            if key == 'gen':
                if self.MAXGEN is None:
                    width = 5
                else:
                    width = max(3, len(str(self.MAXGEN - 1)))  # 因为字符串'gen'长度为3，所以最小要设置长度为3
            elif key == 'eval':
                width = max(8, len(str(self.evalsNum - 1)))  # 因为字符串'eval'长度为4，所以最小要设置长度为4
            else:
                width = 13  # 预留13位显示长度，若数值过大，表格将无法对齐，此时若要让表格对齐，需要自定义算法类重写该函数
            headers.append(key)
            widths.append(width)
            value = self.log[key][-1] if len(self.log[key]) != 0 else "-"
            if isinstance(value, float):
                values.append("%.3E" % value)  # 格式化浮点数，输出时只保留至小数点后5位
            else:
                values.append(value)
        logText = ""
        if len(self.log['gen']) == 1:  # 打印表头
            header_regex = '|'.join(['{}'] * len(headers))
            header_str = header_regex.format(*[str(key).center(width) for key, width in zip(headers, widths)])
            logText = logText + ("=" * len(header_str) + "\n")
            logText = logText + (header_str + "\n")
            logText = logText + ("-" * len(header_str) + "\n")
        if len(self.log['gen']) != 0:  # 打印表格最后一行
            value_regex = '|'.join(['{}'] * len(values))
            value_str = value_regex.format(*[str(value).center(width) for value, width in zip(values, widths)])
            logText = logText + (value_str + "\n")
        self.timeSlot = time.time()  # 更新时间戳
        self.signal.strSignal.emit(logText)

    # def stat(self, pop):

    #     """
    #     描述:
    #         该函数用于分析当代种群的信息。
    #         该函数会在terminated()函数里被调用。

    #     输入参数:
    #         pop : class <Population> - 种群对象。

    #     输出参数:
    #         无输出参数。

    #     """

    #     feasible = np.where(np.all(pop.CV <= 0, 1))[0] if pop.CV is not None else np.arange(pop.sizes)  # 找到满足约束条件的个体的下标
    #     if len(feasible) > 0:
    #         feasiblePop = pop[feasible]  # 获取满足约束条件的个体
    #         if self.logTras != 0 and self.currentGen % self.logTras == 0:
    #             self.logging(feasiblePop)  # 记录日志
    #             if self.verbose:
    #                 self.display()  # 打印日志
    #         self.draw(feasiblePop)  # 展示输出

    # def terminated(self, pop):

    #     """
    #     描述:
    #         该函数用于判断是否应该终止进化，population为传入的种群对象。
    #         该函数会在各个具体的算法类的run()函数中被调用。

    #     输入参数:
    #         pop : class <Population> - 种群对象。

    #     输出参数:
    #         True / False。
            
    #     """

    #     self.check(pop)  # 检查种群对象的关键属性是否有误
    #     self.stat(pop)  # 进行统计分析
    #     self.passTime += time.time() - self.timeSlot  # 更新耗时
    #     # 调用outFunc()
    #     if self.outFunc is not None:
    #         if not callable(self.outFunc):
    #             raise RuntimeError('outFunc must be a function. (如果定义了outFunc，那么它必须是一个函数。)')
    #         self.outFunc(self, pop)
    #     self.timeSlot = time.time()  # 更新时间戳
    #     # 判断是否终止进化，
    #     if self.MAXGEN is None and self.MAXTIME is None and self.MAXEVALS is None:
    #         raise RuntimeError('error: MAXGEN, MAXTIME, and MAXEVALS cannot be all None. (MAXGEN, MAXTIME, 和MAXEVALS不能全为None)')
    #     terminatedFlag = False
    #     if self.MAXGEN is not None and self.currentGen + 1 >= self.MAXGEN:  # 由于代数是从0数起，因此在比较currentGen和MAXGEN时需要对currentGen加1
    #         self.stopMsg = 'The algotirhm stepped because it exceeded the generation limit.'
    #         terminatedFlag = True
    #     if self.MAXTIME is not None and self.passTime >= self.MAXTIME:
    #         self.stopMsg = 'The algotirhm stepped because it exceeded the time limit.'
    #         terminatedFlag = True
    #     if self.MAXEVALS is not None and self.evalsNum >= self.MAXEVALS:
    #         self.stopMsg = 'The algotirhm stepped because it exceeded the function evaluation limit.'
    #         terminatedFlag = True
    #     if terminatedFlag:
    #         return True
    #     else:
    #         self.currentGen += 1  # 进化代数+1
    #         return False

    def finishing(self, pop, globalNDSet=None):

        """
        描述:
            进化完成后调用的函数。

        输入参数:
            pop : class <Population> - 种群对象。
            
            globalNDSet : class <Population> - (可选参数)全局存档。

        输出参数:
            [NDSet, pop]，其中pop为种群类型；NDSet的类型与pop的一致。

        """
        if globalNDSet is None:
            # 得到非支配种群
            [levels, _] = ea.ndsortDED(pop.ObjV, needLevel=1, CV=pop.CV, maxormins=self.problem.maxormins)  # 非支配分层
            NDSet = pop[np.where(levels == 1)[0]]  # 只保留种群中的非支配个体，形成一个非支配种群
            if NDSet.CV is not None:  # CV不为None说明有设置约束条件
                NDSet = NDSet[np.where(np.all(NDSet.CV <= 0, 1))[0]]  # 最后要彻底排除非可行解
        else:
            NDSet = globalNDSet
        if self.logTras != 0 and NDSet.sizes != 0 and (
                len(self.log['gen']) == 0 or self.log['gen'][-1] != self.currentGen):  # 补充记录日志和输出
            self.logging(NDSet)
            if self.verbose:
                self.display()
        self.passTime += time.time() - self.timeSlot  # 更新用时记录，因为已经要结束，因此不用再更新时间戳
        NDSet.save(self.dirName) # 把非支配种群的信息保存到文件中
        self.signal.strSignal.emit('用时：%s 秒' % self.passTime)
        if NDSet.sizes != 0:
            self.signal.strSignal.emit('非支配个体数：%d 个' % NDSet.sizes)
        else:
            self.signal.strSignal.emit('没有找到可行解！')
        self.draw(NDSet, EndFlag=True)  # 显示最终结果图
        # if self.plotter is not None:
        #     self.plotter.show()
        # 返回帕累托最优个体以及最后一代种群
        return [NDSet, pop]