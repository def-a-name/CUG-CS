import numpy as np
import geatpy as ea
import os

'''
惩罚函数
参考文献:
    [1] Yonas Gebre Woldesenbet, Gary G. Yen, Fellow, IEEE, and Biruk G. Tessema. 
    Constraint Handling in Multiobjective Evolutionary Optimization[J]. 
    IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 13, NO. 3, JUNE 2009.
'''
def getFPunish(f, CV):
    popSize, objNum = f.shape
    fNorm = np.copy(f).astype(np.float64)
    for i in range(objNum):
        fNorm[:, i] = (fNorm[:, i]-np.min(fNorm[:, i])) / (np.max(fNorm[:, i])-np.min(fNorm[:, i]))
    _, consNum = CV.shape
    if(popSize != _):
        raise RuntimeError('error: CV is illegal. (违反约束程度矩阵CV的数据格式不合法，请检查CV的计算。)')
    CVNorm = CV.copy()
    CVNorm[CVNorm < 0] = 0
    for j in range(consNum):
        if(np.max(CVNorm[:, j]) > 0):
            CVNorm[:, j] = CVNorm[:, j] / np.max(CVNorm[:, j])
    v = np.sum(CVNorm, axis=1) / consNum
    rf = np.sum(v==0) / popSize
    # print("CV_best: %lf, rf: %lf" % (np.min(v), rf))
    
    d = np.zeros([popSize, objNum])
    if(rf == 0):
        for p in range(popSize):
            d[p, :] = v[p]
    else:
        for p in range(popSize):
            d[p, :] = np.sqrt(fNorm[p, :]**2+v[p]**2)
    
    X = np.zeros([popSize, objNum])
    Y = np.copy(fNorm)
    if(rf != 0):
        for p in range(popSize):
            X[p, :] = v[p]
    Y[v==0, :] = 0
    p = (1-rf)*X + rf*Y
    
    # print("F: ", d+p)
    return d+p 

# 问题类参考格式
class MyProblem(ea.Problem):  # 继承Problem父类
	# pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=None, Dim=None):  # M : 目标维数;Dim : 决策变量维数
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 2  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [5, 3]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)
        self.punFlag = punFlag

    def evalVars(self, Vars):  # 目标函数
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        f1 = 4 * x1**2 + 4 * x2**2
        f2 = (x1 - 5)**2 + (x2 - 5)**2
        f = np.hstack([f1, f2])
        CV = np.hstack([(x1 - 5)**2 + x2**2 - 25,
                        -(x1 - 8)**2 - (x2 + 3)**2 + 7.7])  
        
        if(self.punFlag == 1):
	        f = getFPunish(f, CV)          
        
        return f, CV

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = 10000  # 欲得到10000个真实前沿点
        x1 = np.linspace(0, 5, N)
        x2 = x1.copy()
        x2[x1 >= 3] = 3
        return np.vstack((4 * x1**2 + 4 * x2**2, (x1 - 5)**2 + (x2 - 5)**2)).T