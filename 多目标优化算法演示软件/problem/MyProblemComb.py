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

#10个组合优化问题
class MP1(ea.Problem):  # 继承Problem父类
    """
    min f1 = -25 * (x1 - 2)**2 - (x2 - 2)**2 - (x3 - 1)**2 - (x4 - 4)**2 - (x5 - 1)**2
    min f2 = (x1 - 1)**2 + (x2 - 1)**2 + (x3 - 1)**2 + (x4 - 1)**2 + (x5 - 1)**2
    s.t.
    x1 + x2 >= 2
    x1 + x2 <= 6
    x1 - x2 >= -2
    x1 - 3*x2 <= 2
    4 - (x3 - 3)**2 - x4 >= 0
    (x5 - 3)**2 + x4 - 4 >= 0
    x1,x2,x3,x4,x5 ∈ {0,1,2,3,4,5,6,7,8,9,10}
    """
    def __init__(self, punFlag=0, M=2):
        name = 'MP1'  # 初始化name（函数名称，可以随意设置）
        Dim = 5  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [10] * Dim  # 决策变量上界
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
        x3 = Vars[:, [2]]
        x4 = Vars[:, [3]]
        x5 = Vars[:, [4]]
        f1 = -25 * (x1 - 2)**2 - (x2 - 2)**2 - (x3 - 1)**2 - (x4 - 4)**2 - (x5 - 1)**2
        f2 = (x1 - 1)**2 + (x2 - 1)**2 + (x3 - 1)**2 + (x4 - 1)**2 + (x5 - 1)**2
        f = np.hstack([f1, f2])
        CV = np.hstack([
                2 - x1 - x2,
                x1 + x2 - 6,
                -2 - x1 + x2,
                x1 - 3 * x2 - 2, (x3 - 3)**2 + x4 - 4,
                4 - (x5 - 3)**2 - x4
            ])

        if(self.punFlag == 1):
           f = getFPunish(f, CV)
        
        return f, CV

class KNAPSACK1(ea.Problem):  # 继承Problem父类
    """一个带约束的多目标背包问题.

        假设有5类物品，每类物品中包含着四个具体的物品，要求从这五种类别的物品中分别选择一个物品放进背包，
    使背包内的物品总价最高，总体积最小，且背包的总质量不能超过92kg。用矩阵P代表背包中物品的价值；
    矩阵R代表背包中物品的体积；矩阵C代表物品的质量。P,R,C的取值如下:
    P=[[3,4,9,15,2],      R=[[0.2, 0.3, 0.4, 0.6, 0.1],     C=[[10,13,24,32,4],
       [4,6,8,10,2.5],       [0.25,0.35,0.38,0.45,0.15],       [12,15,22,26,5.2],
       [5,7,10,12,3],        [0.3, 0.37,0.5, 0.5, 0.2],        [14,18,25,28,6.8],
       [3,5,10,10,2]]        [0.3, 0.32,0.45,0.6, 0.2]]        [14,14,28,32,6.8]]
    分析：
        这是一个0-1背包问题，但如果用一个元素为0或1的矩阵来表示哪些物品被选中，则不利于后面采用进
    化算法进行求解。可以从另一种角度对背包问题进行编码：由于问题要求在每类物品均要选出一件，这里我
    们可以用0, 1, 2, 3来表示具体选择哪件物品。因此染色体可以编码为一个元素属于{0, 1, 2, 3}的1x5Numpy ndarray一维数组，
    比如：[0,0,0,0,0]表示从这五类物品中各选取第一个物品。
    """
    # pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=2):
        name = 'KNAPSACK1'  # 初始化name（函数名称，可以随意设置）
        maxormins = [-1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 5  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0] * Dim  # 决策变量下界
        ub = [3] * Dim  # 决策变量上界
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
        # 添加几个属性来存储P、R、C
        self.P = np.array([[3, 4, 9, 15, 2], [4, 6, 8, 10, 2.5],
                           [5, 7, 10, 12, 3], [3, 5, 10, 10, 2]])
        self.R = np.array([[0.2, 0.3, 0.4, 0.6,
                            0.1], [0.25, 0.35, 0.38, 0.45,
                                   0.15], [0.3, 0.37, 0.5, 0.5, 0.2],
                           [0.3, 0.32, 0.45, 0.6, 0.2]])
        self.C = np.array([[10, 13, 24, 32, 4], [12, 15, 22, 26, 5.2],
                           [14, 18, 25, 28, 6.8], [14, 14, 28, 32, 6.8]])
        self.punFlag = punFlag

    def evalVars(self, Vars):  # 目标函数
        x = Vars.astype(int)
        f1 = np.sum(self.P[x, [0, 1, 2, 3, 4]], 1)
        f2 = np.sum(self.R[x, [0, 1, 2, 3, 4]], 1)
        f = np.vstack([f1, f2]).T
        CV = np.array([np.sum(self.C[x, [0, 1, 2, 3, 4]], 1)]).T - 92
        
        if(self.punFlag == 1):
            f = getFPunish(f, CV)
        
        return f, CV

class KNAPSACK2(ea.Problem):  # 继承Problem父类
    """这是另一个带约束的多目标背包问题.

        假设有6类物品，每类物品中包含着5个具体的物品，要求从这6种类别的物品中分别选择一个物品放进背包，
    使背包内的物品总价最高，总体积最小，且背包的总质量不能超过116kg。用矩阵P代表背包中物品的价值；
    矩阵R代表背包中物品的体积；矩阵C代表物品的质量。P,R,C的取值如下:
    P=[[3,4,9,15,2,6],      R=[[0.2, 0.3, 0.4, 0.6, 0.1, 0.35],     C=[[10,13,24,32,4,24],
       [4,6,8,10,2.5,7],       [0.25,0.35,0.38,0.45,0.15,0.38],       [12,15,22,26,5.2,23],
       [5,7,10,12,3,6.5],      [0.3, 0.37,0.5, 0.5, 0.2,0.32],        [14,18,25,28,6.8,25],
       [3,5,10,10,2,7]         [0.3, 0.32,0.45,0.6, 0.2,0.4]          [14,14,28,32,6.8,26]
       [4,8,7,13,3.5,6]]       [0.2,0.4,0.35,0.55,0.25,0.36]]         [13,19,20,35,7,22]] 
                       
    分析：
        这是一个0-1背包问题，但如果用一个元素为0或1的矩阵来表示哪些物品被选中，则不利于后面采用进
    化算法进行求解。可以从另一种角度对背包问题进行编码：由于问题要求在每类物品均要选出一件，这里我
    们可以用0, 1, 2, 3,4来表示具体选择哪件物品。因此染色体可以编码为一个元素属于{0, 1, 2, 3, 4}的1x5Numpy ndarray一维数组，
    比如：[0,0,0,0,0,0]表示从这6类物品中各选取第一个物品。
    """
    # pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=2):
        name = 'KNAPSACK2'  # 初始化name（函数名称，可以随意设置）
        maxormins = [-1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 6  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0] * Dim  # 决策变量下界
        ub = [4] * Dim  # 决策变量上界
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
        # 添加几个属性来存储P、R、C
        self.P = np.array([[3, 4, 9, 15, 2, 6], [4, 6, 8, 10, 2.5, 7],
                           [5, 7, 10, 12, 3, 6.5], [3, 5, 10, 10, 2, 7], [4, 8, 7, 13, 3.5, 6]])
        self.R = np.array([[0.2, 0.3, 0.4, 0.6, 0.1, 0.35], [0.25, 0.35, 0.38, 0.45, 0.15, 0.38],
                           [0.3, 0.37, 0.5, 0.5, 0.2, 0.32], [0.3, 0.32, 0.45, 0.6, 0.2, 0.4],
                           [0.2, 0.4, 0.35, 0.55, 0.25, 0.36]])
        self.C = np.array([[10, 13, 24, 32, 4, 24], [12, 15, 22, 26, 5.2, 23],
                           [14, 18, 25, 28, 6.8, 25], [14, 14, 28, 32, 6.8, 26],
                           [13, 19, 20, 35, 7, 22]])
        self.punFlag = punFlag

    def evalVars(self, Vars):  # 目标函数
        x = Vars.astype(int)
        f1 = np.sum(self.P[x, [0, 1, 2, 3, 4, 5]], 1)
        f2 = np.sum(self.R[x, [0, 1, 2, 3, 4, 5]], 1)
        f = np.vstack([f1, f2]).T
        # 采用可行性法则处理约束
        CV = np.array([np.sum(self.C[x, [0, 1, 2, 3, 4, 5]], 1)]).T - 116
        
        if(self.punFlag == 1):
            f = getFPunish(f, CV)  
        
        return f, CV

class ZDT5(ea.Problem):  # 继承Problem父类
    def __init__(self, punFlag=0, M=None, Dim=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'ZDT5'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 11  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [30] + [5] * (Dim - 1)  # 决策变量上界
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

    def evalVars(self, Vars):  # 目标函数
        v = np.zeros(Vars.shape)
        v[np.where(Vars < 5)] += 2
        v[np.where(Vars == 5)] = 1
        ObjV1 = 1 + Vars[:, 0]
        g = np.sum(v[:, 1:], 1)
        h = 1 / ObjV1
        ObjV2 = g * h
        f = np.array([ObjV1, ObjV2]).T
        return f

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        ObjV1 = np.arange(1, 32)
        ObjV2 = (self.Dim + 39) / 5 / ObjV1
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

class MOTSP(ea.Problem):  # 继承Problem父类
    # http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/
    def __init__(self, data1, data2, punFlag=0):  # testName为测试集名称
        name = 'MOTSP'  # 初始化name
        # 读取城市坐标数据
        self.places1 = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + "\\TSPdata\\" + data1 + ".tsp")
        self.places2 = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + "\\TSPdata\\" + data2 + ".tsp")
        self.places1 = self.places1[:, 1:]
        self.places2 = self.places2[:, 1:]
        # print(self.places1)
        # print(self.places2)
        M = 2  # 初始化M（目标维数）
        Dim = self.places1.shape[0]  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [Dim - 1] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, x):  # 目标函数
        N = x.shape[0]
        # 添加最后回到出发地
        X = np.hstack([x, x[:, [0]]]).astype(int)
        f1 = []
        f2 = []
        for i in range(N):
            journey1 = self.places1[X[i], :]  # 目标1按既定顺序到达的地点坐标
            journey2 = self.places2[X[i], :]  # 目标2按既定顺序到达的地点坐标
            f1.append(np.sum(np.sqrt(np.sum(np.diff(journey1.T) ** 2, 0))))
            f2.append(np.sum(np.sqrt(np.sum(np.diff(journey2.T) ** 2, 0))))
        
        f = np.vstack([f1, f2]).T
        return f