import numpy as np
import geatpy as ea

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

#10个连续优化问题
class BNH(ea.Problem):  # 继承Problem父类
	# pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=None, Dim=None):  # M : 目标维数;Dim : 决策变量维数
        name = 'BNH'  # 初始化name（函数名称，可以随意设置）
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

class OSY(ea.Problem):  # 继承Problem父类
	# pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=None, Dim=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'OSY'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 6  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0, 0, 1, 0, 1, 0]  # 决策变量下界
        ub = [10, 10, 5, 6, 5, 10]  # 决策变量上界
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

    def evalVars(self, X):  # 目标函数
        x1 = X[:, [0]]
        x2 = X[:, [1]]
        x3 = X[:, [2]]
        x4 = X[:, [3]]
        x5 = X[:, [4]]
        x6 = X[:, [5]]
        f1 = -(25 * (x1 - 2)**2 + (x2 - 2)**2 + (x3 - 1)**2 + (x4 - 4)**2 +
               (x5 - 1)**2)
        f2 = np.sum(X**2, 1, keepdims=True)
        f = np.hstack([f1, f2])
        CV = np.hstack([-(x1+x2-2), -(6-x1-x2), -(2-x2+x1),
                        -(2-x1+3*x2), -(4-(x3-3)**2-x4), -((x5-3)**2+x6-4)])

        if(self.punFlag == 1):
	        f = getFPunish(f, CV)        
        
        return f, CV

class SRN(ea.Problem):  # 继承Problem父类
	# pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=None, Dim=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'SRN'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 2  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [-20] * Dim  # 决策变量下界
        ub = [20] * Dim  # 决策变量上界
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
        f1 = 2 + (x1 - 2)**2 + (x2 - 1)**2
        f2 = 9 * x1 - (x2 - 1)**2
        f = np.hstack([f1, f2])
        CV = np.hstack([x1**2 + x2**2 - 225, x1 - 3 * x2 + 10])

        if(self.punFlag == 1):
            f = getFPunish(f, CV)
        
        return f, CV

class CON(ea.Problem):  # 继承Problem父类
	# pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=None, Dim=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'CON'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 2  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0.1, 0]  # 决策变量下界
        ub = [1, 5]  # 决策变量上界
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
        f1 = x1
        f2 = (1 + x2) / x1
        f = np.hstack([f1, f2])
        CV = np.hstack([6 - x2 - 9 * x1, 1 + x2 - 9 * x1])

        if(self.punFlag == 1):
	        f = getFPunish(f, CV)  
        
        return f, CV

class ZDT1(ea.Problem):  # 继承Problem父类

    def __init__(self, punFlag=0, M=None, Dim=30):  # M : 目标维数；Dim : 决策变量维数
        name = 'ZDT1'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
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
        ObjV1 = Vars[:, 0]
        gx = 1 + 9 * np.sum(Vars[:, 1:], 1) / (self.Dim - 1)
        hx = 1 - np.sqrt(np.abs(Vars[:, 0] / gx))  # 取绝对值是为了避免浮点数精度异常带来的影响
        ObjV2 = gx * hx
        f = np.array([ObjV1, ObjV2]).T
        return f

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = 10000  # 生成10000个参考点
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

class ZDT2(ea.Problem):  # 继承Problem父类

    def __init__(self, punFlag=0, M=None, Dim=30):  # M : 目标维数；Dim : 决策变量维数
        name = 'ZDT2'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
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
        ObjV1 = Vars[:, 0]
        gx = 1 + 9 * np.sum(Vars[:, 1:], 1) / (self.Dim - 1)
        hx = 1 - (ObjV1 / gx)**2
        ObjV2 = gx * hx
        f = np.array([ObjV1, ObjV2]).T
        return f

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = 10000  # 生成10000个参考点
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1**2
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

class ZDT3(ea.Problem):  # 继承Problem父类

    def __init__(self, punFlag=0, M=None, Dim=30):  # M : 目标维数；Dim : 决策变量维数
        name = 'ZDT3'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
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
        ObjV1 = Vars[:, 0]
        gx = 1 + 9 * np.sum(Vars[:, 1:], 1) / (self.Dim - 1)
        hx = 1 - np.sqrt(np.abs(ObjV1 / gx)) - (ObjV1 / gx) * np.sin(
            10 * np.pi * ObjV1)  # 取绝对值是为了避免浮点数精度异常带来的影响
        ObjV2 = gx * hx
        f = np.array([ObjV1, ObjV2]).T
        return f

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = 10000  # 生成10000个参考点
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1**0.5 - ObjV1 * np.sin(10 * np.pi * ObjV1)
        f = np.array([ObjV1, ObjV2]).T
        levels, criLevel = ea.ndsortESS(f, None, 1)
        referenceObjV = f[np.where(levels == 1)[0]]
        return referenceObjV

class ZDT4(ea.Problem):  # 继承Problem父类

    def __init__(self, punFlag=0, M=None, Dim=10):  # M : 目标维数；Dim : 决策变量维数
        name = 'ZDT4'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] + [-5] * (Dim - 1)  # 决策变量下界
        ub = [1] + [5] * (Dim - 1)  # 决策变量上界
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
        ObjV1 = Vars[:, 0]
        Vars1_10 = Vars[:, 1:Vars.shape[1]]
        gx = 1 + 10 * (self.Dim - 1) + np.sum(
            Vars1_10**2 - 10 * np.cos(4 * np.pi * Vars1_10), 1)
        hx = 1 - np.sqrt(np.abs(ObjV1) / gx)  # 取绝对值是为了避免浮点数精度异常带来的影响
        ObjV2 = gx * hx
        f = np.array([ObjV1, ObjV2]).T
        return f

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = 10000  # 生成10000个参考点
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

class ZDT6(ea.Problem):  # 继承Problem父类

    def __init__(self, punFlag=0, M=None, Dim=10):  # M : 目标维数；Dim : 决策变量维数
        name = 'ZDT6'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
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
        ObjV1 = 1 - np.exp(-4 * Vars[:, 0]) * (np.sin(
            6 * np.pi * Vars[:, 0]))**6
        gx = 1 + 9 * np.abs(np.sum(Vars[:, 1:10], 1) / (self.Dim - 1))**0.25
        hx = 1 - (ObjV1 / gx)**2
        ObjV2 = gx * hx
        f = np.array([ObjV1, ObjV2]).T
        return f

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = 10000  # 生成10000个参考点
        ObjV1 = np.linspace(0.280775, 1, N)
        ObjV2 = 1 - ObjV1**2
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

class DTLZ1(ea.Problem):  # 继承Problem父类

    def __init__(self, punFlag=0, M=3, Dim=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'DTLZ1'  # 初始化name（函数名称，可以随意设置）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        if Dim is None:
            Dim = M + 4  # 初始化Dim（决策变量维数）
        varTypes = np.array([0] * Dim)  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
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
        XM = Vars[:, (self.M - 1):]
        g = 100 * (self.Dim - self.M + 1 + np.sum(
            ((XM - 0.5)**2 - np.cos(20 * np.pi * (XM - 0.5))),
            1,
            keepdims=True))
        ones_metrix = np.ones((Vars.shape[0], 1))
        f = 0.5 * np.hstack([
            np.fliplr(np.cumprod(Vars[:, :self.M - 1], 1)), ones_metrix
        ]) * np.hstack([ones_metrix, 1 - Vars[:, range(self.M - 2, -1, -1)]
                        ]) * (1 + g)
        return f

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        uniformPoint, ans = ea.crtup(self.M, 10000)  # 生成10000个在各目标的单位维度上均匀分布的参考点
        referenceObjV = uniformPoint / 2
        return referenceObjV