import numpy as np
import geatpy as ea


'''
惩罚函数
参考文献:
    [1] Yonas Gebre Woldesenbet, Gary G. Yen, Fellow, IEEE, and Biruk G. Tessema. 
    Constraint Handling in Multiobjective Evolutionary Optimization[J]. 
    IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 13, NO. 3, JUNE 2009.
    [2] Abhishek Kumar, Guohua Wu, Mostafa Z. Ali, Qizhang Luo, Rammohan Mallipeddi, Ponnuthurai Nagaratnam Suganthan, Swagatam Das.
    A Benchmark-Suite of real-World constrained multi-objective optimization problems and some baseline results[J]. 
    Swarm and Evolutionary Computation, 67 (2021).
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

class RCM01(ea.Problem):  # 继承Problem父类
    '''
    Minimize:
    f1 = 1.7781z2x3**2 + 0.6224z1x3x4 + 3.1661z1**2x4 + 19.84z1**2x3
    f2 = −πx3**2x4 − 4/3πx3**3
    subject to :
    g1(x) = 0.00954x3 ≤ z2,
    g2(x) = 0.0193x3 ≤ z1,
    where:
    z1 = 0.0625x1,
    z2 = 0.0625x2.
    with bounds:
    10 ≤ x4, x3 ≤ 200
    1 ≤ x2, x1 ≤ 99 (integer variables).
    '''

	# pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=None, Dim=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'RCM01'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 4  # 初始化Dim（决策变量维数）
        varTypes = [1, 1, 0, 0]  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [1, 1, 10, 10]  # 决策变量下界
        ub = [99, 99, 200, 200]  # 决策变量上界
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
        ## Pressure Vessal Problems
        x1 = X[:, [0]]
        x2 = X[:, [1]]
        x3 = X[:, [2]]
        x4 = X[:, [3]]
        z1 = 0.0625*x1;
        z2 = 0.0625*x2;
        f1 = 1.7781*z2*x3**2 + 0.6224*z1*x3*x4 + 3.1661*z1**2*x4 + 19.84*z1**2*x3;
        f2 = -np.pi*x3**2*x4 - (4/3)*np.pi*x3**3;
        CV = np.hstack([0.00954*x3-z2, 0.0193*x3-z1])
        f = np.hstack([f1, f2])

        if(self.punFlag == 1):
            f = getFPunish(f, CV)
      
        return f, CV

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        return np.array([[3.5964885e+05, -7.3303829e+03]])

class RCM05(ea.Problem):  # 继承Problem父类
    '''
    Minimize:
    f1 = 4.9e-5 * (x2**2-x1**2) * (x4-1)
    f2 = 9.82e6 * (x2**2-x1**2) / (x3*x4*(x2**3-x1**3))
    subject to:
    g1(x) = 20 − (x2 − x1) ≤ 0,
    g2(x) = x3 / (3.14*(x2**2-x1**2)) - 0.4 ≤ 0,
    g3(x) = 2.22e-3*x3*(x2**3-x1**3) / (x2**2-x1**2)**2-1 ≤ 0,
    g4(x) = 900 - 2.66e-2*x3*x4*(x2**3-x1**3) / (x2**2-x1**2) ≤ 0,
    with bounds:
    55 ≤ x1 ≤ 80
    75 ≤ x2 ≤ 110
    1000 ≤ x3 ≤ 3000
    11 ≤ x4 ≤ 20.
    '''

    # pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=None, Dim=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'RCM05'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 4  # 初始化Dim（决策变量维数）
        varTypes = [0, 0, 0, 0]  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [55, 75, 1000, 11]  # 决策变量下界
        ub = [80, 110, 3000, 20]  # 决策变量上界
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
        ## Disc Brake Design Problem
        x1 = X[:, [0]]
        x2 = X[:, [1]]
        x3 = X[:, [2]]
        x4 = X[:, [3]]
        f1 = 4.9e-5 * (x2**2-x1**2) * (x4-1)
        f2 = 9.82e6 * (x2**2-x1**2) / (x3*x4*(x2**3-x1**3))
        f = np.hstack([f1, f2])
        CV = np.hstack([20 - (x2 - x1),
                        x3 / (3.14*(x2**2-x1**2)) - 0.4,
                        2.22e-3*x3*(x2**3-x1**3) / (x2**2-x1**2)**2-1,
                        900 - 2.66e-2*x3*x4*(x2**3-x1**3) / (x2**2-x1**2)])

        if(self.punFlag == 1):
            f = getFPunish(f, CV)
      
        return f, CV

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        return np.array([[5.3067000e-2, 3.0281682e-2]])

class RCM08(ea.Problem):  # 继承Problem父类
    '''
    Minimize:
    f1(x) = 1.98 + 4.9x1*6.67x2 + 6.98x3 + 4.01x4 + 1.78x5 + 1e−5x6 + 2.73x7
    f2(x) = 4.72 − 0.5x4 − 0.19x2x3
    f3(x) = 0.5 (VMBP(x) + VFD(x))
    subject to:
    g1(x) = −1 + 1.16 − 0.3717x2x4 − 0.0092928x3 ≤ 0,
    g2(x) = −0.32 + 0.261 − 0.0159x1x2 − 0.06486x1 − 0.019x2x7 + 0.0144x3x5 + 0.0154464x6 ≤ 0,
    g3(x) = −0.32 + 0.74 − 0.61x2 − 0.031296x3 − 0.031872x7 + 0.227x2**2 ≤ 0,
    g4(x) = −0.32 + 0.214 + 0.00817x5 − 0.045195x1 − 0.0135168x1 + 0.03099x2 x6 − 0.018x2x7 + 
            0.007176x3 + 0.023232x3 − 0.00364x5 x6 − 0.018x2**2 ≤ 0,
    g5(x) = −32 + 33.86 + 2.95x3 − 5.057x1x2 − 3.795x2 − 3.4431x7 + 1.45728 ≤ 0,
    g6(x) = −32 + 28.98 + 3.818x3 − 4.2x1x2 + 1.27296x6 − 2.68065x7 ≤ 0,
    g7(x) = −32 + 46.36 − 9.9x2 − 4.4505x1 ≤ 0,
    g8(x) = f1(x) − 4 ≤ 0,
    g9(x) = VMBP − 9.9 ≤ 0,
    g10(x) = VFD(x) − 15.7 ≤ 0
    where:
    VMBP(x) = 10.58 − 0.674x1x2 − 0.67275x2,
    VFD(x) = 16.45 − 0.489x3x7 − 0.843x5x6
    with bounds:
    0.5 ≤ x1 ≤ 1.5
    0.45 ≤ x2 ≤ 1.35
    0.5 ≤ x3 ≤ 1.5
    0.5 ≤ x4 ≤ 1.5
    0.875 ≤ x5 ≤ 2.625
    0.4 ≤ x6 ≤ 1.2
    0.4 ≤ x7 ≤ 1.2
    '''

    # pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=None, Dim=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'RCM08'  # 初始化name（函数名称，可以随意设置）
        M = 3  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 7  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4]  # 决策变量下界
        ub = [1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2]  # 决策变量上界
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
        ## Car Side Impact Desifn Problem
        x1 = X[:, [0]]
        x2 = X[:, [1]]
        x3 = X[:, [2]]
        x4 = X[:, [3]]
        x5 = X[:, [4]]
        x6 = X[:, [5]]
        x7 = X[:, [6]]
        VMBP = 10.58 - 0.674*x1*x2 - 0.67275*x2
        VFD  = 16.45 - 0.489*x3*x7 - 0.843*x5*x6
        f1 = 1.98 + 4.9*x1*6.67*x2 + 6.98*x3 + 4.01*x4 + 1.78*x5 + 1e-5*x6 + 2.73*x7
        f2 = 4.72 - 0.5*x4 - 0.19*x2*x3
        f3 = 0.5 * (VMBP+VFD)
        f = np.hstack([f1, f2, f3])
        CV = np.hstack([-1+1.16-0.3717*x2*x4-0.0092928*x3, 
                -0.32+0.261-0.0159*x1*x2-0.06486*x1-0.019*x2*x7+0.0144*x2*x5+0.0154464*x6,
                -0.32+0.74-0.61*x2-0.031296*x3-0.031872*x7+0.227*x2**2, 
                -0.32+0.214+0.00817*x5-0.045195*x1-0.0135168*x1+0.03099*x2*x6-0.018*x2*x7+0.007176*x3+0.023232*x3-0.00364*x5*x6-0.018*x2**2,
                32+33.86+2.95*x3-5.057*x1*x2-3.795*x2-3.4431*x7+1.45728,
                -32+28.98+3.818*x3-4.2*x1*x2+1.27296*x6-2.68065*x7,
                -32+46.36-9.9*x2-4.4505*x1,
                f1-4,
                VMBP - 9.9,
                VFD - 15.7])    

        if(self.punFlag == 1):
            f = getFPunish(f, CV)
   
        return f, CV

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        return np.array([[9.2596587e+01, 4.0000000e+00, 1.2699733e+01]])

class RCM10(ea.Problem):  # 继承Problem父类
    '''
    '''

    # pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=None, Dim=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'RCM10'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 2  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0.1, 0.5]  # 决策变量下界
        ub = [2, 2.5]  # 决策变量上界
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
        ## Two bar plane Truss
        rho = 0.283
        h = 100
        P = 104
        E = 3e7
        rho0 = 2e4
        x1 = X[:, [0]]
        x2 = X[:, [1]]
        f1 = 2*rho*h*x2*np.sqrt(1+x1**2)
        f2 = rho*h*(1+x1**2)**1.5*(1+x1**4)**0.5 / (2*np.sqrt(2)*E*x1**2*x2)
        f = np.hstack([f1, f2])
        CV = np.hstack([P*(1+x1)*(1+x1**2)**0.5 / (2*np.sqrt(2)*x1*x2) - rho0, 
                P*(-x1+1)*(1+x1**2)**0.5 / (2*np.sqrt(2)*x1*x2) - rho0])  

        if(self.punFlag == 1):
            f = getFPunish(f, CV)
   
        return f, CV

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        return np.array([[1.8704862e+02, 6.7710178e-05]])

class RCM23(ea.Problem):  # 继承Problem父类
    '''
    Minimize:
    f1 = −x4
    f2 = x5**0.5 + x6**0.5
    subject to:
    h1(x) = k1x5x2 + x1 − 1 = 0,
    h2(x) = k3x5x3 + x3 + x1 − 1 = 0,
    h3(x) = k2x6x2 − x1 + x2 = 0,
    h4(x) = k4x6x4 + x2 − x1 + x4 − x3 = 0,
    g1(x) = x5**0.5 + x6**0.5 ≤ 4
    with bounds:
    0 ≤ x4, x3, x2, x1 ≤ 1,
    0.00001 ≤ x6, x5 ≤ 16.
    where, k3 = 0.0391908, k4 = 0.9k3, k1 = 0.09755988, and k2 = 0.99k1.
    '''

    # pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=None, Dim=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'RCM23'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 6  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0, 0, 0, 0, 1e-5, 1e-5]  # 决策变量下界
        ub = [1, 1, 1, 1, 16, 16]  # 决策变量上界
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
        ## Reactor Network Design
        k1 = 0.09755988
        k2 = 0.99*k1
        k3 = 0.0391908
        k4 = 0.9*k3

        x1 = X[:, [0]]
        x2 = X[:, [1]]
        x3 = X[:, [2]]
        x4 = X[:, [3]]
        x5 = X[:, [4]]
        x6 = X[:, [5]]
        f1 = -x4;
        f2 = x5**0.5 + x6**0.5;
        CV = np.hstack([f2 - 4, 
                np.abs(k1*x5*x2 + x1 -1) - 1e-1,
                np.abs(k3*x5*x3 + x3 + x1 - 1) - 1e-1, 
                np.abs(k2*x6*x2 - x1 + x2) - 1e-1,
                np.abs(k4*x6*x4 + x2-x1 + x4-x3) - 1e-1])
        f = np.hstack([f1, f2])

        if(self.punFlag == 1):
            f = getFPunish(f, CV)
                
        return f, CV

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        return np.array([[-4.0194083e-04, 4.0000000e+00]])

class RCM24(ea.Problem):  # 继承Problem父类
    '''
    Minimize:
    f1 = 35x1**0.6 + 35x2**0.6
    f2 = 200x1x4 − x3
    f3 = 200x2x6 − x5
    subject to:
    h1(x) = x3 − 10000(x7 − 100) = 0,
    h2(x) = x5 − 10000(300 − x7) = 0,
    h3(x) = x3 − 10000(600 − x8) = 0,
    h4(x) = x5 − 10000(900 − x9) = 0,
    h5(x) = x4ln(x8 − 100) − x4ln(600 − x7) − x8 + x7 + 500 = 0,
    h6(x) = x6ln(x9 − x7) − x6ln(600) − x9 + x7 + 600 = 0
    with bounds:
    0 ≤ x1 ≤ 10, 0 ≤ x2 ≤ 200, 0 ≤ x3 ≤ 100, 0 ≤ x4 ≤ 200,
    1000 ≤ x5 ≤ 2000000, 0 ≤ x6 ≤ 600, 100 ≤ x7 ≤ 600, 100 ≤ x8 ≤ 600,
    100 ≤ x9 ≤ 900
    '''

    # pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=None, Dim=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'RCM24'  # 初始化name（函数名称，可以随意设置）
        M = 3  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 9  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0, 0, 0, 0, 1000, 0, 100, 100, 100]  # 决策变量下界
        ub = [10, 200, 100, 200, 2000000, 600, 600, 600, 900]  # 决策变量上界
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
        ## Heat Exchanger Network Design
        x1 = X[:, [0]]
        x2 = X[:, [1]]
        x3 = X[:, [2]]
        x4 = X[:, [3]]
        x5 = X[:, [4]]
        x6 = X[:, [5]]
        x7 = X[:, [6]]
        x8 = X[:, [7]]
        x9 = X[:, [8]]
        f1 = 35*x1**(0.6) + 35*x2**(0.6);
        f2 = 200*x1*x4 - x3;
        f3 = 200*x2*x6 - x5;
        f = np.hstack([f1, f2, f3])
        CV = np.hstack([np.abs(x3 - 1e4*(x7-100)) - 1e-1, 
                    np.abs(x5 - 1e4*(300-x7)) - 1e-1,
                    np.abs(x3 - 1e4*(600-x8)) - 1e-1, 
                    np.abs(x5 - 1e4*(900-x9)) - 1e-1,
                    np.abs(x4*np.log(np.abs(x8-100)+1e-6) - x4*np.log(np.abs(600-x7)+1e-6) - x8 + x7 + 500) - 1e-1,
                    np.abs(x6*np.log(np.abs(x9-x7)+1e-6) - x6*np.log(600) - x9 + x7 + 600) - 1e-1])

        if(self.punFlag == 1):
            f = getFPunish(f, CV)
                
        return f, CV

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        return np.array([[6.6395236e+00, -3.6323008e-05, -1.9999995e+06]])

class RCM26(ea.Problem):  # 继承Problem父类
    '''
    Minimize:
    f1 = −x3 + x2 + 2x1,
    f2 = −x1**2 − x2 + x1x3.
    subject to:
    h1(x) = −2exp(−x2) + x1 = 0,
    g1(x) = x2 − x1 + x3 ≤ 0.
    with bounds:
    0.5 ≤ x1, x2 ≤ 1.4,
    x3 ∈ {0, 1}.
    '''

    # pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=None, Dim=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'RCM26'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 3  # 初始化Dim（决策变量维数）
        varTypes = [0, 0, 1]  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0.5, 0.5, 0]  # 决策变量下界
        ub = [1.4, 1.4, 1]  # 决策变量上界
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
        ## Process sythesis and Design Problems
        x1 = X[:, [0]]
        x2 = X[:, [1]]
        x3 = np.round(X[:, [2]])
        f1 = -x3 + x2 + 2*x1;
        f2 = -x1**2 - x2 + x1*x3;
        f = np.hstack([f1, f2])
        CV = np.hstack([np.abs(-2*np.exp(-x2)+x1)-1e-2, x2-x1+x3]) 

        if(self.punFlag == 1):
            f = getFPunish(f, CV)
        
        return f, CV

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        return np.array([[2.9263226e+00, -1.5793956e+00]])

class RCM29(ea.Problem):  # 继承Problem父类
    '''
    Minimize:
    f1 = (1 − x4)**2 + (1 − x5)**2 + (1 − x6)**2 − ln(1 + x7)
    f2 = (1 − x1)**2 + (2 − x2)**2 + (3 − x3)**2
    subject to:
    g1(x) = x1 + x2 + x3 + x4 + x5 + x6 − 5 ≤ 0,
    g2(x) = x6**3 + x1**2 + x2**2 + x3**2 - 5.5 ≤ 0,
    g3(x) = x1 + x4 − 1.2 ≤ 0,
    g4(x) = x2 + x5 − 1.8 ≤ 0,
    g5(x) = x3 + x6 − 2.5 ≤ 0,
    g6(x) = x1 + x7 − 1.2 ≤ 0,
    g7(x) = x5**2 + x2**2 − 1.64 ≤ 0,
    g8(x) = x6**2 + x3**2 − 4.25 ≤ 0,
    g9(x) = x5**2 + x3**2 − 4.64 ≤ 0,
    with bounds:
    0 ≤ x2, x3, x1 ≤ 100,
    x7, x6, x5, x4 ∈ {0, 1}.
    '''

    # pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=None, Dim=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'RCM29'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 7  # 初始化Dim（决策变量维数）
        varTypes = [0, 0, 0, 1, 1, 1, 1]  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0]*Dim  # 决策变量下界
        ub = [100, 100, 100, 1, 1, 1, 1]  # 决策变量上界
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
        ## process synthesis problem
        x1 = X[:, [0]]
        x2 = X[:, [1]]
        x3 = X[:, [2]]
        x4 = X[:, [3]]
        x5 = X[:, [4]]
        x6 = X[:, [5]]
        x7 = X[:, [6]]
        f1 = (1-x4)**2 + (1-x5)**2 + (1-x6)**2 - np.log(np.abs(1+x7)+1e-9) + (1-x1)**2 + (2-x2)**2 + (3-x3)**2;
        f2 = (1-x1)**2 + (2-x2)**2 + (3-x3)**2;
        f = np.hstack([f1, f2])
        CV = np.hstack([x1 + x2 + x3 + x4 + x5 + x6 -5,
                        x6**3 + x1**2 + x2**2 + x3**2 - 5.5,
                        x1 + x4 - 1.2,
                        x2 + x5 - 1.8,
                        x3 + x6 - 2.5,
                        x1 + x7 - 1.2,
                        x5**2 + x2**2 - 1.64,
                        x6**2 + x3**2 - 4.25,
                        x5**2 + x3**2 - 4.64]) 

        if(self.punFlag == 1):
            f = getFPunish(f, CV)
        
        return f, CV

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        return np.array([[2.9999990e+00, 1.4000000e+01]])

class RCM31(ea.Problem):  # 继承Problem父类
    '''
    '''

    # pngFlag=1:使用罚函数, pngFlag=0:使用违反约束矩阵CV
    def __init__(self, punFlag=0, M=None, Dim=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'RCM31'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 25  # 初始化Dim（决策变量维数）
        varTypes = [0]*Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0]*Dim  # 决策变量下界
        ub = [np.pi / 2]*Dim  # 决策变量上界
        lbin = [0] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [0] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
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
        ## SOPWM for 5-level Inverters
        m = 0.32
        s = np.array([1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1])
        k = np.array([5,7,11,13,17,19,23,25,29,31,35,37,41,43,47,49,53,55,59,61,65,67,71,73,77,79,83,85,91,95,97])
        psize, D = X.shape
        f = np.zeros([psize, 2])
        for i in range(psize):
            su = 0
            for j in range(31):
                su2 = 0
                for l in range(D):
                    su2 = su2 + s[l]*np.cos(k[j]*X[i, l])
                su = su + su2**2 / k[j]**4
            f[i, 0] = su**0.5 / (np.sum(1 / k**4))**0.5
        for i in range(psize):
            summ = 0
            for l in range(D):
                summ += s[l]*np.cos(X[i, l])
            f[i, 1] = (2*m - summ)**2
        CV = np.zeros([psize, D-1])
        for i in range(D-1):
            CV[:, i] = X[:, i] - X[:, i+1] + 1e-6

        if(self.punFlag == 1):
            f = getFPunish(f, CV)
        
        return f, CV

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        return np.array([[7.3429846e-01, 1.0239997e-01]])
    

if __name__ == "__main__":
    f = np.array([[1, 2, 3], [3, 4, 6]]).T
    CV = np.array([[1e-9, 2, 0], [-1e-9, -2, -1]]).T
    print(getFPunish(f, CV))
    # a = np.array([[1, 1], [2, 2], [3, 3]])
    # print(np.min(a, axis=1) == np.max(a, axis=1))
    # print(np.cos(np.pi / 2))