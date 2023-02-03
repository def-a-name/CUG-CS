import numpy as np
import geatpy as ea  # 导入geatpy库
import sys
from algorithm.my_moeaalgorithm import my_MoeaAlgorithm


class my_NSGA2(my_MoeaAlgorithm):

    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 dirName=None,
                 **kwargs):
        # 先调用父类构造方法
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing, dirName)
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'my_NSGA2'
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  # 采用ENS_SS进行非支配排序
        else:
            self.ndSort = ea.ndsortTNS  # 高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS要快
        self.selFunc = 'urs'  # 选择方式，采用锦标赛选择
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=1)  # 生成部分匹配交叉算子对象
            self.mutOper = ea.Mutinv(Pm=1)  # 生成逆转变异算子对象
        elif population.Encoding == 'BG':
            self.recOper = ea.Xovud(XOVR=1)  # 生成均匀交叉算子对象
            self.mutOper = ea.Mutbin(Pm=None)  # 生成二进制变异算子对象，Pm设置为None时，具体数值取变异算子中Pm的默认值
        elif population.Encoding == 'RI':
            self.recOper = ea.Recsbx(XOVR=1, n=20)  # 生成模拟二进制交叉算子对象
            self.mutOper = ea.Mutpolyn(Pm=1 / self.problem.Dim, DisI=20)  # 生成多项式变异算子对象
        else:
            raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')
    
    def call_aimFunc(self, pop):
        """
        描述: 调用问题类的aimFunc()或evalVars()完成种群目标函数值和违反约束程度的计算。
        例如：population为一个种群对象，则调用call_aimFunc(population)即可完成目标函数值的计算。
             之后可通过population.ObjV得到求得的目标函数值，population.CV得到违反约束程度矩阵。

        """
        feasFlag = 1    # 记录当前种群有无可行解，1有，0无
        pop.Phen = pop.decoding()  # 染色体解码
        if self.problem is None:
            raise RuntimeError('error: problem has not been initialized. (算法类中的问题对象未被初始化。)')
        self.problem.evaluation(pop)  # 调用问题类的evaluation()

        # 若个体i所有目标函数值Fj(xi)都相等，说明F(xi)=v(xi)，个体i不是可行解
        infeasNum = np.sum(np.min(pop.ObjV, axis=1) == np.max(pop.ObjV, axis=1))
        # if(infeasNum == pop.sizes):
        if(pop.sizes - infeasNum <= 5):
            feasFlag = 0

        self.evalsNum = self.evalsNum + pop.sizes if self.evalsNum is not None else pop.sizes  # 更新评价次数
        # 格式检查
        if not isinstance(pop.ObjV, np.ndarray) or pop.ObjV.ndim != 2 or pop.ObjV.shape[0] != pop.sizes or \
                pop.ObjV.shape[1] != self.problem.M:
            raise RuntimeError('error: ObjV is illegal. (目标函数值矩阵ObjV的数据格式不合法，请检查目标函数的计算。)')
        if pop.CV is not None:
            if not isinstance(pop.CV, np.ndarray) or pop.CV.ndim != 2 or pop.CV.shape[0] != pop.sizes:
                raise RuntimeError('error: CV is illegal. (违反约束程度矩阵CV的数据格式不合法，请检查CV的计算。)')
        return feasFlag    

    def reinsertion(self, population, offspring, NUM):
        """
        描述:
            重插入个体产生新一代种群（采用父子合并选择的策略）;
            NUM为所需要保留到下一代的个体数目;
            重写了geatpy代码，只考虑临界层的排序
        
        """
        population = population + offspring

        [levels, _] = self.ndSort(population.ObjV, NUM, None, None, self.problem.maxormins)
        # population.FitnV = (1 / levels).reshape(-1, 1)
        # print(levels)
        # print(_)
        population_last = population[levels == _]
        levels_last = np.full(len(population_last), _)
        dis_last = ea.crowdis(population_last.ObjV, levels_last)
        # print(dis_last)
        # chooseFlag = ea.selecting('dup', population.FitnV, NUM)
        p1 = population[levels < _]
        lastNUM = NUM - len(p1)
        p2 = population_last[np.argsort(-dis_last)[:lastNUM]]
        return p1 + p2

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法类的一些动态参数
        
        # ===========================准备进化============================
        population.initChrom()  # 初始化种群染色体矩阵
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        anyFeasible = self.call_aimFunc(population)  # 计算种群的目标函数值
        # [levels, _] = self.ndSort(population.ObjV, NIND, None, population.CV, self.problem.maxormins)  # 对NIND个个体进行非支配分层
        # population.FitnV = (1 / levels).reshape(-1, 1)  # 直接根据levels来计算初代个体的适应度
        
        # ===========================开始进化============================
        while not self.terminated(population):
            # 选择个体参与进化
            offspring = population[ea.selecting(self.selFunc, NIND, NIND)]
            # 对选出的个体进行进化操作
            offspring.Chrom = self.recOper.do(offspring.Chrom)  # 重组
            # if self.currentGen > self.MAXGEN * 0.5:
            #     offspring.Chrom = ea.mutmani(offspring.Encoding, offspring.Chrom, offspring.Field, self.problem.M-1)
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
            anyFeasible = self.call_aimFunc(offspring)  # 求进化后个体的目标函数值
            # 只有出现可行解时，种群才进行重插入操作，否则种群直接被重组变异后的子代替换
            if(anyFeasible):
                population = self.reinsertion(population, offspring, NIND)
            else:
                population = offspring
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果