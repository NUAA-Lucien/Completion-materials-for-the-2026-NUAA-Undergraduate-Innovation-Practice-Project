import yaml
from torch import load
from NN2 import test_model, MyNet
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class NSGA_II:

    def __init__(self, generation=50, population=50):
        self.num_generation = generation
        self.num_population = population
        self.num_target = 4
        self.num_gene = 4
        self.num_chromosome = 24

        self.individuals_p = []
        self.individuals_q = []
        self.fronts = []
        self.data= []
        self.best_individual = Individual(8,8,8,8)

    def main(self):
        """算法主程序"""
        self.__init_generation()
        for i in range(self.num_generation-1):
            print(f'Generation {i+1}')
            self.__fast_non_dominated_sort()
            self.__next_generation_p()
            self.__next_generation_q()
        print(f'Generation {self.num_generation}')
        self.__fast_non_dominated_sort()
        self.save_file()

    def __fast_non_dominated_sort(self):
        """快速非支配排序"""
        self.individuals_q.extend(self.individuals_p)
        for individual in self.individuals_q:
            individual.rank = 0
            individual.slaves = []
            individual.masters_num = 0
        self.fronts = []
        front = []
        i = 0
        while i<(2*self.num_population):
            weight_i, mises_i, stiffy_i, stiffz_i = self.individuals_q[i].result
            if stiffy_i>1e8 and stiffz_i>1e8 and weight_i<=self.best_individual.result[0]:
                self.best_individual = Individual(chromosome=self.individuals_q[i].chromosome)
            j = i+1
            while j<2*self.num_population:
                weight_j, mises_j, stiffy_j, stiffz_j = self.individuals_q[j].result
                if stiffy_i>0.9e8 and stiffz_i>0.9e8 and stiffz_j>0.9e8 and stiffz_j>0.9e8:
                    rate1 = 1.00
                    rate2 = 1/rate1
                else:
                    rate1 = 1.03
                    rate2 = 1/rate1
                if (weight_i<rate1*weight_j) and (mises_i<mises_j) and (stiffy_i>stiffy_j) and (stiffz_i>stiffz_j):
                    self.individuals_q[i].slaves.append(self.individuals_q[j])
                    self.individuals_q[j].masters_num += 1
                elif (weight_i>rate2*weight_j) and (mises_i>mises_j) and (stiffy_i<stiffy_j) and (stiffz_i<stiffz_j):
                    self.individuals_q[i].masters_num += 1
                    self.individuals_q[j].slaves.append(self.individuals_q[i])
                j += 1
            if self.individuals_q[i].masters_num==0:
                self.individuals_q[i].rank = 1
                front.append(self.individuals_q[i])
            i += 1
        self.fronts.append(front)
        rank = 1
        while self.fronts[-1]:
            front = []
            for individual_i in self.fronts[-1]:
                for individual_j in individual_i.slaves:
                    individual_j.masters_num -= 1
                    if individual_j.masters_num==0:
                        individual_j.rank = rank+1
                        front.append(individual_j)
            rank += 1
            self.fronts.append(front)
        del self.fronts[-1]
        print(len(self.fronts))
        # 记录数据
        generation_data = []
        for front in self.fronts:
            front_data = []
            for individual in front:
                front_data.append([*individual.paras, *individual.result])
            generation_data.append(front_data)
        self.data.append(generation_data)

    def __crowding_distance_assign(self, front):
        """拥挤度计算"""
        individuals = front[:]
        num_individual = len(front)
        for individual in individuals:
            individual.distance = 0
        for target in range(self.num_target):
            self.__sort_insert(individuals, target)
            individuals[0].distance = 1e99
            individuals[-1].distance = 1e99
            target_min = individuals[0].result[target]
            target_max = individuals[-1].result[target]
            target_dta = target_max-target_min
            if target_dta==0:
                continue
            for i in range(1, num_individual-1):
                individuals[i].distance += (individuals[i+1].result[target]-individuals[i-1].result[target])/target_dta

    def __sort_insert(self, individuals, target):
        """插入排序"""
        num_individual = len(individuals)
        for i in range(num_individual-1):
            value_min = 1e100
            index_min = None
            for j in range(i, num_individual):
                if target<0:
                    value = individuals[j].distance
                else:
                    value = individuals[j].result[target]
                if value<value_min:
                    value_min = value
                    index_min = j
            temp = individuals[i]
            individuals[i] = individuals[index_min]
            individuals[index_min] = temp

    def __crowded_comparison_operator(self, i, j):
        """优劣算子"""
        if (i.rank<j.rank) or ((i.rank==j.rank) and (i.distance<j.distance)):
            True
        else:
            False

    def __init_generation(self):
        """初始化种群"""
        for i in range(2*self.num_population):
            self.individuals_q.append(Individual(randomc=True))

    def __next_generation_p(self):
        """下一代父代"""
        self.individuals_p = [self.best_individual]
        i = 0
        while (len(self.individuals_p)+len(self.fronts[i]))<=self.num_population:
            self.__crowding_distance_assign(self.fronts[i])
            self.individuals_p.extend(self.fronts[i])
            i += 1
        self.__crowding_distance_assign(self.fronts[i])
        self.__sort_insert(self.fronts[i], -1)
        self.individuals_p.extend(self.fronts[i][:self.num_population-len(self.individuals_p)])

    def __next_generation_q(self):
        """下一代子代"""
        self.individuals_q = []
        for i in range(self.num_population):
            # 选择
            samples = random.sample(self.individuals_p, int(0.2*self.num_population))
            num_samples = len(samples)
            best_f, best_m = samples[0], samples[1]
            for j in range(2, len(samples)):
                if self.__crowded_comparison_operator(samples[j], best_f):
                    best_m = best_f
                    best_f = samples[j]
            # 交叉
            point = random.randint(1, self.num_chromosome-2)
            new_chromosome = best_f.chromosome[:point]+best_m.chromosome[point:]
            # 变异
            point = random.randint(0, self.num_chromosome-1)
            new_chromosome[point] = 1-new_chromosome[point]
            self.individuals_q.append(Individual(chromosome=new_chromosome))

    def save_file(self):
        file = open('data.txt','w')
        i = 1
        for generation_data in self.data:
            file.write(f'Generation {i}:\n')
            i += 1
            for front_data in generation_data:
                for individual_data in front_data:
                    for value in individual_data:
                        file.write(f'{value}\t')
                    file.write('\n')
                file.write('\n')
            file.write('\n')
        file.close()


class Individual:

    __model = MyNet()
    __model.load_state_dict(load('model2.pt'))

    __genes_attr = None
    __gene_num = 0
    __chromosome_size = 0

    def __init__(self, *paras, chromosome=None, randomc=False):
        if randomc:
            chromosome = []
            for bit in range(self.__chromosome_size):
                chromosome.append(random.randint(0, 1))
        if chromosome is not None:
            self.paras = self.__decode(chromosome)
            self.chromosome = chromosome
        elif paras:
            self.paras = paras
            self.chromosome = self.__encode(paras)
        else:
            raise Exception('Individual instantiated without parameters!!!')
        self.result = test_model(self.__model, self.paras)

        # 快速非支配排序参数
        self.rank = 0
        self.slaves = []
        self.masters_num = 0

        # 拥挤度参数
        self.distance = 0

    def __repr__(self):
        return str(self.paras)

    @classmethod
    def __encode(cls, paras):
        """二进制编码"""
        chromosome = []
        for i in range(cls.__gene_num):
            start, step, digit = cls.__genes_attr[i]
            gene_i = format(int((paras[i]-start)/step),'0>%sb'% digit)
            for bit in gene_i:
                chromosome.append(int(bit))
        return chromosome

    @classmethod
    def __decode(cls, chromosome):
        """二进制解码"""
        paras = []
        for i in range(cls.__gene_num):
            start, step, digit = cls.__genes_attr[i]
            gene_i = chromosome[:digit]
            chromosome = chromosome[digit:]
            num = 0
            for power, bit in enumerate(gene_i[::-1]):
                num += bit*2**power
            para = start+num*step
            paras.append(round(para, 1))
        return paras

    @classmethod
    def set_attr(cls, attrs):
        """设置属性"""
        cls.__gene_num = len(attrs)
        cls.__genes_attr = []
        for start, end, step in attrs:
            digit = len(format(int((end-start)/step), 'b'))
            cls.__genes_attr.append([start, step, digit])
            cls.__chromosome_size += digit


class DataProcessing:

    def __init__(self, file_name='data.txt'):
        self.data = []
        file = open(file_name, 'r')
        mark = 0
        generation_data = []
        while True:
            line = file.readline().split()
            if line==[]:
                if front_data:
                    generation_data.append(front_data)
                elif mark==0:
                    if generation_data:
                        self.data.append(generation_data)
                    mark = 1
                else:
                    break
                front_data = []
                continue
            if line[0]=='Generation':
                mark = 0
                generation_data = []
                front_data = []
                continue
            individual_data = []
            for datum in line:
                individual_data.append(float(datum))
            front_data.append(individual_data)
        file.close()
        self.num_generation = len(self.data)

    def plot_data(self, x_num=6, y_num=7, z_num=None):
        # 数据
        x1, y1, z1 = [], [], []
        x2, y2, z2 = [], [], []
        for individual_data in self.data[0][0]:
            x1.append(individual_data[x_num])
            y1.append(individual_data[y_num])
            z1.append(individual_data[z_num])
        for individual_data in self.data[-1][0]:
            x2.append(individual_data[x_num])
            y2.append(individual_data[y_num])
            z2.append(individual_data[z_num])
        # 画图
        data_labels = {
            4: '质量（t）',
            5: '最大应力（MPa）',
            6: 'Y方向刚度(N/m)',
            7: 'Z方向刚度(N/m)'
        }
        plt.figure()
        plt.rcParams['font.family'] = ['SimSun']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.unicode_minus'] = False
        if z_num:
            ax = plt.axes(projection='3d')
            ax.scatter3D(x1, y1, z1, label='第一代非支配解集')
            ax.scatter3D(x2, y2, z2, label='最终代非支配解集')
            ax.set_xlabel(data_labels[x_num])
            ax.set_ylabel(data_labels[y_num])
            ax.set_zlabel(data_labels[z_num])
        else:
            plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)
            plt.scatter(x1, y1, label='第一代非支配解集')
            plt.scatter(x2, y2, label='最终代非支配解集')
            plt.xlabel(data_labels[x_num])
            plt.ylabel(data_labels[y_num])
        plt.legend()
        plt.show()

    def select_best(self):
        best_i = None
        best_individual = [0,0,0,0,1,0,0,0]
        for i in range(self.num_generation):
            for individual_data in self.data[i][0]:
                if individual_data[6]>1e8 and individual_data[7]>1e8 and individual_data[4]<=best_individual[4]:
                    best_individual = individual_data[:]
                    best_i = i
        print(best_individual)


if __name__ == '__main__':
    # # 优化
    # genes_attr = [[3.0, 9.3, 0.1], [3.0, 9.3, 0.1], [4.0, 10.3, 0.1], [2.0, 8.3, 0.1], ]
    # Individual.set_attr(genes_attr)
    # # item1 = Individual(randomc=True)
    # # print(item1.paras, *item1.result)
    #
    # optimization = NSGA_II()
    # optimization.main()

    # 整理
    datashow = DataProcessing('data.txt')
    datashow.select_best()
    datashow.plot_data(6,7,4)
