import math
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from ga_lib import myGA


class Problem1:
    def __init__(self, n_var, n_obj, n_ieq_constr, xl, xu, target, maximize):
        self.target = target
        self.n_var = n_var
        self.n_obj = n_obj
        if maximize:
            self.sign = -1
        else:
            self.sign = 1

    def evaluate(self, x, *args, **kwargs):
        score = self.sign*np.bitwise_not(np.bitwise_xor(x, self.target)).sum()
        return {'score': score, 'constraint': 0}


class Problem2:
    def __init__(self, n_var, n_obj, n_ieq_constr, xl, xu, target, maximize):
        self.target = target
        self.n_var = n_var
        self.n_obj = n_obj
        if maximize:
            self.sign = -1
        else:
            self.sign = 1

    def evaluate(self, x, *args, **kwargs):
        score = self.sign*np.bitwise_not(np.bitwise_xor(x, self.target)).sum()
        return {'score': score, 'constraint': 0}


class Problem3_Single_Object:
    def __init__(self, n_var, n_obj, n_ieq_constr, n_eq_constr, xl, xu):
        self.xl = xl
        self.xu = xu
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_ieq_constr = n_ieq_constr
        self.n_eq_constr = n_eq_constr

    def evaluate(self, x, *args, **kwargs):
        """Booth function """
        score = (x[0]+2*x[1]-7)**2+(2*x[0]+x[1]-5)**2
        return {'score': score, 'constraint': 0}


class Problem3_Constrained_Optimize:
    def __init__(self, n_var, n_obj, n_ieq_constr, n_eq_constr, xl, xu):
        self.xl = xl
        self.xu = xu
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_ieq_constr = n_ieq_constr
        self.n_eq_constr = n_eq_constr

    def evaluate(self, x, *args, **kwargs):
        """Rosenbrock function constrained to a disk"""
        score = (1-x[0])**2+100*(x[1]-x[0]**2)**2
        # out['F'] = [(1-x[0])**2+100*(x[1]-x[0]**2)**2 for x in X]
        # out['G'] = [2-x[0]**2-x[1]**2 for x in X]
        constraint = 2-x[0]**2-x[1]**2
        return {'score': score, 'constraint': constraint}


class Problem4:
    def __init__(self, n_var, n_obj, xl, xu, santa, claus, xmas, sqr=False):
        self.xl = xl
        self.xu = xu
        self.n_var = n_var
        self.n_obj = n_obj
        self.santa = santa
        self.claus = claus
        self.xmas = xmas
        self.sqr = sqr

    def evaluate(self, x, *args, **kwargs):
        num1 = int(''.join(map(str, x[self.santa])))
        num2 = int(''.join(map(str, x[self.claus])))
        num3 = int(''.join(map(str, x[self.xmas])))
        if self.sqr:
            score = (num1-num2-num3)**2
        else:
            score = abs(num1-num2-num3)
        return {'score': score, 'constraint': 0}


def run_problem1():
    ascii_art = np.array([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                          [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                          [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]]).astype(np.bool_)

    ascii_art = ascii_art.flatten()
    problem = Problem1(
        n_var=ascii_art.shape[0], n_obj=1, n_ieq_constr=0, xl=0, xu=1, target=ascii_art, maximize=False)

    alg = myGA(problem)
    best_ind = alg.evolve(
        n_pop=200,
        n_gen=200,
        n_off=100,
        sampling_method="binary_sampling",
        crossover_method="point_crossover",
        mutation_method="bit_flip_mutation",
        crossover_rate=0.7,
        mutation_rate=1/problem.n_var,
    )
    best_ind.gene = best_ind.gene.reshape(5, -1)

    def show_mat(x, keep=False):
        x = x.astype(np.uint8)
        fig, ax = plt.subplots()
        ax.matshow(x, cmap='gray')
        # plt.savefig(os.path.join(os.path.dirname(__file__),'figs/p1_2.png'), bbox_inches='tight')
        plt.show(block=keep)

    show_mat(best_ind.gene, keep=True)


def run_problem2():
    ascii_art = np.array([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                          [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                          [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]]).astype(np.bool_)

    ascii_art = ascii_art.flatten()
    problem = Problem2(
        n_var=ascii_art.shape[0], n_obj=1, n_ieq_constr=0, xl=0, xu=1, target=ascii_art, maximize=True)

    alg = myGA(problem)

    population_size = [50, 100]
    generations = [200]
    crossover_rate = [0.5, 0.7]
    mutation_rate = [0.1, 0.3]

    all_results = {}
    for pop_size in population_size:
        for gen in generations:
            for cr in crossover_rate:
                for mr in mutation_rate:
                    alg = myGA(problem)
                    best_ind = alg.evolve(
                        n_pop=pop_size,
                        n_gen=gen,
                        n_off=pop_size//2,
                        crossover_rate=cr,
                        mutation_rate=mr,
                        sampling_method="binary_sampling",
                        crossover_method="point_crossover",
                        mutation_method="bit_flip_mutation",
                    )
                    # TODO: do statistics
                    print(f"Best individual: {best_ind}")
                    print(f"Best gene: {best_ind.gene}")
                    print(f"Best fitness: {best_ind.score}")
                    print(f"hyper-parameters are: pop size={pop_size} gen={gen} cr={cr} mr={mr}")
                    all_results[f"pop-{pop_size}_gen-{gen}_cr-{cr}_mr-{mr}"] = alg.log
                    # fig, ax = plt.subplots()
                    # ax.matshow(best_ind.gene, cmap='gray')
                    # # plt.savefig(os.path.join(os.path.dirname(__file__),f'figs/{pop_size}_{gen}_{cr}_{mr}.png'), bbox_inches='tight')
                    # plt.show()
    statistics_names = ['min', 'max', 'mean', 'median', 'std' ]
    keys = list(all_results.keys())
    vals = np.array([all_results[key] for key in keys])
    vals = np.transpose(vals, axes=(2,0,1))
    for i in range(len(vals)):
        fig, ax = plt.subplots()
        for j,key in enumerate(keys):
            ax.plot(vals[i,j,:], label=key)
        ax.legend()
        ax.set_title(f'{statistics_names[i]} fitness value comparsion with different hyper-parameter configuration')
        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness')
        plt.savefig(os.path.join(os.path.dirname(__file__),f'figs/{statistics_names[i]}_200.png'), bbox_inches='tight')
        plt.show()

def run_problem3():
    problem_so = Problem3_Single_Object(
        n_var=2, n_obj=1, xl=-10, xu=10, n_ieq_constr=0, n_eq_constr=0)
    problem_co = Problem3_Constrained_Optimize(
        n_var=2, n_obj=1, xl=-1.5, xu=1.5, n_ieq_constr=1, n_eq_constr=0)
    # alg = myGA(problem_co)
    population_size = [100, 200]
    generations = [50]
    crossover_rate = [0.5, 0.7]
    mutation_rate = [0.1, 0.3]
    all_results = {}
    for gen in generations:
        for pop_size in population_size:
            for cr in crossover_rate:
                for mr in mutation_rate:
                    print(f'pop_size: {pop_size}, gen: {gen}, cr: {cr}, mr: {mr}')

                    alg = myGA(problem_so)
                    best_ind = alg.evolve(
                        n_pop=pop_size,
                        n_gen=gen,
                        n_off=pop_size//2,
                        sampling_method="float_sampling",
                        crossover_method="uniform_crossover",
                        mutation_method="gaussian_mutation",
                        crossover_rate=0.7,
                        mutation_rate=0.1
                    )
                    print('best solution:{}'.format(best_ind.gene))
                    # alg.plot()
                    all_results[f"pop-{pop_size}_gen-{gen}_cr-{cr}_mr-{mr}"] = alg.log
    statistics_names = ['min', 'max', 'mean', 'median', 'std' ]
    keys = list(all_results.keys())
    vals = np.array([all_results[key] for key in keys])
    vals = np.transpose(vals, axes=(2,0,1))
    for i in range(len(vals)):
        fig, ax = plt.subplots()
        for j,key in enumerate(keys):
            ax.plot(vals[i,j,:], label=key)
        ax.legend()
        ax.set_title(f'{statistics_names[i]} fitness value comparsion with different hyper-parameter configuration')
        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness')
        plt.savefig(os.path.join(os.path.dirname(__file__),f'figs/p3_{statistics_names[i]}_1.png'), bbox_inches='tight')
        plt.show()
    ###############################################
    # time.sleep(3)
    # problem_co = Problem3_Constrained_Optimize(
    #     n_var=2, n_obj=1, xl=-1.5, xu=1.5, n_ieq_constr=1, n_eq_constr=0, sqr_e = True)
    # alg = myGA(problem_co)
    # best_ind = alg.evolve(
    #     n_pop=200,
    #     n_gen=400,
    #     n_off=200,
    #     sampling_method="float_sampling",
    #     crossover_method="uniform_crossover",
    #     mutation_method="gaussian_mutation",
    #     crossover_rate=0.7,
    #     mutation_rate=0.5
    # )
    # print('best solution:{}'.format(best_ind.gene))


def run_problem4():
    Letters = ["A", "C", "L", "M", "N", "S", "T", "U", "X"]
    SANTA = [Letters.index(char) for char in "SANTA"]
    CLAUS = [Letters.index(char) for char in "CLAUS"]
    XMAS = [Letters.index(char) for char in "XMAS"]

    problem = Problem4(10, 1, 0, 9, SANTA, CLAUS, XMAS, sqr=True)
    alg = myGA(problem)
    best_ind = alg.evolve(
        n_pop=50,
        n_gen=100,
        n_off=25,
        sampling_method="permutation_sampling",
        crossover_method="order_crossover",
        mutation_method="inversion_mutation",
        crossover_rate=0.7,
        mutation_rate=0.1
    )
    print('best map solution:{}'.format(best_ind.gene))
    alg.plot()


if __name__ == "__main__":
    # run_problem1()
    # time.sleep(3)
    run_problem2()
    # time.sleep(3)
    # run_problem3()
    # time.sleep(3)
    # run_problem4()
