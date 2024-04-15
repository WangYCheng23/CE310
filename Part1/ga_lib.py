import copy
import pprint
from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.spatial import distance


class Invidual:
    def __init__(self, gene=None, score=None, constraint=0):
        self.gene = gene
        self.score = score
        self.contraint = constraint

    def __len__(self):
        return len(self.gene)


class Population(np.ndarray):
    pass


class myGA:
    def __init__(self, problem, n_max_iterations=100, minimize=True):
        self.problem = problem
        self.n_var = self.problem.n_var
        self.eval = self.problem.evaluate
        self.n_max_iterations = n_max_iterations
        self.population = []
        self.minimize = minimize
        self.log = []

    def permutation_sampling(self, n_samples):
        # permutation random sampling
        X = np.full((n_samples, self.problem.n_var), 0, dtype=int)
        for i in range(n_samples):
            X[i, :] = np.random.permutation(self.problem.n_var)
        return X

    def float_sampling(self, n_samples):
        # float random sampling
        X = np.random.random((n_samples, self.problem.n_var))
        if self.problem.xl and self.problem.xu:
            xl, xu = self.problem.xl, self.problem.xu
            assert np.all(xu >= xl)
            X = xl + (xu - xl) * X
        return X

    def binary_sampling(self, n_samples):
        # binary random sampling
        val = np.random.random((n_samples, self.problem.n_var))
        return (val < 0.5).astype(bool)

    def tournament_selection(self, pop, n_selects, n_parents=2, tournament_size=2):
        # tournament selection

        # number of random individuals needed
        n_random = n_selects * n_parents * tournament_size

        # number of permutations needed
        n_perms = math.ceil(n_random / len(pop))

        # get random permutations and reshape them
        P = []
        for i in range(n_perms):
            P.append(np.random.permutation(len(pop)))
        P = np.concatenate(P)[:n_random]
        P = np.reshape(P, (n_selects * n_parents, tournament_size))

        # compare using tournament function
        S = np.full(P.shape[0], np.nan)
        for i in range(P.shape[0]):
            a, b = P[i, 0], P[i, 1]
            if self.minimize:
                if pop[a].contraint > 0 or pop[b].contraint > 0:
                    S[i] = a if pop[a].contraint < pop[b].contraint else b
                else:
                    S[i] = a if pop[a].score < pop[b].score else b
            else:
                if pop[a].contraint > 0 or pop[b].contraint > 0:
                    S[i] = a if pop[a].contraint > pop[b].contraint else b
                else:
                    S[i] = a if pop[a].score > pop[b].score else b
        S = S[:, None].astype(int)
        return np.reshape(S, (n_selects, n_parents))

    def uniform_crossover(self, X, crossover_rate=0.7):
        n_matings, n_var = X.shape[0], self.problem.n_var
        _X = copy.deepcopy(X)
        M = np.random.random((n_matings, n_var)) < 0.5
        for i in range(n_matings):
            if np.random.rand() < crossover_rate:
                _X[i][0].gene[M[i]] = X[i][1].gene[M[i]]
                _X[i][1].gene[M[i]] = X[i][0].gene[M[i]]
            _X[i][0].score = 0
            _X[i][1].score = 0

        return _X.flatten()

    def order_crossover(self, X, crossover_rate=0.7, shift=False):
        n_matings, n_var = X.shape[0], self.problem.n_var

        def ox(receiver, donor, seq=None, shift=False):
            assert len(donor) == len(receiver)
            # the sequence which shall be use for the crossover
            seq = seq if not None else random_sequence(len(receiver))
            start, end = seq
            # the donation and a set of it to allow a quick lookup
            donation = np.copy(donor[start:end + 1])
            donation_as_set = set(donation)
            # the final value to be returned
            y = []
            for k in range(len(receiver)):
                # do the shift starting from the swapped sequence - as proposed in the paper
                i = k if not shift else (start + k) % len(receiver)
                v = receiver[i]
                if v not in donation_as_set:
                    y.append(v)
            # now insert the donation at the right place
            y = np.concatenate([y[:start], donation, y[start:]]).astype(
                copy=False, dtype=int)
            return y
        _X = copy.deepcopy(X)
        for i in range(n_matings):
            a, b = X[i, :]
            n = len(a)
            if np.random.rand() < crossover_rate:
                # define the sequence to be used for crossover
                start, end = np.sort(np.random.choice(n, 2, replace=False))
                _X[i][0].gene = ox(
                    a.gene, b.gene, seq=(start, end), shift=shift)
                _X[i][1].gene = ox(
                    b.gene, a.gene, seq=(start, end), shift=shift)
            _X[i][0].score = 0
            _X[i][1].score = 0

        return _X.flatten()

    def point_crossover(self, X, crossover_rate=0.7, n_points=2):
        # get the X of parents and count the matings
        n_matings, n_var = X.shape[0], self.problem.n_var
        # start point of crossover
        r = np.row_stack([np.random.permutation(n_var - 1) +
                         1 for _ in range(n_matings)])[:, :n_points]
        r.sort(axis=-1)
        r = np.column_stack([r, np.full(n_matings, n_var)])
        # create for each individual the crossover range
        _X = copy.deepcopy(X)
        for i in range(n_matings):
            M = np.full(n_var, False)
            j = 0
            while j < r.shape[1] - 1:
                a, b = r[i, j], r[i, j + 1]
                M[a:b] = True
                j += 2

            if np.random.rand() < crossover_rate:
                _X[i][0].gene[M] = X[i][1].gene[M].copy()
                _X[i][1].gene[M] = X[i][0].gene[M].copy()
            _X[i][0].score = 0
            _X[i][1].score = 0

        return _X.flatten()

    def gaussian_mutation(self, pop, mutation_rate=0.5, sigma=0.2):
        for ind in pop:
            if np.random.random() < mutation_rate:
                _gene = np.full(ind.gene.shape, np.inf)
                _gene = np.random.normal(
                    ind.gene, sigma*(self.problem.xu-self.problem.xl))
                if _gene[0] > self.problem.xu:
                    _gene[0] = self.problem.xu - \
                        np.random.random()*(self.problem.xu-ind.gene[0])
                if _gene[1] < self.problem.xl:
                    _gene[1] = self.problem.xl + \
                        np.random.random()*(ind.gene[1]-self.problem.xl)
                ind.gene = _gene

        return pop

    def inversion_mutation(self, pop, mutation_rate=0.5):
        for ind in pop:
            if np.random.random() < mutation_rate:
                start, end = np.sort(np.random.choice(
                    len(ind.gene), 2, replace=False))
                ind.gene[start:end] = np.flip(ind.gene[start:end])
        return pop

    def bit_flip_mutation(self, pop, mutation_rate=0.5):
        for ind in pop:
            if np.random.random() < mutation_rate:
                flip = (np.random.random(ind.gene.shape) < 0.5).astype(bool)
                ind.gene[flip] = ~ind.gene[flip]
        return pop

    def eliminate_duplicates(self, pop, epsilon=1e-16):
        # return np.unique(pop, axis=0)[:n_parents]
        genes = np.array([ind.gene for ind in pop])
        D = distance.cdist(genes.astype(float), genes.astype(float))
        D[np.triu_indices(len(genes))] = np.inf
        D[np.isnan(D)] = np.inf
        return pop[~np.any(D <= epsilon, axis=1)]

    def survival(self, pop, n_survival):
        if n_survival == 0:
            return []
        if len(pop) < n_survival:
            return pop
        if self.minimize:
            S = np.argsort(np.array([ind.score for ind in pop]))
        else:
            S = np.argsort(np.array([ind.score for ind in pop]))[::-1]
        return [pop[s]for s in S[:n_survival]]

    def generate_offsprings(self, pop, num_offsprings, selection_method,
                            crossover_method, crossover_rate, mutation_method, mutation_rate):
        n_selects = math.ceil(num_offsprings/2)
        n_infills = 0
        off = np.array([])
        while len(off) < num_offsprings:
            parents = getattr(self, selection_method)(pop, n_selects)
            parents = np.array([[pop[p1], pop[p2]] for p1, p2 in parents])
            _off = getattr(self, crossover_method)(parents, crossover_rate)
            _off = getattr(self, mutation_method)(_off, mutation_rate)
            _off = self.eliminate_duplicates(_off)
            if len(off)+len(_off) > num_offsprings:
                n_remaining = num_offsprings - len(off)
                _off = _off[:n_remaining]
            off = np.concatenate([off, _off])
            n_infills += 1
            if n_infills >= self.n_max_iterations:
                break
        return off

    def initlize_population(self, pop_size, sampling_method):
        genes = getattr(self, sampling_method)(pop_size)
        populations = [Invidual(gene=g, score=self.eval(
            g)['score'], constraint=self.eval(g)['constraint']) for g in genes]
        return populations

    def evolve(self, n_pop, n_gen, n_off, sampling_method, crossover_method, crossover_rate,
               mutation_method, mutation_rate, selection_method="tournament_selection"):
        pop = self.initlize_population(n_pop, sampling_method)
        for i in range(n_gen):
            off = self.generate_offsprings(
                pop, n_off, selection_method, crossover_method, crossover_rate,
                mutation_method, mutation_rate)
            if len(off) == 0:
                return
            for o in off:
                o.score = self.eval(o.gene)['score']
                o.constraint = self.eval(o.gene)['constraint']

            """ Steady State GA: Origin Pop -> Death + Offsprings """
            death = len(off)
            n_survival = n_pop-death
            pop = self.survival(pop, n_survival)
            pop = np.concatenate([pop, off])

            # statistic
            min_score = np.min([p.score for p in pop])
            max_score = np.max([p.score for p in pop])
            mean_score = np.mean([p.score for p in pop])
            median_score = np.median([p.score for p in pop])
            std_score = np.std([p.score for p in pop])
            # self.log.append({'min': min_score, 'max': max_score, 'mean': mean_score, 'median': median_score, 'std': std_score})
            self.log.append([min_score, max_score, mean_score, median_score, std_score])           
            print(
                f'itr{i}: min:{min_score} max:{max_score} mean:{mean_score} median:{median_score} std:{std_score}')
        print("finished evolving!")
        best_ind = sorted(pop, key=lambda x: x.score)[0]
        return best_ind

    def plot(self):
        self.log = np.array(self.log)
        itr = list(range(len(self.log)))
        min = self.log[:, 0]
        max = self.log[:, 1]
        mean = self.log[:, 2]
        median = self.log[:, 3]
        std = self.log[:, 4]
        fig, ax = plt.subplots()
        ax.plot(itr, min, label='min')
        ax.plot(itr, max, label='max')
        ax.plot(itr, mean, label='mean')
        ax.plot(itr, median, label='median')
        ax.plot(itr, std, label='std')
        ax.legend()
        ax.set_xlabel('iteration')
        ax.set_ylabel('score') 
        ax.set_title('evolution of scores')
        
        plt.show()
     
    def log(self):
        return np.array(self.log)
        