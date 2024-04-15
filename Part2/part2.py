#!/usr/bin/env python3

################################################################################
"""
CE310 Evolutionary Computation and Genetic Programming
Tasos Papastylianou, Spring 2024
Assignment: Programming Assignment and mini project Part 2 (of 2) – Mini project
            Genetic Programming and Symbolic Regression

For information on symbolic regression refer to the relevant Unit 5 slides on
Moodle, or visit: https://en.wikipedia.org/wiki/Symbolic_regression

This code makes use of the DEAP package for the Genetic Programming parts.
Visit https://deap.readthedocs.io to learn more about DEAP. """
################################################################################

# Import relevant builtin/core Python modules
import datetime
import os
import pprint
import sys
import operator
import math
import random
import numpy
import pandas as pd
from matplotlib import pyplot
from functools import partial

# Import DEAP modules
import deap.algorithms
import deap.base
import deap.creator
import deap.tools
import deap.gp

import numpy as np

###############
# PARAMETERS
###############

# These variables are defined globally in the module.

# Configurable parameters; CHANGE THESE APPROPRIATELY (or pass as arguments to
# the script at the commandline, to have them populated automatically when
# running as a script: see the 'python boilerplate' at the end of this file).
#
# In your experiments, make sure to perform at least 10 runs per unique
# hyperparameter set, not just a single run.

Problem = None
PopulationSize = None  # can be overriden by arguments passed to script
TournamentSize = None    # can be overriden by arguments passed to script

# Fixed parameters
NumGenerations = 30    # number of generations

# other parameters that you can change and explore
CrossoverRate = 0.7
MutationRate = 0.3
UseSqError = True   # whether to use Least Squares approach or Least Absolute Error

# other global variables, which will be initialised and used by functions later
InputPoints = None
NumInputPoints = None
TargetFunction = None
TargetPoints = None
PrimitiveSet = None
Toolbox = None
MStats = None

DIR = os.path.dirname(__file__)
current_time_str = str(datetime.datetime.now()).replace(
    ":", "").replace(" ", "").replace("-", "").replace(".", "")
figs_folder = os.path.join(DIR, "figs/" + current_time_str + "/")
logs_folder = os.path.join(DIR, "logs/" + current_time_str + "/")
res_folder = os.path.join(DIR, "res/" + current_time_str + "/")
os.makedirs(figs_folder, exist_ok=True)
os.makedirs(logs_folder, exist_ok=True)
os.makedirs(res_folder, exist_ok=True)
####################
# Helper functions
####################


def p1(x):
    return 5*x**5 + 4*x**4 + 3*x**3 + 2*x**2 + x


def p2(x):
    return 4*math.sin(5*math.pi*x/4)


def create_dataset():
    """
    Create dataset (input-points, target-function, and target-points).
    Note: These are updated in-place at global scope, not returned.
    """

    global InputPoints
    global NumInputPoints
    global TargetFunction
    global TargetPoints
    global Problem

    InputPoints = numpy.linspace(-math.pi, math.pi, 65).tolist()
    NumInputPoints = len(InputPoints)
    TargetFunction = Problem
    TargetPoints = [TargetFunction(Input) for Input in InputPoints]


def protectedDiv(Numerator, Denominator):
    """
    Protected division operator, to prevent division by zero errors during GP runs.
    """
    return 1 if Denominator == 0 else Numerator / Denominator


def create_primitive_set():
    """
    Create a primitive set for use with DEAP's GP functions, suitable for our
    problem.

    Note: This is updated *in-place* at global scope, not returned.
    """

    global PrimitiveSet

    PrimitiveSet = deap.gp.PrimitiveSet("MAIN", 1)

    PrimitiveSet.addPrimitive(operator.add, 2)
    PrimitiveSet.addPrimitive(operator.sub, 2)
    PrimitiveSet.addPrimitive(operator.mul, 2)
    PrimitiveSet.addPrimitive(operator.neg, 1)
    PrimitiveSet.addPrimitive(math.cos, 1)
    PrimitiveSet.addPrimitive(math.sin, 1)
    PrimitiveSet.addPrimitive(protectedDiv, 2)   # defined in this module

    PrimitiveSet.addTerminal(1)
    PrimitiveSet.addTerminal(-1)

    PrimitiveSet.addEphemeralConstant(
        "rand101", partial(random.randint, -1, 1))
    #    f"rand{random.randint(100,999)}", partial(random.randint, -1, 1))
    PrimitiveSet.renameArguments(ARG0='x')


def create_toolbox():
    """
    Create a suitable 'toolbox' object for use with DEAP's GP functions.
    Note: This is updated *in-place* at global scope, not returned.
    """

    global Toolbox

    # The 'create' method is a convenience method for creating classes, which then
    # become assigned to the deap.creator module (i.e. as attributes).
    # The -1.0 weight tells DEAP this is a minimization goal
    deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
    # Results in a class called "Individual" assigned to the 'creator' module.
    deap.creator.create("Individual", deap.gp.PrimitiveTree,
                        fitness=deap.creator.FitnessMin)

    Toolbox = deap.base.Toolbox()   # A deap object, designed to hold function
    # definitions for convenience.

    # The 'register' method is a convenience wrapper which gives an alias to a
    # 'partial function call', i.e. calling another function, potentially with
    # some arguments as already given. The alias then appears as a callable
    # function, which is assigned as an attribute to the Toolbox object. For more
    # information about the aliased functions themselves, see their documentation.
    # This results in `Toolbox.expr()` returning a random subtree expression, than can be given as input to create an Individual from it
    Toolbox.register("expr", deap.gp.genHalfAndHalf,
                     pset=PrimitiveSet, min_=1, max_=2)
    # This results in `Toolbox.individual()` returning `Individual( expr() )`, i.e. a new individual.
    Toolbox.register("individual", deap.tools.initIterate,
                     deap.creator.Individual, Toolbox.expr)
    # This results in `Toolbox.population( N )` returning a python list of N randomly generated individuals
    Toolbox.register("population", deap.tools.initRepeat,
                     list, Toolbox.individual)
    # This results in `Toolbox.compile( Individual )` returning a proper, python-runnable function from a syntax tree.
    Toolbox.register("compile", deap.gp.compile, pset=PrimitiveSet)

    # This results in `Toolbox.evaluate( Individual )` returning a fitness score (in this case, a minimizable one).
    Toolbox.register("evaluate", EvaluateSymbolicRegression)
    # This results in `Toolbox.select( Individuals, k )` performing k tournaments on the provided individuals.
    Toolbox.register("select", deap.tools.selTournament,
                     tournsize=TournamentSize)
    # This results in `Toolbox.mate( Ind1, Ind2 )` performing crossover *in-place* (and also returning the resulting individuals)
    Toolbox.register("mate", deap.gp.cxOnePoint)
    # This results in `Toolbox.expr_mut` returning a tree of 'full' size. Used in mutation operations.
    Toolbox.register("expr_mut", deap.gp.genFull, min_=0, max_=2)
    # This results in `Toolbox.mutate( Individual )` performing mutation *in-place* (and also returning the mutated individual)
    Toolbox.register("mutate", deap.gp.mutUniform,
                     expr=Toolbox.expr_mut, pset=PrimitiveSet)

  # Modify crossover and mutation operations to prevent exceeding certain depth limits
    Toolbox.decorate("mate", deap.gp.staticLimit(
        key=operator.attrgetter("height"), max_value=64))
    Toolbox.decorate("mutate", deap.gp.staticLimit(
        key=operator.attrgetter("height"), max_value=64))


def EvaluateSymbolicRegression(individual):
    """
    The error function used to evaluate individuals' fitness.
    Used internally later by the toolbox.evaluate function.
    """

  # Transform the tree expression to a callable function
    func = Toolbox.compile(expr=individual)

    if UseSqError:

      # squared error
        error = (abs(func(x) - TargetFunction(x)) ** 2 for x in InputPoints)

    else:

      # Absolute distance between target curve and solution
        error = (abs(func(x) - TargetFunction(x)) for x in InputPoints)

    return math.fsum(error)/len(InputPoints),


def create_mstats_object():
    """
    A DEAP multistats object, to keep track of generation statistics.
    """

    global MStats

    stats_fit = deap.tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = deap.tools.Statistics(len)
    mstats = deap.tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("mdn", numpy.median)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    MStats = mstats


def plot_fitness_and_size(Logging):
    """
    Plot a comparison of fitness and size as a function of generation.
    """
    global Problem
    global PopulationSize
    global TournamentSize

    x = numpy.arange(0, NumGenerations+1)
    size = Logging.chapters['size']
    fitness = Logging.chapters['fitness']

    s = size.select("mdn")
    f = fitness.select("mdn")

    df_size = pd.DataFrame({'gen': x, 'max': size.select("max"), 'mdn': size.select("mdn"),
                            'min': size.select("min"), 'avg': size.select("avg"), 'std': size.select("std")})
    df_fitness = pd.DataFrame({'Generation': x, 'max': fitness.select("max"), 'mdn': fitness.select("mdn"),
                               'min': fitness.select("min"), 'avg': fitness.select("avg"), 'std': fitness.select("std")})
    df_size.to_excel(os.path.join(os.path.dirname(__file__),
                                  'logs/{}/Size_{}_{}_{}.xlsx'.format(current_time_str, Problem.__name__, PopulationSize, TournamentSize)))
    df_fitness.to_excel(os.path.join(os.path.dirname(__file__),
                                     'logs/{}/Fitness_{}_{}_{}.xlsx'.format(current_time_str, Problem.__name__, PopulationSize, TournamentSize)))

    fig, ax = pyplot.subplots()
    ax.plot(x, f/max(f), 'k--', label='Fitness')
    ax.plot(x, s/max(s), 'k:', label='Size')
    ax.set_xlabel('Generations')
    ax.set_ylabel('Normalised Fitness/Size')
    # ax.set_title('Median')
    legend = ax.legend(shadow=True, fontsize='x-large')
    ax.set_title('Median_fitness_and_size_{}_{}_{}'.format(
        Problem.__name__, PopulationSize, TournamentSize))
    print('Fitnes: [' + str(min(f))+', '+str(max(f))+']')
    print('Size: [' + str(min(s))+', '+str(max(s))+']')
    print('Evaluations: ' + str(sum(Logging.select("nevals"))))

    pyplot.savefig(os.path.join(os.path.dirname(__file__),
                                'figs/{}/Fitness_and_size_{}_{}_{}.png'.format(current_time_str, Problem.__name__, PopulationSize, TournamentSize)))

    pyplot.show()


def plot_dataset(InputPoints, TargetPoints):
    """
    Visualise our known solution, that the GP is expected to approximate.
    """
    Fig, Ax = pyplot.subplots(figsize=(15, 4))
    Ax.scatter(InputPoints, TargetPoints)
    Ax.set_xlabel('Test points')
    Ax.set_ylabel('Measurements')
    Ax.set_title('Dataset')
    pyplot.show()


def plot_comparison_Target_Vs_EvolvedSolution(HOF):
    """
    Compare the found (i.e. 'evolved') solution to the known one.
    """
    global Problem
    global PopulationSize
    global TournamentSize

    x = InputPoints
    f = Toolbox.compile(expr=HOF[0])

    y = numpy.empty(len(x))
    for i in range(len(x)):
        y[i] = f(x[i])

    fig, ax = pyplot.subplots()
    ax.plot(x, y, 'r-', label='Best Solution')
    ax.plot(x, TargetPoints, 'k-', label='Target func')
    # legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    legend = ax.legend(shadow=True, fontsize='x-large')
    ax.set_xlabel('Test Range')
    ax.set_ylabel('Test Function Value')
    ax.set_title('Comparison_Target_Vs_EvolvedSolution_{}_{}_{}'.format(
        Problem.__name__, PopulationSize, TournamentSize))

    pyplot.savefig(os.path.join(os.path.dirname(__file__),
                                './figs/{}/Comparison_Target_Vs_EvolvedSolution_{}_{}_{}.png'.format(current_time_str, Problem.__name__, PopulationSize, TournamentSize)))

    pyplot.show()


#################
# Main function
#################

# Note: makes use of parameters defined above, and helper functions declared below

def try_another_10(problems):
    global PopulationSize
    global TournamentSize

    # best configuration
    PopulationSize = 2000
    TournamentSize = 5

    for problem in problems:
        global Problem
        Problem = problem
        HOFs = []
        min = []
        max = []
        avg = []
        mdn = []
        std = []
        for step in range(10):
            print('#'*50 + f'step {step}' + '#'*50)
            # Initialise the necessary resources for the run
            create_dataset()
            create_primitive_set()
            create_toolbox()
            create_mstats_object()
            # keeps track of the best individual 'ever' at each generation
            HOF = deap.tools.HallOfFame(1)

            # Let's visualise our data first
            # plot_dataset(InputPoints, TargetPoints)

            # Create an initial random population of individuals
            random.seed()
            Population = Toolbox.population(n=PopulationSize)

            # Run the GP
            Population, Logging = deap.algorithms.eaSimple(Population, Toolbox, CrossoverRate, MutationRate, NumGenerations, stats=MStats,
                                                           halloffame=HOF, verbose=False)
            size = Logging.chapters['size']
            fitness = Logging.chapters['fitness']

            min.append(fitness.select("min")[-1])
            max.append(fitness.select("max")[-1])
            avg.append(fitness.select("avg")[-1])
            std.append(fitness.select("std")[-1])
            mdn.append(fitness.select("mdn")[-1])
            # Inspect results
            # plot_fitness_and_size(Logging)
            # print(HOF[0])   # Best individual
            # with open(os.path.join(os.path.dirname(__file__), f"res/{current_time_str}/{Problem.__name__}, {PopulationSize}, {TournamentSize}.txt"), "w") as f:
            #     pprint.pprint(str(HOF[0]), stream=f)
            # plot_comparison_Target_Vs_EvolvedSolution(HOF)
            HOFs.append(HOF[0])
        df = pd.DataFrame({'min': min, 'max': max, 'avg': avg, 'std': std, 'mdn': mdn})
        df.to_excel(os.path.join(os.path.dirname(__file__),f'logs/{Problem.__name__}.xlsx'))
        fig, ax = pyplot.subplots()
        x = InputPoints
        ax.plot(x, TargetPoints, 'k-', label='Target func')
        for idx, hof in enumerate(HOFs):

            f = Toolbox.compile(expr=hof)
            y = numpy.empty(len(x))
            for i in range(len(x)):
                y[i] = f(x[i])
            ax.plot(x, y, '--', label=f'Best Solution with {idx} times run')
            ax.legend()
            ax.set_xlabel('Test Range')
            ax.set_ylabel('Test Function Value')
            ax.set_title('Comparison_Target_Vs_EvolvedSolution')
        pyplot.savefig(os.path.join(os.path.dirname(__file__),
                       f'figs/comparsion_{Problem.__name__}.png'))
        pyplot.show()


def main(problems, population_sizes, tounament_sizes):
    """
    GP Goal: evolve a function f(x) (mathematical expression or model) with
    x = InputPoints that best fits a target dataset
    """
    for problem in problems:
        global Problem
        Problem = problem
        param = []
        min = []
        max = []
        avg = []
        mdn = []
        std = []
        HOFs = []
        for pop_size in population_sizes:
            global PopulationSize
            PopulationSize = pop_size
            for tour_size in tounament_sizes:
                global TournamentSize
                TournamentSize = tour_size
                
                print('#'*100)
                print(
                    f"Running GP for problem {problem.__name__}, pop size: {pop_size}, tour size: {tour_size}")
                # Initialise the necessary resources for the run
                create_dataset()
                create_primitive_set()
                create_toolbox()
                create_mstats_object()
                # keeps track of the best individual 'ever' at each generation
                HOF = deap.tools.HallOfFame(1)

                # Let's visualise our data first
                # plot_dataset(InputPoints, TargetPoints)

                # Create an initial random population of individuals
                random.seed()
                Population = Toolbox.population(n=PopulationSize)

                # Run the GP
                Population, Logging = deap.algorithms.eaSimple(Population, Toolbox, CrossoverRate, MutationRate, NumGenerations, stats=MStats,
                                                               halloffame=HOF, verbose=False)
                # Inspect results
                # plot_fitness_and_size(Logging)
                # print(HOF[0])   # Best individual
                # with open(os.path.join(os.path.dirname(__file__), f"res/{current_time_str}/{Problem.__name__}, {PopulationSize}, {TournamentSize}.txt"), "w") as f:
                #     pprint.pprint(str(HOF[0]), stream=f)
                # plot_comparison_Target_Vs_EvolvedSolution(HOF)

                # statistic
                HOFs.append(HOF[0])

                param.append(f"{pop_size}_{tour_size}")
                size = Logging.chapters['size']
                fitness = Logging.chapters['fitness']

                min.append(fitness.select("min"))
                max.append(fitness.select("max"))
                avg.append(fitness.select("avg"))
                std.append(fitness.select("std"))
                mdn.append(fitness.select("mdn"))

                print(min[-1][-1], mdn[-1][-1])

        fig, ax = pyplot.subplots()
        x = InputPoints
        ax.plot(x, TargetPoints, 'k-', label='Target func')
        for i, hof in enumerate(HOFs):
            hypcfg = param[i]

            f = Toolbox.compile(expr=hof)
            y = numpy.empty(len(x))
            for i in range(len(x)):
                y[i] = f(x[i])
            ax.plot(x, y, '--', label=f'Best Solution with config {hypcfg}')
            ax.legend()
            ax.set_xlabel('Test Range')
            ax.set_ylabel('Test Function Value')
            ax.set_title('Comparison_Target_Vs_EvolvedSolution')
        pyplot.savefig(os.path.join(os.path.dirname(__file__),
                       f'figs/comparsion_{Problem.__name__}.png'))
        pyplot.show()

        min = np.array(min)
        max = np.array(max)
        avg = np.array(avg)
        std = np.array(std)
        mdn = np.array(mdn)
        D = {'min': min, 'max': max, 'avg': avg, 'std': std, 'mdn': mdn}
        for k, v in D.items():
            fig, ax = pyplot.subplots()
            x = list(range(v.shape[1]))
            for i, label in enumerate(param):
                ax.plot(x, v[i], label=label)
            ax.legend()
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.set_title(f"comparison of {k} values in { Problem.__name__}")
            pyplot.savefig(os.path.join(os.path.dirname(
                __file__), f'figs/{k}_{Problem.__name__}.png'))
            pyplot.show()


#####################################################################
# Python 'boilerplate' (for when the module is invoked as a script)
#####################################################################
if __name__ == "__main__":

    if len(sys.argv) == 1:   # i.e., no arguments provided to script on the commandline

        # use default values for NumGenerations, PopulationSize, and TournamentSize, as defined earlier in the module at top-level scope
        print(
            f"Using default values for NumGenerations ({NumGenerations}), PopulationSize ({PopulationSize}), and TournamentSize ({TournamentSize}).")
        # main(problems=[p1, p2], population_sizes=[
        #      500, 2000], tounament_sizes=[2, 5])
        try_another_10(problems=[p1, p2])

    elif len(sys.argv) == 4:   # i.e., script was called with 2 arguments

        try:
            NumGenerations = int(sys.argv[1])
            PopulationSize = int(sys.argv[2])
            TournamentSize = int(sys.argv[3])
        except ValueError:
            print(
                "Error: NumGenerations, PopulationSize, and TournamentSize should be valid integer inputs")
            exit()

        main(problems=[p1, p2], population_sizes=[
             500, 2000], tounament_sizes=[2, 5])

    else:

        ErrorMessage = "\nERROR: When called as a script, you need to either call the script with three arguments (correponding to the NumGenerations, PopulationSize, and TournamentSize respectively), or with no arguments (which then uses the defaults, i.e. 30, 500 and 5)."

        raise RuntimeError(ErrorMessage)
