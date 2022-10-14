import mlrose_hiive as mlrose
# import six
# import sys
# sys.modules['sklearn.externals.six'] = six
# import mlrose
import numpy as np
from matplotlib import pyplot as plt


def run_four_peaks():
    MAX_ITR = 5000
    LEN = 100
    fitness = mlrose.FourPeaks(t_pct=0.4)
    problem = mlrose.DiscreteOpt(length=LEN,fitness_fn=fitness, maximize=True)
    # score = fitness.evaluate(init_state)
    # print(score)

    # randome hill climb
    init_state = np.random.randint(2,size=LEN)
    best_state_hc, best_fit_hc, curve_hc = mlrose.random_hill_climb(problem, 10, MAX_ITR, 10, init_state, curve=True)
    # print(curve_hc)

    # simulated annealing  
    init_state = np.random.randint(2,size=LEN)
    schedule = mlrose.ExpDecay()
    best_state_sa, best_fit_sa, curve_sa = mlrose.simulated_annealing(problem, schedule, 10, MAX_ITR, init_state, curve=True)
    # print(curve_sa)

    # GA
    best_state_ga, best_fit_ga, curve_ga = mlrose.genetic_alg(problem, max_iters=MAX_ITR, curve=True)
    # MIMIC
    best_state_mm, best_fit_mm, curve_mm = mlrose.mimic(problem, max_iters=MAX_ITR, curve=True)

    fig, axs = plt.subplots()
    axs.plot(curve_hc[:,0], 'o', color='g', label='Random Hill Climb')
    axs.plot(curve_sa[:,0], color='b', label='Simulated Annealing')
    axs.plot(curve_ga[:,0], color='k', label='Genetic Algorithm')
    axs.plot(curve_mm[:,0], color='g', label='MIMIC')
    axs.legend()
    axs.grid()
    plt.savefig('FourPeaks.png')
    plt.clf()

    

def run_flip_flop():
    MAX_ITR = 1000
    LEN = 20
    fitness = mlrose.FlipFlop()

    state = np.random.randint(2,size=LEN)
    problem = mlrose.DiscreteOpt(length=LEN,fitness_fn=fitness)

def main():
    np.random.seed(456)
    run_four_peaks()

if __name__ == '__main__':
    main()