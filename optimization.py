import mlrose_hiive as mlrose
# import six
# import sys
# sys.modules['sklearn.externals.six'] = six
# import mlrose
import numpy as np
from matplotlib import pyplot as plt
import sys

from sklearn.model_selection import train_test_split

ALGOS = [
    'Random Hill Climb',
    'Simulated Annealing',
    'Genetic Algorithm',
    'MIMIC'
]

RAND_SEED = 456

def run_hc(init_state, state_len, max_iter, fitfn, max_att, num_restarts):
    problem = mlrose.DiscreteOpt(length=state_len,fitness_fn=fitfn)
    best_state_hc, best_fit_hc, curve_hc = mlrose.random_hill_climb(
        problem, max_att, max_iter, restarts=num_restarts, init_state=init_state, curve=True, random_state=RAND_SEED)
    return best_state_hc, best_fit_hc, curve_hc

def run_sa(init_state, state_len, max_iter, fitfn, max_att, decay_r):
    problem = mlrose.DiscreteOpt(length=state_len,fitness_fn=fitfn)
    schedule = mlrose.GeomDecay(decay=decay_r)
    best_state, best_fit, curve = mlrose.simulated_annealing(
        problem, schedule, max_att, max_iter, init_state=init_state, curve=True, random_state=RAND_SEED)
    return best_state, best_fit, curve

def run_ga(init_state, state_len, max_iter, fitfn, max_att, pop_size, mut_prob):
    problem = mlrose.DiscreteOpt(length=state_len,fitness_fn=fitfn)
    best_state, best_fit, curve = mlrose.genetic_alg(
        problem, max_attempts=max_att, max_iters=max_iter, curve=True,
        random_state=RAND_SEED, pop_size=pop_size, mutation_prob=mut_prob)
    return best_state, best_fit, curve

def run_mm(init_state, state_len, max_iter, fitfn, max_att, pop_size, keep_pct):
    problem = mlrose.DiscreteOpt(length=state_len,fitness_fn=fitfn)
    best_state, best_fit, curve = mlrose.mimic(
        problem, max_attempts=max_att, max_iters=max_iter, curve=True,
        random_state=RAND_SEED, pop_size=pop_size, keep_pct=keep_pct)
    return best_state, best_fit, curve


def test_problem(name):
    if name == 'OneMax':
        MAX_ITR = 200
        LEN = 50
        MAX_ATT = 100
        fitness = mlrose.OneMax()
        init_state = np.random.randint(2,size=LEN)
    elif name == 'FlipFlop':
        MAX_ITR = 1000
        LEN = 50
        MAX_ATT = 200
        fitness = mlrose.FlipFlop()
        init_state = np.random.randint(2,size=LEN)
    elif name == 'FourPeaks':
        MAX_ITR = 2000
        LEN = 50
        MAX_ATT = 500
        fitness = mlrose.FourPeaks(t_pct=0.4)
        init_state = np.random.randint(2,size=LEN)
    else:
        return

    print('Testing problem ', name)

    best_state_hc, best_fit_hc_0,  curve_hc_0  = run_hc(init_state, LEN, MAX_ITR, fitness, MAX_ATT, 0)
    best_state_hc, best_fit_hc_10, curve_hc_10 = run_hc(init_state, LEN, MAX_ITR, fitness, MAX_ATT, 10)
    best_state_hc, best_fit_hc_20, curve_hc_20 = run_hc(init_state, LEN, MAX_ITR, fitness, MAX_ATT, 20)

    best_state_sa, best_fit_sa_99, curve_sa_99 = run_sa(init_state, LEN, MAX_ITR, fitness, MAX_ATT, 0.99)
    best_state_sa, best_fit_sa_85, curve_sa_85 = run_sa(init_state, LEN, MAX_ITR, fitness, MAX_ATT, 0.85)
    best_state_sa, best_fit_sa_70, curve_sa_70 = run_sa(init_state, LEN, MAX_ITR, fitness, MAX_ATT, 0.70)

    best_state_ga, best_fit_ga_51, curve_ga_51 = run_ga(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*0.5, 0.1)
    best_state_ga, best_fit_ga_55, curve_ga_55 = run_ga(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*0.5, 0.5)
    best_state_ga, best_fit_ga_21, curve_ga_21 = run_ga(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*2, 0.1)
    best_state_ga, best_fit_ga_25, curve_ga_25 = run_ga(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*2, 0.5)
    best_state_ga, best_fit_ga_41, curve_ga_41 = run_ga(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*4, 0.1)
    best_state_ga, best_fit_ga_45, curve_ga_45 = run_ga(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*4, 0.5)

    best_state_mm, best_fit_mm_52, curve_mm_52 = run_mm(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*0.5, 0.2)
    best_state_mm, best_fit_mm_55, curve_mm_55 = run_mm(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*0.5, 0.5)
    best_state_mm, best_fit_mm_22, curve_mm_22 = run_mm(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*2, 0.2)
    best_state_mm, best_fit_mm_25, curve_mm_25 = run_mm(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*2, 0.5)
    best_state_mm, best_fit_mm_42, curve_mm_42 = run_mm(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*4, 0.2)
    best_state_mm, best_fit_mm_45, curve_mm_45 = run_mm(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*4, 0.5)

    print('RHC best fit scores: ', best_fit_hc_0, best_fit_hc_10, best_fit_hc_20)
    print('SA  best fit scores: ', best_fit_sa_99, best_fit_sa_85, best_fit_sa_70)
    print('GA  best fit scores: ', best_fit_ga_51, best_fit_ga_55, best_fit_ga_21, best_fit_ga_25, best_fit_ga_41, best_fit_ga_45)
    print('MM  best fit scores: ', best_fit_mm_52, best_fit_mm_55, best_fit_mm_22, best_fit_mm_25, best_fit_mm_42, best_fit_mm_45)

    fig, axs = plt.subplots(2,2, figsize=(10,8))

    axs[0,0].plot(curve_hc_0[:,0],  color='g', label='restarts 0')
    axs[0,0].plot(curve_hc_10[:,0], color='b', label='restarts 10')
    axs[0,0].plot(curve_hc_20[:,0], 'r-.', label='restarts 20')

    axs[0,1].plot(curve_sa_99[:,0], color='g', label='decay r=0.99')
    axs[0,1].plot(curve_sa_85[:,0], color='b', label='decay r=0.85')
    axs[0,1].plot(curve_sa_70[:,0], 'r-.', label='decay r=0.70')

    axs[1,0].plot(curve_ga_51[:,0], color='g', label='pop 50%, mut prob 0.1')
    axs[1,0].plot(curve_ga_55[:,0], color='b', label='pop 50%, mut prob 0.5')
    axs[1,0].plot(curve_ga_21[:,0], color='r', label='pop 200%, mut prob 0.1')
    axs[1,0].plot(curve_ga_25[:,0], color='k', label='pop 200%, mut prob 0.5')
    axs[1,0].plot(curve_ga_41[:,0], 'r-.', label='pop 400%, mut prob 0.1')
    axs[1,0].plot(curve_ga_45[:,0], 'k-.', label='pop 400%, mut prob 0.5')

    axs[1,1].plot(curve_mm_52[:,0], color='g', label='pop 50%, keep 0.2')
    axs[1,1].plot(curve_mm_55[:,0], color='b', label='pop 50%, keep 0.5')
    axs[1,1].plot(curve_mm_22[:,0], color='r', label='pop 200%, keep 0.2')
    axs[1,1].plot(curve_mm_25[:,0], color='k', label='pop 200%, keep 0.5')
    axs[1,1].plot(curve_mm_42[:,0], 'r-.', label='pop 400%, keep 0.2')
    axs[1,1].plot(curve_mm_45[:,0], 'k-.', label='pop 400%, keep 0.5')
    iii = 0
    for row in axs:
        for ax in row:
            ax.grid()
            ax.legend(loc='lower right')
            ax.set_xlabel('iteration')
            ax.set_ylabel('fitness')
            ax.set_title(ALGOS[iii])
            iii+=1
            ax.set_xlim(0,MAX_ITR)
            ax.set_ylim(0, LEN+1)
    fig.tight_layout()
    plt.savefig(f'{name}.png')
    plt.clf()

    if name=='OneMax':
        fig, ax = plt.subplots()
        ax.plot(curve_hc_10[:,0], color='b', label='RHC with restarts 10')
        ax.plot(curve_sa_99[:,0], color='g', label='SA with decay r=0.99')
        ax.plot(curve_ga_45[:,0], color='r', label='GA with pop 400%, mut prob 0.5')
        ax.plot(curve_mm_42[:,0], color='k', label='MIMIC with pop 400%, keep 0.2')
        ax.set_xlim(-10, 200)
        ax.set_ylim(0, LEN+1)
        ax.grid()
        ax.legend()
        ax.set_xlabel('iteration')
        ax.set_ylabel('fitness')
        plt.savefig(f'{name}_compare.png')
    elif name=='FlipFlop':
        fig, ax = plt.subplots()
        ax.plot(curve_hc_10[:,0], color='b', label='RHC with restarts 10')
        ax.plot(curve_sa_70[:,0], color='g', label='SA with decay r=0.7')
        ax.plot(curve_ga_41[:,0], color='r', label='GA with pop 400%, mut prob 0.1')
        ax.plot(curve_mm_42[:,0], color='k', label='MIMIC with pop 400%, keep 0.2')
        ax.set_xlim(-10, 600)
        ax.set_ylim(0, LEN+1)
        ax.grid()
        ax.legend()
        ax.set_xlabel('iteration')
        ax.set_ylabel('fitness')
        plt.savefig(f'{name}_compare.png')


####################### NN test ##########################
RICE_PATH = './Rice_Cammeo_Osmancik.csv'
def do_split(X, Y):
    """do 80/20 split on list of samples"""
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return x_train, y_train, x_test, y_test
def prepare_data(path):
    X, Y, attributes, attr_inds, ind_attrs, classes = load_data(path)

    print(f'Dataset of {path}:')
    print(f'Attributes: {attributes}, total {len(attributes)} attributes')
    print(f'Classes: {classes}')
    C = int(np.max(Y)) + 1
    for i in range(C):
        this_count = np.sum(Y==i)
        print(f'  class {i} has {this_count} samples, takes {round(this_count/len(Y),2)} of total {len(Y)}')
    # print('Basic statistics: total num of samples - ' , len(Y), ', class balance: ', round(1-np.sum(Y)/len(Y),3), ':', round(np.sum(Y)/len(Y),3))

    x_train, y_train, x_test, y_test = do_split(X, Y)
    print('After train test split, train vs test samples: ' ,len(x_train), len(x_test))
    print('Train set balance:')
    for i in range(C):
        this_count = np.sum(y_train==i)
        print(f'  class {i} has {this_count} samples, takes {round(this_count/len(y_train),2)} of total {len(y_train)}')
    print('Test set balance:')
    for i in range(C):
        this_count = np.sum(y_test==i)
        print(f'  class {i} has {this_count} samples, takes {round(this_count/len(y_test),2)} of total {len(y_test)}')
    print('========================')
    return x_train, y_train, x_test, y_test, attributes, classes

def test_nn():
    pass


def main():
    np.random.seed(RAND_SEED)
    test_problem('OneMax')
    test_problem('FlipFlop')
    test_problem('FourPeaks')
    # test_four_peaks()
    # test_flip_flop()

if __name__ == '__main__':
    main()