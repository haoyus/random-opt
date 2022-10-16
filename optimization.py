import mlrose_hiive as mlrose
# import six
# import sys
# sys.modules['sklearn.externals.six'] = six
# import mlrose
import numpy as np
from matplotlib import pyplot as plt
import sys
import csv
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

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


def test_problem(name, problem_size):
    if name == 'OneMax':
        MAX_ITR = 200
        LEN = 50
        MAX_ATT = 100
        fitness = mlrose.OneMax()
        init_state = np.random.randint(2,size=LEN)
    elif name == 'FlipFlop':
        MAX_ITR = 1000
        LEN = 50
        MAX_ATT = 100
        fitness = mlrose.FlipFlop()
        init_state = np.random.randint(2,size=LEN)
    elif name == 'FourPeaks':
        MAX_ITR = 2000
        LEN = problem_size
        MAX_ATT = 200
        fitness = mlrose.FourPeaks(t_pct=0.2)
        init_state = np.random.randint(2,size=LEN)
    else:
        return

    print('Testing problem', name)

    best_state_hc, best_fit_hc_0,  curve_hc_0  = run_hc(init_state, LEN, MAX_ITR, fitness, MAX_ATT, 0)
    best_state_hc, best_fit_hc_10, curve_hc_10 = run_hc(init_state, LEN, MAX_ITR, fitness, MAX_ATT, 10)
    t_start = time.time()
    best_state_hc, best_fit_hc_20, curve_hc_20 = run_hc(init_state, LEN, MAX_ITR, fitness, MAX_ATT, 20)
    t_end = time.time()
    print('RHC avg time', (t_end-t_start)/len(curve_hc_20[:,0]))

    best_state_sa, best_fit_sa_99, curve_sa_99 = run_sa(init_state, LEN, MAX_ITR, fitness, MAX_ATT, 0.99)
    best_state_sa, best_fit_sa_85, curve_sa_85 = run_sa(init_state, LEN, MAX_ITR, fitness, MAX_ATT, 0.85)
    t_start = time.time()
    best_state_sa, best_fit_sa_70, curve_sa_70 = run_sa(init_state, LEN, MAX_ITR, fitness, MAX_ATT, 0.70)
    t_end = time.time()
    print('SA avg time ', (t_end-t_start)/len(curve_sa_70[:,0]))

    best_state_ga, best_fit_ga_51, curve_ga_51 = run_ga(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*0.5, 0.1)
    best_state_ga, best_fit_ga_55, curve_ga_55 = run_ga(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*0.5, 0.5)
    best_state_ga, best_fit_ga_21, curve_ga_21 = run_ga(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*2, 0.1)
    best_state_ga, best_fit_ga_25, curve_ga_25 = run_ga(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*2, 0.5)
    best_state_ga, best_fit_ga_41, curve_ga_41 = run_ga(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*4, 0.1)
    t_start = time.time()
    best_state_ga, best_fit_ga_45, curve_ga_45 = run_ga(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*4, 0.5)
    t_end = time.time()
    print('GA avg time', (t_end-t_start)/len(curve_ga_45[:,0]))

    best_state_mm, best_fit_mm_52, curve_mm_52 = run_mm(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*0.5, 0.2)
    best_state_mm, best_fit_mm_55, curve_mm_55 = run_mm(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*0.5, 0.5)
    best_state_mm, best_fit_mm_22, curve_mm_22 = run_mm(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*2, 0.2)
    best_state_mm, best_fit_mm_25, curve_mm_25 = run_mm(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*2, 0.5)
    best_state_mm, best_fit_mm_42, curve_mm_42 = run_mm(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*4, 0.2)
    t_start = time.time()
    best_state_mm, best_fit_mm_45, curve_mm_45 = run_mm(init_state, LEN, MAX_ITR, fitness, MAX_ATT, LEN*4, 0.5)
    t_end = time.time()
    print('MIMIC avg time', (t_end-t_start)/len(curve_mm_45[:,0]))

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
    if name=='FourPeaks':
        YLIM = int(1.8*LEN)
    for row in axs:
        for ax in row:
            ax.grid()
            ax.legend(loc='lower right')
            ax.set_xlabel('iteration')
            ax.set_ylabel('fitness')
            ax.set_title(ALGOS[iii])
            if name=='FourPeaks' and iii==2:
                MAX_ITR=500
            if name=='FourPeaks' and iii==3:
                MAX_ITR=50
            ax.set_xlim(0,MAX_ITR)
            ax.set_ylim(0, YLIM)
            iii+=1
    fig.tight_layout()
    plt.savefig(f'{name}_{LEN}.png')
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
    elif name=='FourPeaks':
        fig, ax = plt.subplots()
        ax.plot(curve_hc_20[:,0], color='b', label='RHC with restarts 20')
        ax.plot(curve_sa_99[:,0], color='g', label='SA with decay r=0.99')
        ax.plot(curve_ga_45[:,0], color='r', label='GA with pop 400%, mut prob 0.2')
        ax.plot(curve_mm_42[:,0], color='k', label='MIMIC with pop 400%, keep 0.5')
        ax.set_xlim(-10, 800)
        ax.set_ylim(0, YLIM)
        ax.grid()
        ax.legend()
        ax.set_xlabel('iteration')
        ax.set_ylabel('fitness')
        plt.savefig(f'{name}_{LEN}_compare.png')


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

rice_cls2int = {'Cammeo':0, 'Osmancik':1}

def load_data(path):
    if 'Cammeo' in path:
        classes = ['Cammeo', 'Osmancik']
        cls2int = rice_cls2int
    else:
        raise Exception('Please name your dataset path following the RICE_PATH and RICE_MSC_PATH')

    loaded = csv.reader(open(path))
    n = 0
    attributes = []
    contents = []
    for row in loaded:
        if n==0:
            attributes = row
        else:
            for i in range(len(row)-1):
                if row[i]=='':
                    row[i]=0
                row[i] = float(row[i])
            row[-1] = cls2int[row[-1]]
            contents.append(row)
        n += 1
    
    attr_inds = {attr:ind for ind, attr in enumerate(attributes)}
    ind_attrs = {ind:attr for ind, attr in enumerate(attributes)}
    # print(len(contents))
    # print(contents[1])
    contents = np.array(contents)
    x = contents[:,:-1]
    y = contents[:, -1]

    return x, y, attributes[:-1], attr_inds, ind_attrs, classes

def test_nn():
    x_train, y_train, x_test, y_test, attributes, classes = prepare_data(RICE_PATH)
    print('size of train data: ', x_train.shape, ', size of test data: ', x_test.shape)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    RUNS = 50
    print('gradient descent')
    gd_train_f1 = []
    gd_test_f1 = []
    for i in range(RUNS):
        max_itr = 10*i
        f1_train, f1_test = fit_nn('gradient_descent', max_itr, x_train,y_train,x_test,y_test)
        gd_train_f1.append(f1_train)
        gd_test_f1.append(f1_test)
    print(gd_train_f1)
    print(gd_test_f1)
    with open('gd_train.npy', 'wb') as f:
        np.save(f, np.array(gd_train_f1))
    with open('gd_test.npy', 'wb') as f:
        np.save(f, np.array(gd_test_f1))

    print('RHC')
    hc_train_f1 = []
    hc_test_f1 = []
    iterations = []
    for i in range(RUNS):
        max_itr = 10*i
        f1_train, f1_test = fit_nn('random_hill_climb', max_itr, x_train,y_train,x_test,y_test)
        hc_train_f1.append(f1_train)
        hc_test_f1.append(f1_test)
        iterations.append(max_itr)
    print(hc_train_f1)
    print(hc_test_f1)
    with open('rhc_train.npy', 'wb') as f:
        np.save(f, np.array([hc_train_f1, iterations]))
    with open('rhc_test.npy', 'wb') as f:
        np.save(f, np.array(hc_test_f1))

    print('SA')
    sa_train_f1 = []
    sa_test_f1 = []
    for i in range(RUNS):
        max_itr = 10*i
        f1_train, f1_test = fit_nn('simulated_annealing', max_itr, x_train,y_train,x_test,y_test)
        sa_train_f1.append(f1_train)
        sa_test_f1.append(f1_test)
    print(sa_train_f1)
    print(sa_test_f1)
    with open('sa_train.npy', 'wb') as f:
        np.save(f, np.array(sa_train_f1))
    with open('sa_test.npy', 'wb') as f:
        np.save(f, np.array(sa_test_f1))

    print('GA')
    ga_train_f1 = []
    ga_test_f1 = []
    for i in range(RUNS):
        max_itr = 10*i
        f1_train, f1_test = fit_nn('genetic_alg', max_itr, x_train,y_train,x_test,y_test)
        ga_train_f1.append(f1_train)
        ga_test_f1.append(f1_test)
    print(ga_train_f1)
    print(ga_test_f1)
    with open('ga_train.npy', 'wb') as f:
        np.save(f, np.array(ga_train_f1))
    with open('ga_test.npy', 'wb') as f:
        np.save(f, np.array(ga_test_f1))
    

def fit_nn(algo, max_itr, x_train, y_train, x_test, y_test):
    if algo not in set(['random_hill_climb', 'gradient_descent', 'simulated_annealing', 'genetic_alg']):
        raise Exception('Wrong algo')
    schedule = mlrose.GeomDecay()
    if algo == 'random_hill_climb' or algo == 'gradient_descent':
        model = mlrose.NeuralNetwork(hidden_nodes = [15], activation = 'relu', \
                                    algorithm = algo, max_iters = max_itr, \
                                    bias = True, is_classifier = True, learning_rate = 0.0001, \
                                    early_stopping = False, clip_max = 5, max_attempts = 100, \
                                    restarts=50)
    elif algo == 'simulated_annealing':
        model = mlrose.NeuralNetwork(hidden_nodes = [15], activation = 'relu', \
                                    algorithm = algo, max_iters = max_itr, \
                                    bias = True, is_classifier = True, learning_rate = 0.0001, \
                                    early_stopping = False, clip_max = 5, max_attempts = 100, \
                                    schedule=schedule)
    elif algo == 'genetic_alg':
        model = mlrose.NeuralNetwork(hidden_nodes = [15], activation = 'relu', \
                                    algorithm = algo, max_iters = max_itr, \
                                    bias = True, is_classifier = True, learning_rate = 0.0001, \
                                    early_stopping = False, clip_max = 5, max_attempts = 100, \
                                    pop_size=200, mutation_prob=0.5)
    model.fit(x_train, y_train)
    y_train_pred= model.predict(x_train)
    y_test_pred = model.predict(x_test)
    f1_train = f1_score(y_train,y_train_pred,average='weighted')
    f1_test  = f1_score(y_test,y_test_pred,average='weighted')
    return f1_train, f1_test

def parse_record():
    gd_train  = np.load('gd_train.npy')
    gd_test   = np.load('gd_test.npy')
    rhc_train = np.load('rhc_train.npy')
    rhc_test  = np.load('rhc_test.npy')
    ga_train  = np.load('ga_train.npy')
    ga_test   = np.load('ga_test.npy')
    sa_train  = np.load('sa_train.npy')
    sa_test   = np.load('sa_test.npy')

    fig, ax = plt.subplots(2,2, figsize=(10,8))
    ax[0,0].plot(range(0,500,10), gd_train, 'b', label ='Train F1 GD (baseline)')
    ax[0,0].plot(range(0,500,10), gd_test,  'b-.', label ='Test F1 GD (baseline)')

    ax[0,1].plot(range(0,500,10), rhc_train[0],'g', label='Train F1 RHC')
    ax[0,1].plot(range(0,500,10), rhc_test, 'g-.', label='Test F1 RHC')

    ax[1,0].plot(range(0,500,10), ga_train, 'r', label ='Train F1 GA')
    ax[1,0].plot(range(0,500,10), ga_test,  'r-.', label ='Test F1 GA')

    ax[1,1].plot(range(0,500,10), sa_train, 'k', label ='Train F1 SA')
    ax[1,1].plot(range(0,500,10), sa_test,  'k-.', label ='Test F1 SA')
    for row in ax:
        for a in row:
            a.set_xlim(0, 500)
            a.set_ylim(0, 1)
            a.grid()
            a.legend()
            a.set_xlabel('opt total iterations')
            a.set_ylabel('F1 score')
    plt.savefig('NN.png')
    plt.clf()

    fig, ax = plt.subplots()
    ax.plot(range(0,500,10), gd_train, 'b', label ='Train F1 GD (baseline)')
    ax.plot(range(0,500,10), gd_test,  'b-.', label ='Test F1 GD (baseline)')
    ax.plot(range(0,500,10), ga_train, 'r', label ='Train F1 GA')
    ax.plot(range(0,500,10), ga_test,  'r-.', label ='Test F1 GA')
    ax.legend()
    ax.grid()
    ax.set_xlabel('opt total iterations')
    ax.set_ylabel('F1 score')
    plt.savefig('NN_final.png')



def main():
    np.random.seed(RAND_SEED)
    test_problem('OneMax', 50)
    test_problem('FlipFlop', 50)
    test_problem('FourPeaks', 50)

    test_nn()
    parse_record()

if __name__ == '__main__':
    main()