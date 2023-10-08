from pennylane import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score
from ChemModel import translator, quantum_net
from Arguments import Arguments
import random
import pickle
import csv, os
import torch.multiprocessing as mp
import pennylane as qml
from math import pi

def net2str(net):
        net_str = ''
        for i in range(len(net)):
              net_str += str(net[i])
        return net_str

def get_time(f):
        def inner(*arg,**kwarg):
            s_time = time.time()
            res = f(*arg,**kwarg)
            e_time = time.time()
            print('Time：{}'.format(round((e_time - s_time), 2)))
            return res
        return inner

@get_time
def chemistry(hamiltonian, design, net, rate = 100, verbose=None):
    seeds = [20, 21, 30, 33, 36, 42, 43, 55, 67, 170]

    args = Arguments()
    lr = args.qlr

    symbols = ["O", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.45, -0.1525, -0.8454]])

    # Building the molecular hamiltonian for the trihydrogen cation
    # hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, charge=1)
    

    dev = qml.device("lightning.qubit", wires=args.n_qubits)
    @qml.qnode(dev, diff_method="adjoint")

    def cost_fn(theta):
        quantum_net(theta, design)
        return qml.expval(hamiltonian)
   
    energy = []
    for i in range(1):
        np.random.seed(seeds[5])  #42
        if verbose: print('seed:', seeds[i])
        q_params = 2 * pi * np.random.rand(design['layer_repe'] * args.n_qubits * 2)
        opt = qml.GradientDescentOptimizer(stepsize = lr)
        # opt = qml.AdamOptimizer(stepsize=0.01, beta1=0.9, beta2=0.99, eps=1e-08)

        for n in range(rate):
            q_params, prev_energy = opt.step_and_cost(cost_fn, q_params)
            if verbose: print(f"--- Step: {n}, Energy: {cost_fn(q_params):.8f}")
        energy.append(cost_fn(q_params))
    
    metrics = np.mean(energy)
    report = {'energy': metrics}
    print(metrics)

    # filename = 'models/' + net
    # with open(filename, 'wb') as file:
    #     pickle.dump([report, q_params], file)

    # with open('NoiseModel/fakekolkata.pkl', 'rb') as file:
    #     noise_model = pickle.load(file)
    # import qiskit_aer.noise as noise
    # noise_model1 = noise.NoiseModel()
    # noise_modelreal = noise_model1.from_dict(noise_model)
    # dev = qml.device('qiskit.aer', wires=args.n_qubits,  noise_model=noise_modelreal)
    # @qml.qnode(dev)
    # def cost_noise(theta):
    #     quantum_net(theta, design)
    #     return qml.expval(hamiltonian)    
    # print("Noise:", cost_noise(q_params))

    return report

def mask(net, positions = None):
    if positions:
         single = positions[0]
         double = positions[1]
         for i in single:
              net[i] = 'n'
         for i in double:
              net[12 + i] = 'n'
    else:
        mask = [random.sample(range(1, 12), 3), random.sample(range(12, 24), 10)]
        rz = mask[1][-1]        
        for i in range(2):
            for j in range(len(mask[i])):
                net[mask[i][j]] = 'n'
        net[rz] = rz - 12
    return net

def search(hamiltonian, train_space, index, size):
    filename = 'train_results_{}.csv'.format(index)
    if os.path.isfile(filename) == False:
        with open(filename, 'w+', newline='') as res:
                writer = csv.writer(res)
                writer.writerow(['Num', 'sample_id', 'arch_code', 'Energy'])

    csv_reader = csv.reader(open(filename))
    i = len(list(csv_reader)) - 1
    j = index * size + i

    while len(train_space) > 0:
        net = train_space[i]
        net = mask(net)
        print('Net', j, ":", net)
        design = translator(net)       
        report = chemistry(hamiltonian, design, net2str(net))       

        with open(filename, 'a+', newline='') as res:
            writer = csv.writer(res)           
            metrics = report['energy']
            writer.writerow([i, j, net, metrics])
        j += 1
        i += 1

def run(net, rate=None):
    with open('data/OHhamiltonian', 'rb') as outfile:
        hamiltonian = pickle.load(outfile)
    design = translator(net)
    net = net2str(net)   
    report = chemistry(hamiltonian, design, net, rate)

if __name__ == '__main__':

    with open('data/OHhamiltonian', 'rb') as outfile:
        hamiltonian = pickle.load(outfile)
          
    train_space = []
    filename = 'data/train_space_1'

    with open(filename, 'rb') as file:
        train_space = pickle.load(file)
    
    model_path = 'models'
    if os.path.exists(model_path) == False:
        os.makedirs(model_path)

    num_processes = 10
    size = int(len(train_space) / num_processes)
    space = []
    for i in range(num_processes):
        space.append(train_space[i*size : (i+1)*size]) 
    
    with mp.Pool(processes = num_processes) as pool:
        pool.starmap(search, [(hamiltonian, space[i], i, size) for i in range(num_processes)])
    
    

    # net = [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 3, 7, 6, 3, 1, 1, 10, 6, 5, 6, 6, 7]
    # mask = [random.sample(range(1, 12), 6), random.sample(range(12, 24), 6)]
    # for i in range(2):
    #      for j in range(len(mask[i])):
    #           net[mask[i][j]] = 'n'
    # print(net)
    # net = [0, 0, 0, 0, 1, 'n', 0, 1, 'n', 'n', 0, 'n', 'n', 'n', 6, 3, 'n', 1, 'n', 7, 'n', 6, 'n', 7]
    # design = translator(net)
    # net = net2str(net)   
    # report = chemistry(hamiltonian, design, net, 'print')
        
    # with open('models/'+ net, 'rb') as file:
    #     a = pickle.load(file)
    