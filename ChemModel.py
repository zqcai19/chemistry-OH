import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
from math import pi
from Arguments import Arguments
args = Arguments()

def translator(net):
    assert type(net) == type([])
    updated_design = {}

    # r = net[0]
    q = net[0:12]
    # c = net[8:15]
    p = net[12:24]

    # num of layer repetitions
    layer_repe = 1
    updated_design['layer_repe'] = layer_repe

    # categories of single-qubit parametric gates
    for i in range(args.n_qubits):
        if q[i] == 0:
            category = 'Rx'
        elif q[i] == 1:
            category = 'Ry'
        else:
            category = 'None'        
        updated_design['rot' + str(i)] = category

    # categories and positions of entangled gates
    for j in range(args.n_qubits):
        # if c[j] == 0:
        #     category = 'IsingXX'
        # else:
        #     category = 'IsingZZ'
        if p[j] == j:
            updated_design['enta' + str(j)] = (12)
        elif p[j] == 'n':
            updated_design['enta' + str(j)] = 'None'
        else:        
            updated_design['enta' + str(j)] = ([j, p[j]])

    updated_design['total_gates'] = len(q) + len(p)
    return updated_design

def quantum_net(q_params, design):
    
    current_design = design
    q_weights = q_params.reshape(current_design['layer_repe'], args.n_qubits, 2)
    for layer in range(current_design['layer_repe']):        
        for j in range(args.n_qubits):
            if current_design['rot' + str(j)] == 'Rx':
                qml.RX(np.pi/2, wires=j)
                qml.RZ(q_weights[layer][j][0], wires=j)
                qml.RX(np.pi/2, wires=j)
                
            elif current_design['rot' + str(j)] == 'Ry':
                # qml.RY(q_weights[layer][j][0], wires=j)
                qml.RX(np.pi/2, wires=j)
                qml.RZ(q_weights[layer][j][0], wires=j)
                qml.RX(np.pi/2, wires=j)
                
            
            if current_design['enta' + str(j)] != 'None':
                if current_design['enta' + str(j)] == 12:
                    qml.RZ(q_weights[layer][j][1], wires=j)
                else:           
                    # qml.IsingZZ(q_weights[layer][j][1], wires=current_design['enta' + str(j)])
                    qml.CNOT(wires=current_design['enta' + str(j)])
                    qml.RZ(q_weights[layer][j][1], wires=current_design['enta' + str(j)][1])



