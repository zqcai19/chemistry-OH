import pickle
import os
import random
import json
import csv
import numpy as np
import torch
from Node import Node
from ChemModel import translator
from schemes import chemistry


def num2ord(num):
    if num % 10 == 1:
        ord_str = str(num) + 'st'
    elif num % 10 == 2:
        ord_str = str(num) + 'nd'
    elif num % 10 == 3:
        ord_str = str(num) + 'rd'
    else:
        ord_str = str(num) + 'th'
    return ord_str


class MCTS:
    def __init__(self, search_space, tree_height, arch_code_len):
        assert type(search_space)    == type([])
        assert len(search_space)     >= 1
        assert type(search_space[0]) == type([])

        self.search_space   = search_space
        self.ARCH_CODE_LEN  = arch_code_len
        self.ROOT           = None
        self.Cp             = 0.5
        self.nodes          = []
        self.samples        = {}
        self.TASK_QUEUE     = []
        self.DISPATCHED_JOB = {}
        self.mae_list       = []
        self.JOB_COUNTER    = 0
        self.TOTAL_SEND     = 0
        self.TOTAL_RECV     = 0
        self.ITERATION      = 0
        self.MAX_MAEINV     = 0
        self.MAX_SAMPNUM    = 0
        self.sample_nodes   = []

        # initialize a full tree
        total_nodes = 2**tree_height - 1
        for i in range(1, total_nodes + 1):
            is_good_kid = False
            if (i-1) > 0 and (i-1) % 2 == 0:
                is_good_kid = False
            if (i-1) > 0 and (i-1) % 2 == 1:
                is_good_kid = True

            parent_id = i // 2 - 1
            if parent_id == -1:
                self.nodes.append(Node(None, is_good_kid, self.ARCH_CODE_LEN, True))
            else:
                self.nodes.append(Node(self.nodes[parent_id], is_good_kid, self.ARCH_CODE_LEN, False))

        self.ROOT = self.nodes[0]
        self.CURT = self.ROOT
        self.init_train()


    # def dump_all_states(self, num_samples):
    #     node_path = 'states/mcts_agent'
    #     with open(node_path+'_'+str(num_samples), 'wb') as outfile:
    #         pickle.dump(self, outfile)


    def reset_node_data(self):
        for i in self.nodes:
            i.clear_data()


    # def populate_training_data(self):
    #     self.reset_node_data()
    #     for k, v in self.samples.items():
    #         self.ROOT.put_in_bag(json.loads(k), v)


    def populate_prediction_data(self):
        self.reset_node_data()
        for k in self.search_space:
            self.ROOT.put_in_bag(k, 0.0)


    def predict_nodes(self):
        for i in self.nodes:
            i.predict()


    def check_leaf_bags(self):
        counter = 0
        for i in self.nodes:
            if i.is_leaf is True:
                counter += len(i.bag)
        assert counter == len(self.search_space)


    def reset_to_root(self):
        self.CURT = self.ROOT


    def print_tree(self):
        print('\n'+'-'*100)
        for i in self.nodes:
            print(i)
        print('-'*100)

def sampling_node(agent, nodes, dataset, iteration, verbose = None):
    leaf_nodes = []
    for i in agent.nodes:
        if i.is_leaf is True:
            leaf_nodes.append(i)
    print("there are {} leaf nodes in total".format(len(leaf_nodes)))
    energy_list = []
    for j in nodes:  # leaf nodes for sampling
        target_bin = leaf_nodes[j]       
        number = 100 if len(target_bin.bag) > 100 else len(target_bin.bag)
        sampled_arch_list = random.sample(list(target_bin.bag.keys()), number)        
        energy = []
        for sample_no in range(len(sampled_arch_list)):  # the number of nodes needed to be sampled
            sampled_arch = json.loads(sampled_arch_list[sample_no])  
            report = {'energy': dataset.get(str(sampled_arch))}
            # design = translator(sampled_arch)
            # print("translated to:\n{}".format(design))           
            # if str(sampled_arch) in dataset:
            #     report = {'energy': dataset.get(str(sampled_arch))}               
            # else:
            #     report = chemistry(design)
            if verbose:
                print("\nstart training:")
                print("\nsampled from node", j)
                print("sample no.{}".format(sample_no))
                print("sampled arch:", sampled_arch)
                print(report)
            metrics = report['energy']
            energy.append(metrics)
            # with open('results_sampling.csv', 'a+', newline='') as res:
            #     writer = csv.writer(res)                
            #     writer.writerow([j, sampled_arch, sample_no, metrics])           
        energy_list.append(np.mean(energy))
    print("\033[1;33;40mResult: {}\033[0m".format(energy_list))
    if os.path.isfile('results_sampling.csv') == False:
        with open('results_sampling.csv', 'w+', newline='') as res:
            writer = csv.writer(res)
            nodes.insert(0, 'iteration')
            writer.writerow(nodes)
    with open('results_sampling.csv', 'a+', newline='') as res:        
        energy_list.insert(0, iteration)
        writer = csv.writer(res)                       
        writer.writerow(energy_list)

if __name__ == '__main__':
    # set random seed
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)
    
    with open('data/chemistry_dataset', 'rb') as file:
        dataset = pickle.load(file)            
    
    state_path = 'states'
    files = os.listdir(state_path)
    if files:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(state_path, x)))
        node_path = os.path.join(state_path, files[-1])
        # node_path = 'states/mcts_agent_10000'
        with open(node_path, 'rb') as json_data:
            agent = pickle.load(json_data)
        with open('search_space_1', 'rb') as file:
            search_space_1 = pickle.load(file)
        agent.search_space+=search_space_1

        print("\nresume searching,", agent.ITERATION, "iterations completed before")
        print("=====>loads:", len(agent.nodes), "nodes")
        print("=====>loads:", len(agent.samples), "samples")
        print("=====>loads:", len(agent.search_space), "archs")
        
        
    print("\nclear the data in nodes...")
    agent.reset_node_data()
    print("finished")

    print("\npopulate prediction data...")
    agent.populate_prediction_data()
    print("finished")

    print("\npredict and partition nets in search space...")
    agent.predict_nodes()
    # agent.check_leaf_bags()
    print("finished")
    agent.print_tree()
       
    nodes = [0, 1, 4, 5, 8, 9, 14, 15]    
    sampling_node(agent, nodes, dataset, 1)
        
    
