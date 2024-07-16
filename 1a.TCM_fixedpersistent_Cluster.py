import sys
para1 = int(sys.argv[1])  #N network size
para2 = int(sys.argv[2])  #T timestep
para3 = int(sys.argv[3])  #replication

import numpy as np
import networkx as nx
import time
import random
start = time.time()
N = para1 #network size
timestep = para2 #time step
persistent_rate = .8
meank = 6 #mean degree

#Define a function to estimate the first moment

def firstmoment_estimation(G,t):
    phatvector = np.full(shape=t, fill_value=np.nan)
    for i in range(t):
        stubs = []
        broken = []
        n1 = len(list(G.edges())) #track number of edges before broke
        for edge_tmp in list(G.edges()):
            p_pair = persistent_rate
            if np.random.uniform() <= (1-p_pair):
                stubs.extend(edge_tmp)
                broken.append(edge_tmp)
                G.remove_edge(edge_tmp[0], edge_tmp[1])
        phat = (n1 - len(broken))/n1  #ratio of edges remaining
        phatvector[i] = phat
        random.shuffle(stubs)
        it = iter(stubs)
        new_edges = list(zip(it, it))
        G.add_edges_from(new_edges)
        G = cleaningnetwork(G)
    return (phatvector)





# Draw degree distribution
ERpk = np.random.poisson(meank, N)
# If odd number of stubs, remove one at random
if sum(ERpk) % 2 != 0:
    node = np.random.choice(range(N))
    ERpk[node] -=1
    
###Generate the network
degree_seq = ERpk.tolist()
G = nx.configuration_model(degree_seq) #create a network by a configuration model

#Define a function to remove self-loop edges and duplicate edges of G to make a graph H

def cleaningnetwork(mynet):
    A = nx.to_numpy_array(mynet) 
    np.fill_diagonal(A, 0) # REMOVE SELF LOOPS 
    A1 = np.triu(A,k=0) #Get lower triangular
    return (nx.from_numpy_array(A1))

G = cleaningnetwork(G) #clean the network before process further
  

phatestimation_vector = firstmoment_estimation(G,timestep)
results = [np.mean(phatestimation_vector),  phatestimation_vector[0],persistent_rate] 
# check out mean, first component, and truth

filename = "WFixedconsistentfirstmomentestimationvector-N"+str(para1)+"-timestep"+str(para2)+"-rep"+str(para3)+".txt"
np.savetxt(filename, results)


end = time.time()

print(f'Total Time: {end-start} seconds')
