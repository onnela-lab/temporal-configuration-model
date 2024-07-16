
import numpy as np
import networkx as nx
from scipy.optimize import least_squares
import time
import os
import random

start = time.time()


def firstandsecond(para):
    x =para[0]
    y = para[1]
    EX = x/(x+y)
    VX = x*y/(((x+y)**2)*(x+y+1))
    EX2 = VX+EX**2
    return([EX,EX2])




parameter = [4,1]


def momentandvariance(para):
    x =para[0]
    y = para[1]
    EX = x/(x+y)
    VX = x*y/(((x+y)**2)*(x+y+1))
    return([EX,VX])

momentandvariance(parameter)

firstandsecond(parameter)

a = parameter[0]
b = parameter[1]
meank = 20
N = 1000
timestep = 100


# Draw persistence probabilities and set up matrix
betap = np.random.beta(a, b, size = int(N*(N+1)/2)) # random generate from beta distribution

ps = np.zeros((N,N))
ps[np.tril_indices(N)] = betap
ps = ps + np.tril(ps, -1).T
np.fill_diagonal(ps, 0)
persistence = ps  


# For more complicated case, we can create function to solve system of equations as:
def equations(x, mom1, mom2):
    alpha, beta = x[0], x[1]
    eq1 = alpha/(alpha+beta) - mom1
    eq2 = (alpha*beta)/(((alpha+beta)**2)*(alpha+beta+1))+(alpha/(alpha+beta))**2 - mom2
    return (eq1, eq2)

# Create output object
output_array = np.zeros(5)


# Draw degree distribution
ERpk = np.random.poisson(meank, N)
# If odd number of stubs, remove one at random
if sum(ERpk) % 2 != 0:
    node = np.random.choice(range(N))
    ERpk[node] -=1

###Check modify network function
degree_seq = ERpk.tolist()
G = nx.configuration_model(degree_seq) #create a network by a configuration model

#define a function to remove self-loop edges and duplicate edges of G to make a graph H

def cleaningnetwork(mynet):
    A = nx.to_numpy_array(mynet) 
    np.fill_diagonal(A, 0) # REMOVE SELF LOOPS 
    A1 = np.triu(A,k=0) #Get lower triangular
    return (nx.from_numpy_array(A1))

G = cleaningnetwork(G) #clean the network before process further
H0 = G.copy()
##############
# Define a function to see the discrepancy
TCM = [list(G.edges())]
TCM1 = TCM.copy()
TCM_broken = []
TCM_nedges = [len(list(G.edges()))]
TCM_weights = [betap]
rate = np.zeros(int(N*(N+1)/2))



def TCMprocess(G,timestep):
    for i in range(timestep):
        stubs = []
        broken = []
        weights = []
        for edge_tmp in list(G.edges()):
            p_pair = persistence[edge_tmp[0],edge_tmp[1]]
            weights.append(p_pair)
            if np.random.uniform() <= (1-p_pair):
                stubs.extend(edge_tmp)
                broken.append(edge_tmp)
                G.remove_edge(edge_tmp[0], edge_tmp[1])
        random.shuffle(stubs)
        it = iter(stubs)
        new_edges = list(zip(it, it))
        G.add_edges_from(new_edges)
        G = cleaningnetwork(G)
        TCM.append(list(G.edges()))
        TCM_broken.append(broken)
        TCM_nedges.append(len(list(G.edges)))
        TCM_weights.append(weights)
    return (TCM,  TCM_broken) 



mytcm, brokenedges = TCMprocess(G,timestep)

  


B1 = TCM_weights[100]
A1 = TCM_weights[0]


import matplotlib.pyplot as plt

data = [A1, B1]
 
fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)
labels = ['First step', 'Last step']

 
# rectangular box plot
bplot1 = ax.boxplot(data, vert=0,          
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to la


plt.savefig('Compareweightsfirstvslaststep.pdf')

print(time.time() - start)


