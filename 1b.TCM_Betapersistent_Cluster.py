import sys
para1 = int(sys.argv[1]) #N network size
para2 = int(sys.argv[2]) #T time step
para3 = int(sys.argv[3]) #replication



import numpy as np
import networkx as nx
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



parameter = [1,4]

#parameter = [3,2]


a = parameter[0]
b = parameter[1]
meank = 6
N = para1
timestep = para2

# Draw TCM persistence probabilities and set up matrix, window of 2
TCM_persistence =[]
timestephalf = int(timestep/2)
for i in range(timestephalf):
    betap = np.random.beta(a, b, size = int(N*(N+1)/2)) # random generate from beta distribution
    ps = np.zeros((N,N))
    ps[np.tril_indices(N)] = betap
    ps = ps + np.tril(ps, -1).T
    np.fill_diagonal(ps, 0)
    TCM_persistence.append(ps)
    TCM_persistence.append(ps)
    
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
        persistence = TCM_persistence[i]
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

  

M1vector = np.zeros(timestep)

for t in range(timestep):
    net = mytcm[t]
    nedge = len(net)
    broken = len(brokenedges[t])
    M1vector[t] = (nedge-broken)/nedge


timestephalf = int(timestep/2)


##Estimate use even positions
M2vector_even = np.zeros(timestephalf)
for t in range(timestephalf):
    t1 = 2*t
    t2 = 2*t+1
    net = mytcm[t1]
    broken1 = brokenedges[t1]
    broken2 = brokenedges[t2]
    brokentotal = broken1 + broken2
    ##find which broke edges belong to the network
    brokentotalset = set(brokentotal)
    broken_2step = [x for x in net if x in brokentotalset]
    M2vector_even[t] = (len(net) - len(broken_2step))/len(net)
    

##Estimating moments
M1vector_even = M1vector[::2] #extract only even element
phat = np.sum(M1vector_even)/len(M1vector_even) 

p2hat = (np.sum(M2vector_even))/(timestephalf)

truthmoments = firstandsecond(parameter)

results = [phat, p2hat, M1vector_even[0], M2vector_even[0], truthmoments[0], truthmoments[1],parameter[0], parameter[1]]


filename = "WBetarandompersistentestimationvector-N"+str(para1)+"-timestep"+str(para2)+"-rep"+str(para3)+".txt"
np.savetxt(filename, results)



time.time() - start
