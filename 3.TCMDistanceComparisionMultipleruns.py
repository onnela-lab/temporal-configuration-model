import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import least_squares
from scipy.stats import beta
from AddonTCM import indicator, modify_network, standardize_size_function, total_variance_distance_function, mysummaryfunction, degreedis_function, plot_degree_dist,  Hellinger_distance_function
from scipy.stats import truncnorm
bt = pd.read_csv('bt_symmetric.csv', header = 0, names = ['timestamp','user_a', 'user_b', 'rssi'])
import time
start_time = time.time()
###########################
def construct_daily_nets(input_file):
    """Read input file and return a dictionary of graphs where edge weight "pings" counts
    the number of times each pair has been observed.
    
    Data format and example:
        timestamp,user_a,user_b,rssi
        0,0,-1,0
        0,1,-1,0
        0,2,-1,0
    
    Some important points:
        -Time is quantized into 5-minute bins and timestamps are reported in the multiples of 300 seconds
        -In bins where userA was actively scanning, but found no other Bluetooth devices in proximity, 
         reported the alter ID as −1 and the received signal strength as 0
        -In bins where userA discovered other Bluetooth devices but not other study participants,
         reported the alter ID as −2 and the highest received signal strength measured.
"""
    counter = 0
    Gs = {}
    stats = {}
    for line in open(input_file):
        counter += 1
        # skip the first (header) line
        if counter > 1:
            # extract input
            line = line.rstrip().split(",")
            timestamp = int(line[0])
            user_a = int(line[1])
            user_b = int(line[2])
            rssi = int(line[3])
            day = timestamp // (24 * 60 * 60)

            # keep statistics for each day
            if day not in stats:
                stats[day] = {"pings":1, "empty":0, "other":0}
            else:
                stats[day]["pings"] += 1

            # create a graph for each day
            if day not in Gs:
                Gs[day] = nx.Graph()
            
            # if user_b is a person in the study, such that user_b >=0, update network
            if user_b >= 0:
                # construct daily graph if RSSI > -75
                if rssi >= -77.5:
                    # create new edge if needed
                    if not Gs[day].has_edge(user_a, user_b):
                        Gs[day].add_edge(user_a, user_b, pings = 1)
                    # update weight of existing edge
                    else:
                        weight = Gs[day].edges[user_a, user_b]["pings"]
                        Gs[day].add_edge(user_a, user_b, pings = weight + 1)
            
            # if user b is not a person in the study, update stats
            else:
                if user_b == -1:
                    stats[day]["empty"] += 1
                elif user_b == -2:
                    stats[day]["other"] += 1
                
    return (Gs, stats)

input_file = "bt_symmetric.csv"
(Gs, stats) = construct_daily_nets(input_file)


Gs_empirical = Gs

####Standardized all empirical net for same size
edgeslist = []
for i in range(len(Gs_empirical)):
    Gs_temp = Gs_empirical[i]
    edges_tmp = list(Gs_temp.edges())
    edgeslist  = edgeslist +  edges_tmp
    
edgeslist1 = list(set(edgeslist)) #get a unique list

edge1 = [edge[0] for edge in edgeslist1] #all first elements of edges
edge2 = [edge[1] for edge in edgeslist1] #all second elements of edges   

np.max(edge1)
np.max(edge2)
###############refine the network to avoid noise, wrong data, node>705
edgeslist1[0]
edgeslist1[1]

edge1wrong = []
for i in range(len(edge1)):
    h = edge1[i]
    if h>705:
        edge1wrong = edge1wrong + [i]
        
edge2wrong = []
for i in range(len(edge2)):
    h = edge2[i]
    if h>705:
        edge2wrong = edge2wrong + [i]        
        
edgewrong = edge1wrong + edge2wrong

edge1 = np.delete(edge1,edgewrong)
edge2 = np.delete(edge2,edgewrong)

###############
edgeslist1[0]
n1 = np.max(edge1)

n2 = np.max(edge2)
n = int(np.max([n1,n2])+1) # standardized number of nodes, add 1 to make matrix in the right size needed Python
adjMatrix = np.zeros((n,n))

##Build each daily network based on the benchmark size n of all common networks
def adjacencymatrixbyday(Gs,i,n):
    # define a function to extract the adjacency matrix at day i
    adjMatrix_dayi = np.zeros((n,n))
    Gs_temp = Gs[i]
    edges_tmp = list(Gs_temp.edges())
    edge1_dayi = [edge[0] for edge in edges_tmp] #all first elements of edges
    edge2_dayi = [edge[1] for edge in edges_tmp] #all
    edge1wrong = []
    for i in range(len(edge1_dayi)):
        h = edge1_dayi[i]
        if h>705:
            edge1wrong = edge1wrong + [i]
            
    edge2wrong = []
    for i in range(len(edge2_dayi)):
        h = edge2_dayi[i]
        if h>705:
            edge2wrong = edge2wrong + [i]        
            
    edgewrong = edge1wrong + edge2wrong

    edge1_dayi = np.delete(edge1_dayi,edgewrong)
    edge2_dayi = np.delete(edge2_dayi,edgewrong)
    for i in range(len(edge1_dayi)):
        u = edge1_dayi[i]
        v = edge2_dayi[i]
        adjMatrix_dayi[u][v] = 1
        
    adjMatrix_dayi = adjMatrix_dayi.transpose() + adjMatrix_dayi   
    return(adjMatrix_dayi)


##2. Save adjacency matrix in a list
mylist = list()
Gs_empirical_standardized = {}

for i in range(len(Gs_empirical)):
    adjacency_tmp = adjacencymatrixbyday(Gs_empirical,i,n)
    tmp_network = nx.from_numpy_array(adjacency_tmp)
    Gs_empirical_standardized[i] = tmp_network

def indicator_squarematrix(A):
    for i in range(len(A[1,:])):
        for j in range(len(A[1,:])):
            if A[i,j] != 0:
                A[i,j] = 1
    return(A)
################

Gs_empirical_standardized_MA = {}    
blockMA = 7
for i in range(4):
    tmp_matrix1 = np.zeros((n,n))
    j1 = i*blockMA 
    j2 = (i+1)*blockMA
    for j in range(j1,j2):
        adjacency_tmp = adjacencymatrixbyday(Gs_empirical,j,n)
        tmp_matrix1 = tmp_matrix1 + adjacency_tmp
        tmp_matrix = indicator_squarematrix(tmp_matrix1)
        tmp_network = nx.from_numpy_array(tmp_matrix)
    Gs_empirical_standardized_MA[i] = tmp_network   
    
#######check number edges
for i in range(4):
    G_tmp_len = Gs_empirical_standardized_MA[i]
    

    
G1 = Gs_empirical_standardized_MA[0]
G2 = Gs_empirical_standardized_MA[1]
G3 = Gs_empirical_standardized_MA[2]


week1adjacency1 = nx.to_numpy_array(G1)
week2adjacency1 = nx.to_numpy_array(G2)
week3adjacency1 = nx.to_numpy_array(G3)
    
adjacency12 = week2adjacency1*week1adjacency1

 
adjacency123 = adjacency12*week3adjacency1
    
phat = sum(sum(adjacency12))/sum(sum(week1adjacency1))          
  
p2hat = sum(sum(adjacency123))/sum(sum(week1adjacency1))          
  

###############################
nrep = 100
dmat_totalvariance = np.zeros((nrep,12))
dmat_Hellinger = np.zeros((nrep,12))
#############################
for realization in range(nrep):
      # loop number
    
    ##################3. Estimation   ####################
    ##Model 1 with phat for all persistent rate the same###########
    
    persistence_matrix1 = np.ones((n,n))*phat

    #########Serie of network predicted based on Model 1
    Gs_M1 = {}
    Gs_M1[0] = G1 #start from the first one and predict
    

    for day in range(1,len(Gs_empirical_standardized_MA)):
        G_tmp = Gs_M1[day-1]
        tmp = modify_network(G_tmp, persistence_matrix1)
        Gs_M1[day] = tmp


    M1 = phat
    M2 = p2hat

    ahat =(M1*M2-M1**2)/(M1**2-M2)
    bhat = (1-M1)*(M2-M1)/(M1**2-M2)


    b_dist_M2 = beta(a=ahat, b=bhat)
    edgeweights = b_dist_M2.rvs(n*n)
    persistence_matrix2 = edgeweights.reshape(n,n)
    
    ####MODEL 2

    Gs_M2 = {}
    Gs_M2[0] = G1
    

    for day in range(1,len(Gs_empirical_standardized_MA)):
        G_tmp = Gs_M2[day-1]
        tmp = modify_network(G_tmp, persistence_matrix2)
        Gs_M2[day] = tmp

    
    ############MODEL 3
    M1_M3 = phat**.5
    M2_M3 = p2hat**.5
    ahat_M3 =(M1_M3*M2_M3 - M1_M3**2)/(M1_M3**2 - M2_M3)
    bhat_M3 = (1 - M1_M3)*(M2_M3 - M1_M3)/(M1_M3**2 - M2_M3)

    b_dist_M3 = beta(a=ahat_M3, b=bhat_M3)
    nodeweights = b_dist_M3.rvs(n)
       
    
    persistence_matrix3 = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            persistence_matrix3[i,j]=nodeweights[i]*nodeweights[j]
            
    #######################
    Gs_M3 = {}
    Gs_M3[0] = G1
    

    for day in range(1,len(Gs_empirical_standardized_MA)):
        G_tmp = Gs_M3[day-1]
        tmp = modify_network(G_tmp, persistence_matrix3)
        Gs_M3[day] = tmp

        
    #######MODEL 4 (OR MODEL 0) Random configuration
    Gs_M4 = {}
    Gs_M4[0] = G1
    
    persistence_matrix4 = np.zeros((n,n))
    
    for day in range(1,len(Gs_empirical_standardized_MA)):
        G_tmp = Gs_M4[day-1]
        tmp = modify_network(G_tmp, persistence_matrix4)
        Gs_M4[day] = tmp

    
    for network_index in range(1,4):
       
        degrees_network_index = [Gs_empirical_standardized_MA[network_index].degree(n) for n in Gs_empirical_standardized_MA[network_index].nodes()]
        h_network_index = degreedis_function(degrees_network_index) 
        h_network_index_dis = h_network_index/sum(h_network_index)
        
        degrees_network_index_M1 = [Gs_M1[network_index].degree(n) for n in Gs_M1[network_index].nodes()]
        h_network_index_M1 = degreedis_function(degrees_network_index_M1) 
        h_network_index_M1_dis = h_network_index_M1/sum(h_network_index_M1)
        
        degrees_network_index_M2 = [Gs_M2[network_index].degree(n) for n in Gs_M2[network_index].nodes()]
        h_network_index_M2 = degreedis_function(degrees_network_index_M2) 
        h_network_index_M2_dis = h_network_index_M2/sum(h_network_index_M2)
        
        degrees_network_index_M3 = [Gs_M3[network_index].degree(n) for n in Gs_M3[network_index].nodes()]
        h_network_index_M3 = degreedis_function(degrees_network_index_M3) 
        h_network_index_M3_dis = h_network_index_M3/sum(h_network_index_M3)
        
        degrees_network_index_M4 = [Gs_M4[network_index].degree(n) for n in Gs_M4[network_index].nodes()]
        h_network_index_M4 = degreedis_function(degrees_network_index_M4) 
        h_network_index_M4_dis = h_network_index_M4/sum(h_network_index_M4)
        
        ##total variance distance
        d1_totalvariance = total_variance_distance_function(h_network_index_dis,h_network_index_M1_dis)
    
        d2_totalvariance = total_variance_distance_function(h_network_index_dis,h_network_index_M2_dis)
    
        d3_totalvariance = total_variance_distance_function(h_network_index_dis,h_network_index_M3_dis)
    
        d4_totalvariance = total_variance_distance_function(h_network_index_dis,h_network_index_M4_dis)
        ###Hellinger distance
        
        d1_Hellinger = Hellinger_distance_function(h_network_index_dis,h_network_index_M1_dis)
    
        d2_Hellinger = Hellinger_distance_function(h_network_index_dis,h_network_index_M2_dis)
    
        d3_Hellinger = Hellinger_distance_function(h_network_index_dis,h_network_index_M3_dis)
    
        d4_Hellinger = Hellinger_distance_function(h_network_index_dis,h_network_index_M4_dis)
        
        i1 = (network_index-1)*4
        i2 = (network_index)*4 
        dmat_totalvariance[realization,i1:i2] = [d1_totalvariance,d2_totalvariance,d3_totalvariance,d4_totalvariance]
        dmat_Hellinger[realization,i1:i2] = [d1_Hellinger,d2_Hellinger,d3_Hellinger,d4_Hellinger]
       

def outputextract_function(dmat):
    outputmat = np.zeros((8,4))
    d1mat = dmat[:,0:4]
    d2mat = dmat[:,4:8]
    d3mat = dmat[:,8:12]
    davemat = (d1mat + d2mat + d3mat)/3
    K = 3 #digits round
    m1 = np.round(np.mean(d1mat,axis=0),K)
    sd1 = np.round( np.std(d1mat,axis=0),K)
    m2 = np.round(np.mean(d2mat,axis=0),K)
    sd2 = np.round( np.std(d2mat,axis=0),K)
    m3 = np.round(np.mean(d3mat,axis=0),K)
    sd3 = np.round( np.std(d3mat,axis=0),K)
    m_ave = np.round(np.mean(davemat,axis=0),K)
    sd_ave = np.round( np.std(davemat,axis=0),K)
    outputmat[0,:] = m1
    outputmat[1,:] = sd1
    outputmat[2,:] = m2
    outputmat[3,:] = sd2
    outputmat[4,:] = m3
    outputmat[5,:] = sd3
    outputmat[6,:] = m_ave
    outputmat[7,:] = sd_ave
    return (outputmat)

##################Compare distance of degree distribution
totalvariance_dis = outputextract_function(dmat_totalvariance)
print(totalvariance_dis)

#############
Hellinger_dis = outputextract_function(dmat_Hellinger)
print(Hellinger_dis)
################################
#make latex table to report

distance_report = np.zeros((4,4))
distance_report[0,0:2] = list(totalvariance_dis[6:8,3]) #random
distance_report[0,2:4] = list(Hellinger_dis[6:8,3]) #random

distance_report[1,0:2] = list(totalvariance_dis[6:8,0])
distance_report[1,2:4] = list(Hellinger_dis[6:8,0])

distance_report[2,0:2] = list(totalvariance_dis[6:8,1])
distance_report[2,2:4] = list(Hellinger_dis[6:8,1])

distance_report[3,0:2] = list(totalvariance_dis[6:8,2])
distance_report[3,2:4] = list(Hellinger_dis[6:8,2])
distance_report
import array_to_latex as a2l

mytable = a2l.to_ltx(distance_report, frmt = '{:6.3f}', arraytype = 'array')
mytable


#############
time.time() - start_time
############Plotting the degree distribution 

G1 = Gs_empirical_standardized_MA[0]
G2 = Gs_empirical_standardized_MA[1]
G3 = Gs_empirical_standardized_MA[2]
G4 = Gs_empirical_standardized_MA[3]


degrees_network1 = [G1.degree(n) for n in G1.nodes()]
h_network1 = degreedis_function(degrees_network1) 
h_network1_degree = h_network1/sum(h_network1)
########
degrees_network2 = [G2.degree(n) for n in G2.nodes()]
h_network2 = degreedis_function(degrees_network2) 
h_network2_degree = h_network2/sum(h_network2)
#########
degrees_network3 = [G3.degree(n) for n in G3.nodes()]
h_network3 = degreedis_function(degrees_network3) 
h_network3_degree = h_network3/sum(h_network3)
##############
degrees_network4 = [G4.degree(n) for n in G4.nodes()]
h_network4 = degreedis_function(degrees_network4) 
h_network4_degree = h_network4/sum(h_network4)
highest_len = int(max(len(h_network1_degree),len(h_network2_degree),len(h_network3_degree),len(h_network4_degree) )+1)


fig, ((axs1,axs2),(axs3,axs4)) = plt.subplots(nrows=2, ncols=2)

axs1.plot (h_network1_degree, linewidth=1)
axs1.legend ([ "$G_1$"], prop={'size': 7})
axs1.set_ylabel ("Probability")
xname = list(range(0,highest_len,20))
axs1.set_xticks(xname)
axs1.axes.xaxis.set_ticklabels([])
axs2.plot (h_network2_degree, linewidth=1)
axs2.legend ([ "$G_2$"], prop={'size': 7})
xname = list(range(0,highest_len,20))
axs2.set_xticks(xname)
axs2.axes.xaxis.set_ticklabels([])
axs2.axes.yaxis.set_ticklabels([])


axs3.plot (h_network3_degree, linewidth=1)
axs3.legend ([ "$G_3$"], prop={'size': 7})
axs3.set_xlabel ("Node degree")
axs3.set_ylabel ("Probability")
xname = list(range(0,highest_len,20))
axs3.set_xticks(xname)
axs4.plot (h_network4_degree, linewidth=1)
axs4.legend ([ "$G_4$"], prop={'size': 7})
axs4.set_xlabel ("Node degree")
xname = list(range(0,highest_len,20))
axs4.set_xticks(xname)
axs4.axes.yaxis.set_ticklabels([])

