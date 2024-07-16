# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:16:06 2024
Function Addon for TCM
@author: PWR844
"""

import numpy as np
import random
import matplotlib.pyplot as plt

def mysummaryfunction(x):
    val1 = np.mean(x)
    val2 = np.std(x)
    val3 = np.median(x)
    val4 = np.min(x)
    val5 = np.max(x)
    val6 = np.sum(x)
    val7 = len(x)
    mydict = {"mean":val1, "std": val2, "median": val3, "min": val4,"max":val5 ,"sum": val6,
              "length":val7}
    return mydict



def indicator(x):
    if abs(x)>0:
        x=1
    return(x)

def modify_network(G, persistence_matrix):
    """
    Modify the network for the next time step.
    """

    # Make a copy of the old network & create an empty stubs list
    H = G.copy()
    stubs = []
    broken = []

    # For every edge in edgelist, determine if will persist to next time step
    for edge in G.edges():
        p_pair = persistence_matrix[edge[0],edge[1]]
        if np.random.uniform() <= (1-p_pair):
            H.remove_edge(edge[0], edge[1])
            stubs.extend(edge)
            broken.append(edge)

    # Reconnect stubs at random
    random.shuffle(stubs)
    it = iter(stubs)
    new_edges = list(zip(it, it))
    H.add_edges_from(new_edges)

    return H

def degreedis_function(degreesequence):
    n = int(np.max(degreesequence)+1)
    degreecount = np.zeros(n)

    for i in range(n):
        for j in range(len(degreesequence)):
            if degreesequence[j] == i:
                degreecount[i] = degreecount[i]+1
    return(degreecount)           
        
def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees)
    plt.show()

def standardize_size_function(vec1, vec2):
    k = int(max(len(vec1),len(vec2)))
    n1 = int(len(vec1))
    n2 = int(len(vec2))
    tmp = np.zeros(k)
    vectorstandardized = np.zeros((2,k))
    if n1<k:
        tmp[0:n1] = vec1
        tmp[n1:k] = np.zeros(k-n1)
        vec1 = tmp
        
    if n2<k:
        tmp[0:n2] = vec2
        tmp[n2:k] = np.zeros(k-n2)
        vec2 = tmp
    vectorstandardized[0,] = vec1
    vectorstandardized[1,] = vec2
    return(vectorstandardized)





def total_variance_distance_function(vec1,vec2):
    standardizeddegree = standardize_size_function(vec1, vec2)
    k = int(len(standardizeddegree[0,:]) )     
    distance = 0
    for i in range(k):
        d = abs( standardizeddegree[0,i]-standardizeddegree[1,i])
        distance = distance + d
    return(distance/2)


def Hellinger_distance_function(vec1,vec2):
    standardizeddegree = standardize_size_function(vec1, vec2)
    k = int(len(standardizeddegree[0,:]) )     
    distance = 0
    for i in range(k):
        d = (standardizeddegree[0,i]**.5 - standardizeddegree[1,i]**.5)**2
        distance = distance + d
    return((distance/2)**.5)

def Bhattacharyya_distance_function(vec1,vec2):
    standardizeddegree = standardize_size_function(vec1, vec2)
    k = int(len(standardizeddegree[0,:]) )     
    distance = 0
    for i in range(k):
        d = (standardizeddegree[0,i]*standardizeddegree[1,i])**.5
        distance = distance + d
    return(-np.log(distance))