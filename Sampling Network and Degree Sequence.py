# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 21:59:27 2019

@author: wany105
"""
import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.special import factorial
from scipy.stats import poisson
from scipy.optimize import curve_fit
from collections import defaultdict
import heapq

def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

def Degree_Sequence(A): #Calculate the number of each possible degree value of nodes in the network, Input: Adjacent Matrix, Output: the range of degree value, the number of each degree value, degree of each node
    Degree = np.sum(A, axis = 1).astype(np.int32)
    Max_A = np.max(Degree)
    Min_A = np.min(Degree)
    NumA = np.arange(Min_A, Max_A + 1)
    NumDegree = []
    for i in range(Min_A, Max_A + 1):
        NumDegree.append(np.sum(Degree == i))
    return NumA, NumDegree, Degree

def Visual_Degree_Distribution(NumA, NumDegree, color, lwidth): #Input: the range and number of each degree value, lwidth: figure parameter
    fig = plt.figure()
    plt.plot(NumA, NumDegree/np.sum(NumDegree), color, linewidth = lwidth)
    plt.xlabel('Degree Value')
    plt.ylabel('The Probability of certain Degree Value')
    plt.show()
    
def Degree_Distribution_Sampling(Network):
    fig = plt.figure()
    NumA, NumDegree, Degree = Degree_Sequence(Network.ImitateAdj)
    entries, bin_edges, patches = plt.hist(Degree, bins=(max(Degree) - min(Degree)), range=[min(Degree)-1, max(Degree)+1], normed=True, label = 'real data')
    bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
    parameters, cov_matrix = curve_fit(poisson, bin_middles, entries)
    Network.ParameterLambda = parameters
    Network.ParameterCov_matrix = cov_matrix
    x_plot = np.arange(min(Degree)-2, max(Degree)+2)
    plt.plot(x_plot, poisson(x_plot, *parameters), 'r-', lw = 2, label = 'fitted data')
    plt.xlabel("The degree value of each vertex")
    plt.ylabel("The probability of each degree value")
    plt.legend(framealpha=1, frameon=False, loc = 'upper right')
    plt.show()

def Bipartite(Network, PoiSample, HighLevel, LowLevel): #Input: Distance Matrix of the network, PoiSample: degree sequence sampling from poisson distribution, A:Adjacent Matrix, NodeNum
    flag = 1
    while(flag):
        for i in range(len(LowLevel)):
            TempDistance = list(Network.DistanceMatrix[LowLevel[i], HighLevel])
            SortIndex = np.argsort(TempDistance)
            p = 0
            while(PoiSample[SortIndex[p]] == 0):
                p += 1
            Network.AdjMatrix[SortIndex[p] + HighLevel[0]][LowLevel[i]] = 1
            PoiSample[SortIndex[p]] -= 1
            if(sum(PoiSample) == 0):
                flag = 0
                break
        if(flag == 0):
            break

def Degree_Visualization(Network):
    fig = plt.figure()
    SupplyTranDegree = Network.SupplyTranPoiSample
    entries, bin_edges, patches = plt.hist(SupplyTranDegree, bins=(max(SupplyTranDegree) - min(SupplyTranDegree)), range=[min(SupplyTranDegree)-1, max(SupplyTranDegree)+1], normed=True, label = 'sample data', color='orange')
    plt.xlabel("The degree value of each vertex")
    plt.ylabel("The probability of each degree value")
    plt.legend(framealpha = 1, frameon = False, loc = 'upper right')
    plt.show()
    
    TranDemandDegree = Network.TranDemandPoiSample
    entries, bin_edges, patches = plt.hist(TranDemandDegree, bins=(max(TranDemandDegree) - min(TranDemandDegree)), range=[min(TranDemandDegree)-1, max(TranDemandDegree)+1], normed=True, label = 'sample data', color='orange')
    plt.xlabel("The degree value of each vertex")
    plt.ylabel("The probability of each degree value")
    plt.legend(framealpha = 1, frameon = False, loc = 'upper right')
    plt.show()


for i in range(NetNum):
    Network = InfrasDict['Objective'][i]
    for j in range(len(Network.NodeLocGeo)):
        for k in range(j):
            Distance = np.sqrt(np.square(Network.NodeLocGeo[j][0] - Network.NodeLocGeo[k][0]) + np.square(Network.NodeLocGeo[j][1] - Network.NodeLocGeo[k][1]))
            Network.DistanceMatrix[j][k] = Network.DistanceMatrix[k][j] = Distance
    Network.SupplyTranDist = Network.DistanceMatrix[Network.SupplyIniNum:Network.TranIniNum, Network.TranIniNum:Network.DemandIniNum]
    Network.TranDemandDist = Network.DistanceMatrix[Network.TranIniNum:Network.DemandIniNum, Network.DemandIniNum:Network.DemandEndNum + 1]
    
    Degree_Distribution_Sampling(Network)
    
    while(1):
        Network.SupplyTranPoiSample = -np.sort(-np.random.poisson(lam = Network.ParameterLambda, size = Network.SupplyNum))
        if(np.max(Network.SupplyTranPoiSample) <= Network.TranNum and sum(Network.SupplyTranPoiSample) >= Network.TranNum and np.count_nonzero(Network.SupplyTranPoiSample) == Network.SupplyNum):
            break
    Bipartite(Network, copy.copy(Network.SupplyTranPoiSample), Network.SupplyNodeSeries, Network.TranNodeSeries)
    while(1):
        Network.TranDemandPoiSample = -np.sort(-np.random.poisson(lam = Network.ParameterLambda, size = Network.TranNum))
        if(np.max(Network.TranDemandPoiSample) <= Network.DemandNum and sum(Network.TranDemandPoiSample) >= Network.DemandNum and np.count_nonzero(Network.TranDemandPoiSample) == Network.TranNum):
            break
    Bipartite(Network, copy.copy(Network.TranDemandPoiSample), Network.TranNodeSeries, Network.DemandNodeSeries)
    
    Network.SupplyTranAdj = Network.AdjMatrix[Network.SupplyIniNum:Network.TranIniNum, Network.TranIniNum:Network.DemandIniNum]
    Network.TranDemandAdj = Network.AdjMatrix[Network.TranIniNum:Network.DemandIniNum, Network.DemandIniNum:Network.DemandEndNum + 1]
    
    m = 300 #the number of segments on the link
    Network.LinkSegment(m, Network)
    Degree_Visualization(Network)
    