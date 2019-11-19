# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:43:24 2019

@author: wany105
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 17:20:43 2019

@author: wany105
"""

## This file is to:
# Apply anneal simulation to locate the optimal solution of infrastructure locations in a single nework
# Anneal Simulation: Objective Function: Ensure the sum of the distance person belonging to a specific square travel to the nearest demand node is the smallest

import numpy as np
import math
import copy




#------------------------------------------------------------------------------Anneal Simulation Function
def cost(sol, lat, lon, PDsub, Type, LAT, LON):
    Sum_Cost = 0
    if(Type == 'Population'):
        for i in range(len(lat)-1):
            for j in range(len(lon)-1):
                temp_X = 0.5*(lat[i]+lat[i+1])
                temp_Y = 0.5*(lon[j]+lon[j+1])
                for k in range(len(sol)):
                    Dist = math.sqrt((temp_X - LAT[sol[k][0]])**2 + (temp_Y - LON[sol[k][1]])**2)
                    Min_Dist = math.inf
                    if(Dist < Min_Dist):
                        Min_Dist = Dist
                        index = k
                Sum_Cost += Min_Dist*PDsub[i][j]
        return Sum_Cost
    else:
        for i in range(len(lat)):
            for k in range(len(sol)):
                Dist = math.sqrt((lat[i] - LAT[sol[k][0]])**2 + (lon[i] - LON[sol[k][1]])**2)
                Min_Dist = math.inf
                if(Dist < Min_Dist):
                    Min_Dist = Dist
                    index = k
            Sum_Cost += Min_Dist
        return Sum_Cost


def neighbor(sol, LAT, LON):
    Index = np.random.randint(0, len(sol))
    Sol = copy.deepcopy(sol)
    while(1):
        Step = np.random.randint(1,5)
        Temp = np.random.randint(0,4)
        if(Temp == 0):
            Sol[Index][0] += Step
        elif(Temp == 1):
            Sol[Index][0] -= Step
        elif(Temp == 2):
            Sol[Index][1] += Step
        elif(Temp == 3):
            Sol[Index][1] -= Step
        if((Sol[Index][0] >= 0 and Sol[Index][0] <= (len(LAT)-1)) and (Sol[Index][1] >= 0 and Sol[Index][1] <= (len(LON)-1))):
            sol = Sol
            break
        else:
            Sol = copy.deepcopy(sol)
    return sol

def acceptance_probability(old_cost, new_cost, T):
    return math.exp((old_cost - new_cost)/T)

def anneal(sol, Type, lat, lon, PDsub, LAT, LON):
    old_cost = cost(sol, lat, lon, PDsub, Type, LAT, LON)
    Time = 1
    T = 1.0
    T_min = 0.1
    alpha = 0.1
    Cost_Iter = []
    while T > T_min:
        i = 1
        while i <= 1:
            new_sol = neighbor(sol, LAT, LON)
            new_cost = cost(new_sol, lat, lon, PDsub, Type, LAT, LON)
            ap = acceptance_probability(old_cost, new_cost, T)
            if(ap >= np.random.rand()):
                sol = new_sol
                old_cost = new_cost
                Cost_Iter.append(old_cost)
            print('Iteration {}, Temperature {}, Cost {}'.format(Time, T, old_cost))
            i += 1
            Time += 1

            
        T = T*alpha
    return sol, old_cost