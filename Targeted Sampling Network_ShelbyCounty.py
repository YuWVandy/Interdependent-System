# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:20:36 2019

@author: wany105
"""
import pandas as pd
import matplotlib.pyplot as plt

WN = pd.read_excel (r'C:\Users\wany105\OneDrive - Vanderbilt\Research\ShelbyCounty_DataRead\WaterNodes.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'
WE = pd.read_excel (r'C:\Users\wany105\OneDrive - Vanderbilt\Research\ShelbyCounty_DataRead\WaterEdges.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'
PN = pd.read_excel (r'C:\Users\wany105\OneDrive - Vanderbilt\Research\ShelbyCounty_DataRead\PowerNodes.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'
PE = pd.read_excel (r'C:\Users\wany105\OneDrive - Vanderbilt\Research\ShelbyCounty_DataRead\PowerEdges.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'
GN = pd.read_excel (r'C:\Users\wany105\OneDrive - Vanderbilt\Research\ShelbyCounty_DataRead\GasNodes.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'
GE = pd.read_excel (r'C:\Users\wany105\OneDrive - Vanderbilt\Research\ShelbyCounty_DataRead\GasEdges.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'


NumberWN = len(WN['X'])
NumberPN = len(PN['X'])
NumberGN = len(GN['X'])

fig = plt.figure()

for i in range(NumberWN):
    if(WN['NODE CLASS'][i] == 'Pump Stations'):
        B_S = plt.scatter(WN['X'][i], WN['Y'][i], alpha = 0.5, s = 50, c = 'blue', marker = 's')
    elif(WN['NODE CLASS'][i] == 'Storage Tanks'):
        B_O = plt.scatter(WN['X'][i], WN['Y'][i], alpha = 0.5, s = 50, c = 'blue', marker = 'o')
    elif(WN['NODE CLASS'][i] == 'Delivery Nodes'):
        B_V = plt.scatter(WN['X'][i], WN['Y'][i], alpha = 0.5, s = 50, c = 'blue', marker = 'v')

for i in range(NumberPN):
    if(PN['NODE CLASS'][i] == 'Gate Station'):
        R_S = plt.scatter(PN['X'][i], PN['Y'][i], alpha = 0.5, s = 50, c = 'R', marker = 's')
    elif(PN['NODE CLASS'][i] == 'Intersection Point'):
        R_O = plt.scatter(PN['X'][i], PN['Y'][i], alpha = 0.5, s = 50, c = 'R', marker = 'o')
    elif(PN['NODE CLASS'][i] == '12kV Substation' or '23kV Substation'):
        R_V = plt.scatter(PN['X'][i], PN['Y'][i], alpha = 0.5, s = 50, c = 'R', marker = 'v')

for i in range(NumberGN):
    if(GN['NODE CLASS'][i] == 'Gate Station'):
        Y_S = plt.scatter(GN['X'][i], GN['Y'][i], alpha = 0.5, s = 50, c = 'Y', marker = 's')
    elif(GN['NODE CLASS'][i] == 'Regulator Station'):
        Y_O = plt.scatter(GN['X'][i], GN['Y'][i], alpha = 0.5, s = 50, c = 'Y', marker = 'o')
    elif(GN['NODE CLASS'][i] == 'Other'):
        Y_V = plt.scatter(GN['X'][i], GN['Y'][i], alpha = 0.5, s = 50, c = 'Y', marker = 'v')


plt.xlabel("X Location")
plt.ylabel("Y Location")
plt.legend([Y_S, Y_O, Y_V, B_S, B_O, B_V, R_S, R_O, R_V],\
           ["Gas Pump", "Regulator Station", "Substation", "Pump Station", "Storage Tanks", "Delivery Nodes", "Power Plant", "Intersection Point", "Substation"], bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    
  
AdWE = np.zeros([NumberWN,NumberWN])
AdPE = np.zeros([NumberPN,NumberPN])
AdGE = np.zeros([NumberGN,NumberGN])

NumberWE = len(WE['LENGTH'])
NumberPE = len(PE['LENGTH'])
NumberGE = len(GE['LENGTH'])



for i in range(NumberWE):
    AdWE[WE['START WATER NODE ID'][i]-1][WE['END WATER NODE ID'][i]-1] = 1
    AdWE[WE['END WATER NODE ID'][i]-1][WE['START WATER NODE ID'][i]-1] = 1

InfrasDict['Objective'][0].ImitateAdj = AdWE
    
for i in range(NumberPE):
    AdPE[PE['START POWER NODE ID'][i]-1][PE['END POWER NODE ID'][i]-1] = 1
    AdPE[PE['END POWER NODE ID'][i]-1][PE['START POWER NODE ID'][i]-1] = 1

InfrasDict['Objective'][1].ImitateAdj = AdPE
  
for i in range(NumberGE):
    AdGE[GE['START GAS NODE ID'][i]-1][GE['END GAS NODE ID'][i]-1] = 1
    AdGE[GE['END GAS NODE ID'][i]-1][GE['START GAS NODE ID'][i]-1] = 1
    
InfrasDict['Objective'][2].ImitateAdj = AdGE