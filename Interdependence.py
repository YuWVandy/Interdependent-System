# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:49:04 2019

@author: wany105
"""
import math
import copy

def NMinIndex(Array, N):
    temp = 0
    Index = []
    while(temp < N):
        Min = math.inf
        for i in range(len(Array)):
            if(Array[i] < Min and i not in Index):
                Min = Array[i]
                Temp = i
        Index.append(Temp)
        temp += 1
    return Index

class Interdependence:
    def __init__(self, Name, Network1, Network2, NodeSeries1, NodeSeries2, Type, Conversion, N):#Network2 is dependent on Network1
        self.Name = Name
        self.Network1 = Network1
        self.Network2 = Network2
        self.NodeSeries1 = NodeSeries1
        self.NodeSeries2 = NodeSeries2
        self.NodeNum = Network1.NodeNum + Network2.NodeNum
        self.Type = Type

    
        self.Adj = np.zeros([len(self.NodeSeries1), len(self.NodeSeries2)])
        self.Dist = np.zeros([len(self.NodeSeries1), len(self.NodeSeries2)])
        
        if(Type == 'Resource'):
            self.ResourceConversion = Conversion #the amount of Resource of network1 to converge single unit of Resource of network2
            self.Network1.ResourceToNetwork = self.Network2
            self.Network2.ResourceFromNetwork = self.Network1
        elif(Type == 'Power'):
            self.PowerConversion = Conversion #the amount of Resource of network1 to maintain the transport of single unit of resource of network2
            self.Network1.PowerToNetwork = self.Network2
            self.Network2.PowerFromNetwork = self.Network1
            
        self.FlowCapacity = self.Network1.FlowCapacity
        self.TransferFee = self.Network1.TransferFee
        self.InitialFlow = dict()
        self.FlowValue = np.zeros([self.Network1.NodeNum, self.Network2.NodeNum])

    
    def DistCalculation(self):
        for i in range(len(self.NodeSeries1)):
            for j in range(len(self.NodeSeries2)):
                self.Dist[i][j] = math.sqrt((self.Network1.NodeLocGeo[i][0] - self.Network2.NodeLocGeo[j][0])**2 + (self.Network1.NodeLocGeo[i][1] - self.Network2.NodeLocGeo[j][1])**2)
        
    def AdjCalculation(self, N):
        for i in range(len(self.NodeSeries2)):
            Index = NMinIndex(self.Dist[:, i], N)
            self.Adj[Index, i] = 1
    
    def Print(self):
        print(self.Name)
        print(self.ResourceConversion)
        
        

TotalNetNum = 5
InterInfrasDict = {"Name": ["ResourceGasElec", "ResourceElecWater", "PowerElecWater", "PowerElecGas", "CoolingWaterElec"], \
                   "NodeSeries": [['Gas.DemandNodeSeries', 'Electricity.SupplyNodeSeries'], ['Electricity.DemandNodeSeries', 'Water.SupplyNodeSeries'],\
                                  ['Electricity.DemandNodeSeries', 'Water.TranNodeSeries'], ['Electricity.DemandNodeSeries','Gas.TranNodeSeries'], ['Water.DemandNodeSeries', 'Electricity.SupplyNodeSeries']],\
                   "TransferFee":[1, 1, 1, 1, 1], "Conversion": [1, 1, 0.1, 0.1, 0.1], "Type": ['Resource', 'Resource', 'Power', 'Power', 'Power'],\
                   "Relations":[["Gas", "Electricity"], ["Electricity", "Water"], ["Electricity", "Water"], ["Electricity", "Gas"], ["Water", "Electricity"]]}



for i in range(TotalNetNum):
    exec('{} = Interdependence("{}", {}, {}, {}, {}, "{}", {}, 5)'\
         .format(InterInfrasDict["Name"][i], InterInfrasDict["Name"][i], InterInfrasDict["Relations"][i][0], InterInfrasDict["Relations"][i][1], \
                 InterInfrasDict["NodeSeries"][i][0], InterInfrasDict["NodeSeries"][i][1], InterInfrasDict["Type"][i], InterInfrasDict["Conversion"][i]))
    
    exec('{}.DistCalculation()'.format(InterInfrasDict["Name"][i]))
    exec('{}.AdjCalculation(5)'.format(InterInfrasDict["Name"][i]))
    if(InterInfrasDict["Type"][i] == 'Resource'):
        exec('{}.ResourceToNetwork = {}'.format(InterInfrasDict["Relations"][i][0], InterInfrasDict["Relations"][i][1]))
        exec('{}.ResourceFromNetwork = {}'.format(InterInfrasDict["Relations"][i][1], InterInfrasDict["Relations"][i][0]))
    else:
        exec('{}.PowerToNetwork = {}'.format(InterInfrasDict["Relations"][i][0], InterInfrasDict["Relations"][i][1]))
        exec('{}.PowerFromNetwork = {}'.format(InterInfrasDict["Relations"][i][1], InterInfrasDict["Relations"][i][0]))

InterInfrasDict['Objective'] = [ResourceGasElec, ResourceElecWater, PowerElecWater, PowerElecGas, CoolingWaterElec]
InterInfrasDict['ResourceObjective'] = [ResourceGasElec, ResourceElecWater]
InterInfrasDict['PowerObjective'] = [PowerElecWater, PowerElecGas, CoolingWaterElec]

SystemNetwork = [Water, Electricity, Gas, ResourceGasElec, ResourceElecWater, PowerElecWater, PowerElecGas, CoolingWaterElec]