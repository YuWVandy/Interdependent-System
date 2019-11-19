# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:24:28 2019

@author: wany105
"""

import numpy as np
from scipy.stats import norm
import copy
import matplotlib.pyplot as plt 

def RR(PGV, K):
    RRRate = K*0.00187*PGV
    return RRRate

def PGAV(M_w, Distance):
    PGA = np.power(10, 3.79 + 0.298*(M_w - 6) - 0.0536*(M_w - 6)**2 - np.log10(Distance) - 0.00135*Distance)
    PGV = np.power(10, 2.04 + 0.422*(M_w - 6) - 0.0373*(M_w - 6)**2 - np.log10(Distance))
    return PGA, PGV


class DisruptionSystem:
    def __init__(self, Target, DisrupLocationLatLon, DisrupIntensity, Type):##Type 1 - System, Type 2 - Single Network
        self.Target = Target
        self.DisrupLocation = [int((DisrupLocationLatLon[0] - Disrupllat)/0.25), int((DisrupLocationLatLon[1] - Disrupllon)/0.25)]
        self.DisrupGeoLocation = [Disruplat[self.DisrupLocation[0]], Disruplon[self.DisrupLocation[1]]]
        self.M_w = DisrupIntensity
        self.Type = Type
        self.t = 0
        
        ##Keep Track the whole process
        self.NodeFail = []
        self.LinkFail = []
        self.NodeFailIndex = []
        self.LinkFailIndex = []
        
        if(Type == 'Network'):
            self.Target.Adj = copy.copy(self.Target.AdjMatrix)
            self.Target.Flow = copy.copy(self.Target.FlowValue)
        
        self.Target.AdjEvolution = []
        self.Target.AdjEvolution.append([self.t, self.Target.Adj])
        
        self.Target.FlowEvolution = []
        self.Target.FlowEvolution.append([self.t, self.Target.Flow])

        
        if(Type == 'System'):
            for Network in self.Target.NetworkObject:
                Network.SatisfyDemand = []
                Network.SatisfyDemand.append(Network.DemandValue)
                Network.Ratio = []
                Network.Ratio.append(np.divide(Network.SatisfyDemand[0], Network.SatisfyDemand[0], out=np.zeros_like(Network.SatisfyDemand[0]), where=Network.SatisfyDemand[0]!=0))
                Network.Performance = [1]
        else:
            self.Target.SatisfyDemand = []
            self.Target.SatisfyDemand.append(self.Target.DemandValue)
            self.Target.Ratio = []
            self.Target.Ratio.append(np.divide(self.Target.SatisfyDemand[0], self.Target.SatisfyDemand[0], out=np.zeros_like(self.Target.SatisfyDemand[0]), where = self.Target.SatisfyDemand[0]!=0))
            
        self.Target.Performance = [1]
        
        
    def DistanceCalculation(self):
        self.DisrupNodeDistance = np.sqrt((self.Target.GeoLocation[:, 0] - self.DisrupGeoLocation[0])**2 + (self.Target.GeoLocation[:, 1] - self.DisrupGeoLocation[1])**2)/1000 #Change the unit to km
        self.PGA, self.PGV = PGAV(self.M_w, self.DisrupNodeDistance)
    
    def NodeFailProbCalculation(self):
        self.NodeFailProb = np.zeros(self.Target.NodeNum)
        
        if(self.Type == 'System'):
            for Network in self.Target.NetworkObject:
                self.NodeFailProb[Network.WholeNodeSeries[Network.SupplyNodeSeries]] = norm.cdf((np.log(self.PGA[Network.WholeNodeSeries[Network.SupplyNodeSeries]]) - Network.DisruptSupplyLambda)/Network.DisruptSupplyZeta)
                
                self.NodeFailProb[Network.WholeNodeSeries[Network.TranNodeSeries]] = norm.cdf((np.log(self.PGA[Network.WholeNodeSeries[Network.TranNodeSeries]]) - Network.DisruptTranLambda)/Network.DisruptTranZeta)
                
                self.NodeFailProb[Network.WholeNodeSeries[Network.DemandNodeSeries]] = norm.cdf((np.log(self.PGA[Network.WholeNodeSeries[Network.DemandNodeSeries]]) - Network.DisruptDemandLambda)/Network.DisruptDemandZeta)
        else:
            self.NodeFailProb[self.Target.SupplyNodeSeries] = norm.cdf((np.log(self.PGA[self.Target.SupplyNodeSeries]) - self.Target.DisruptSupplyLambda)/self.Target.DisruptSupplyZeta)
            
            self.NodeFailProb[self.Target.TranNodeSeries] = norm.cdf((np.log(self.PGA[self.Target.TranNodeSeries]) - self.Target.DisruptTranLambda)/self.Target.DisruptTranZeta)
            
            self.NodeFailProb[self.Target.DemandNodeSeries] = norm.cdf((np.log(self.PGA[self.Target.DemandNodeSeries]) - self.Target.DisruptDemandLambda)/self.Target.DisruptDemandZeta)
            

    def LinkFailProbCalculation(self, m):
        self.DisruptLinkDistance = np.sqrt((self.Target.LinkNodeCoordinates[:, :, 0] - self.DisrupGeoLocation[0])**2 + (self.Target.LinkNodeCoordinates[:, :, 1] - self.DisrupGeoLocation[1])**2)/1000 #Change the unit to km
        self.PGA, self.PGV = PGAV(self.M_w, self.DisruptLinkDistance)

        self.RR = np.zeros(self.PGV.shape)
        for i in range(len(self.Target.LinkInNetwork)):
            self.RR[i] = self.Target.LinkInNetwork[i][0].LinkRRK*self.Target.LinkInNetwork[i][0].LinkRRa*self.PGV[i]
            
        self.LinkFailProb = 1 - np.exp(-self.Target.LinkLength/m*np.sum(self.RR, 1))
    
    def MCFailureSimulation(self):
        self.NodeFailRand = np.random.rand(len(self.NodeFailProb))
        self.LinkFailRand = np.random.rand(len(self.DisruptLinkDistance))
        
        #Failure due to Disruption
        self.NodeFail.append(self.NodeFailProb < self.NodeFailRand)
        self.NodeFailIndex.append(list(np.where(self.NodeFail[-1] == False)[0]))
        self.LinkFail.append(self.LinkFailProb < self.LinkFailRand)
        self.LinkFailIndex.append(list(np.where(self.LinkFail[-1] == False)[0]))        
    
    def Node2LinkFail(self):
        #Link Failure due to Node Failure
        Temp = self.Target.LinkInNetwork
        
        for i in self.NodeFailIndex[-1]:
            for j in range(len(Temp)):
                if(self.Type == 'System'):
                    if((i == Temp[j][0].WholeNodeSeries[Temp[j][2]] or i == Temp[j][1].WholeNodeSeries[Temp[j][3]]) and j not in self.LinkFailIndex[-1]):
                        self.LinkFailIndex[-1].append(j)
                else:
                    if((i == Temp[j][2] or i == Temp[j][3]) and j not in self.LinkFailIndex[-1]):
                        self.LinkFailIndex[-1].append(j)
        
    def AdjFlowUpdate(self):
        Temp = self.Target.LinkInNetwork
        Adj = copy.copy(self.Target.AdjEvolution[-1][1])            
        Adj[self.NodeFailIndex[-1], :] *= 0
        Adj[:, self.NodeFailIndex[-1]] *= 0
            
        for temp in Temp[self.LinkFailIndex[-1]]:
            if(self.Type == "System"):
                Adj[temp[0].WholeNodeSeries[temp[2]]][temp[1].WholeNodeSeries[temp[3]]] = 0
            else:
                Adj[temp[2]][temp[3]] = 0

        self.t += 1

        self.Target.AdjEvolution.append([self.t, Adj])
        
        self.Target.FlowEvolution.append([self.t, copy.copy(self.Target.FlowEvolution[-1][1])*self.Target.AdjEvolution[-1][1]])
 
    
    def SystemFlowRedistribution(self, DefunctionThreshold):
        self.TempFlow = self.Target.FlowEvolution[-1][1]

        Network1 = self.Target.Network1
        Network2 = self.Target.Network2
        Network3 = self.Target.Network3
        
        InterNetwork1 = self.Target.InterNetwork1
        InterNetwork2 = self.Target.InterNetwork2
        InterNetwork3 = self.Target.InterNetwork3
        InterNetwork4 = self.Target.InterNetwork4
        InterNetwork5 = self.Target.InterNetwork5
        
        #Gas Supply - Gas Tran
        for node in Network3.WholeNodeSeries[Network3.SupplyNodeSeries]:
            FlowoutNode = np.sum(self.TempFlow[node, Network3.WholeNodeSeries[Network3.TranNodeSeries]])
            Ratio = self.Target.FlowEvolution[0][1][node, :]/np.sum(self.Target.FlowEvolution[0][1][node, :])*self.Target.AdjEvolution[-1][1][node, :]
            Ratio = Ratio/np.sum(Ratio)
            Ratio[np.isnan(Ratio)] = 0
            if(np.sum(self.Target.AdjEvolution[-1][1][node, :]) == 0):
                self.TempFlow[node, :] = np.zeros(len(self.TempFlow[node, :]))
            else:
                self.TempFlow[node, :] = FlowoutNode*Ratio
        
        #Gas Tran - Gas Demand     
        for node in Network3.WholeNodeSeries[Network3.TranNodeSeries]:
            FlowintoNode = np.sum(self.TempFlow[Network3.WholeNodeSeries[Network3.SupplyNodeSeries], node])
            '''
            PowerintoNode =  np.sum(self.TempFlow[InterNetwork4.Network1.WholeNodeSeries[InterNetwork4.NodeSeries1], node])
            
            print(FlowintoNode, PowerintoNode/(0.1*InterNetwork4.PowerConversion))
            FlowintoNode = min(FlowintoNode, PowerintoNode/(0.1*InterNetwork4.PowerConversion))#0.2-coefficient, conversion robust coefficient
            '''
            Ratio = self.Target.FlowEvolution[0][1][node, :]/np.sum(self.Target.FlowEvolution[0][1][node, :])*self.Target.AdjEvolution[-1][1][node, :]
            Ratio = Ratio/np.sum(Ratio)
            Ratio[np.isnan(Ratio)] = 0
            if(np.sum(self.Target.AdjEvolution[-1][1][node, :]) == 0):
                self.TempFlow[node, :] = np.zeros(len(self.TempFlow[node, :]))
            else:
                self.TempFlow[node, :] = FlowintoNode*Ratio
        

        #Gas Tran - Gas Demand
        temp = 0
        Network3.SatisfyDemand.append([])
        for node in Network3.WholeNodeSeries[Network3.DemandNodeSeries]:
            FlowintoNode = np.sum(self.TempFlow[Network3.WholeNodeSeries[Network3.TranNodeSeries], node])
            FlowintoNode, Network3.DemandValue[temp] = math.floor(FlowintoNode), math.floor(Network3.DemandValue[temp])
            
            if(FlowintoNode >= Network3.DemandValue[temp]):
                FlowintoNode = FlowintoNode - Network3.DemandValue[temp]
                Network3.SatisfyDemand[-1].append(Network3.DemandValue[temp])
            else:
                Network3.SatisfyDemand[-1].append(0.25*FlowintoNode)
                FlowintoNode *= 0.75
            temp += 1
            Ratio = self.Target.FlowEvolution[0][1][node, :]/np.sum(self.Target.FlowEvolution[0][1][node, :])*self.Target.AdjEvolution[-1][1][node, :]
            Ratio = Ratio/np.sum(Ratio)
            Ratio[np.isnan(Ratio)] = 0
            if(np.sum(self.Target.AdjEvolution[-1][1][node, :]) == 0):
                self.TempFlow[node, :] = np.zeros(len(self.TempFlow[node, :]))
            else:
                self.TempFlow[node, :] = FlowintoNode*Ratio
        Network3.SatisfyDemand[-1] = np.array(Network3.SatisfyDemand[-1])

        

        #Gas Demand - Electricity Supply
        for node in Network2.WholeNodeSeries[Network2.SupplyNodeSeries]:
            FlowintoNode = np.sum(self.TempFlow[Network3.WholeNodeSeries[Network3.DemandNodeSeries], node])
            '''
            PowerintoNode =  np.sum(self.TempFlow[InterNetwork5.Network1.WholeNodeSeries[InterNetwork5.NodeSeries1], node])
            print('w', PowerintoNode, FlowintoNode)
            FlowintoNode = min(FlowintoNode, PowerintoNode/(0.01*InterNetwork5.PowerConversion))
            '''
            Ratio = self.Target.FlowEvolution[0][1][node, :]/np.sum(self.Target.FlowEvolution[0][1][node, :])*self.Target.AdjEvolution[-1][1][node, :]
            Ratio = Ratio/np.sum(Ratio)
            Ratio[np.isnan(Ratio)] = 0
            if(np.sum(self.Target.AdjEvolution[-1][1][node, :]) == 0):
                self.TempFlow[node, :] = np.zeros(len(self.TempFlow[node, :]))
            else:
                self.TempFlow[node, :] = InterNetwork2.ResourceConversion*FlowintoNode*Ratio


        #Electricity Supply - Electricity Tran
        for node in Network2.WholeNodeSeries[Network2.TranNodeSeries]:
            FlowintoNode = np.sum(self.TempFlow[Network2.WholeNodeSeries[Network2.SupplyNodeSeries], node])
            
            Ratio = self.Target.FlowEvolution[0][1][node, :]/np.sum(self.Target.FlowEvolution[0][1][node, :])*self.Target.AdjEvolution[-1][1][node, :]
            Ratio = Ratio/np.sum(Ratio)
            Ratio[np.isnan(Ratio)] = 0
            if(np.sum(self.Target.AdjEvolution[-1][1][node, :]) == 0):
                self.TempFlow[node, :] = np.zeros(len(self.TempFlow[node, :]))
            else:
                self.TempFlow[node, :] = FlowintoNode*Ratio

        #Electricity Tran - Electricity Demand
        temp = 0
        Network2.SatisfyDemand.append([])
        for node in Network2.WholeNodeSeries[Network2.DemandNodeSeries]:
            FlowintoNode = np.sum(self.TempFlow[Network2.WholeNodeSeries[Network2.TranNodeSeries], node])
            FlowintoNode, Network2.DemandValue[temp] = math.floor(FlowintoNode), math.floor(Network2.DemandValue[temp])
            if(FlowintoNode >= Network2.DemandValue[temp]):
                FlowintoNode -= Network2.DemandValue[temp]
                Network2.SatisfyDemand[-1].append(Network2.DemandValue[temp])
            else:
                Network2.SatisfyDemand[-1].append(0.8*FlowintoNode)
                FlowintoNode *= 0.2
            temp += 1
            Ratio = self.Target.FlowEvolution[0][1][node, :]/np.sum(self.Target.FlowEvolution[0][1][node, :])*self.Target.AdjEvolution[-1][1][node, :]
            Ratio = Ratio/np.sum(Ratio)
            Ratio[np.isnan(Ratio)] = 0
            if(np.sum(self.Target.AdjEvolution[-1][1][node, :]) == 0):
                self.TempFlow[node, :] = np.zeros(len(self.TempFlow[node, :]))
            else:
                for j in range(len(self.TempFlow[node, :])):
                    if(j in Network1.WholeNodeSeries[Network1.SupplyNodeSeries]):
                        self.TempFlow[node, j] = InterNetwork1.ResourceConversion*FlowintoNode*Ratio[j]
                    elif( j in Network1.WholeNodeSeries[Network1.TranNodeSeries]):
                        self.TempFlow[node, j] = InterNetwork3.PowerConversion*FlowintoNode*Ratio[j]
                    elif( j in Network3.WholeNodeSeries[Network3.TranNodeSeries]):
                        self.TempFlow[node, j] = InterNetwork4.PowerConversion*FlowintoNode*Ratio[j]
                    else:
                        self.TempFlow[node, j] = FlowintoNode*Ratio[j]
        Network2.SatisfyDemand[-1] = np.array(Network2.SatisfyDemand[-1])   

      
        ## Electricity Demand - Water Supply
        for node in Network1.WholeNodeSeries[Network1.SupplyNodeSeries]:
            FlowintoNode = np.sum(self.TempFlow[Network2.WholeNodeSeries[Network2.DemandNodeSeries], node])
            Ratio = self.Target.FlowEvolution[0][1][node, :]/np.sum(self.Target.FlowEvolution[0][1][node, :])*self.Target.AdjEvolution[-1][1][node, :]
            Ratio = Ratio/np.sum(Ratio)
            Ratio[np.isnan(Ratio)] = 0
            if(np.sum(self.Target.AdjEvolution[-1][1][node, :]) == 0):
                self.TempFlow[node, :] = np.zeros(len(self.TempFlow[node, :]))
            else:
                self.TempFlow[node, :] = FlowintoNode*Ratio

        ## Water Supply - Water Tran                
        for node in Network1.WholeNodeSeries[Network1.TranNodeSeries]:
            FlowintoNode = np.sum(self.TempFlow[:, node])
            '''
            PowerintoNode =  np.sum(self.TempFlow[InterNetwork3.Network1.WholeNodeSeries[InterNetwork3.NodeSeries1], node])
            FlowintoNode = min(FlowintoNode, PowerintoNode/(InterNetwork3.PowerConversion))
            '''
            Ratio = self.Target.FlowEvolution[0][1][node, :]/np.sum(self.Target.FlowEvolution[0][1][node, :])*self.Target.AdjEvolution[-1][1][node, :]
            Ratio = Ratio/np.sum(Ratio)
            Ratio[np.isnan(Ratio)] = 0
            if(np.sum(self.Target.AdjEvolution[-1][1][node, :]) == 0):
                self.TempFlow[node, :] = np.zeros(len(self.TempFlow[node, :]))
            else:
                self.TempFlow[node, :] = FlowintoNode*Ratio


        temp = 0
        ## Water Tran - Water Demand
        Network1.SatisfyDemand.append([])
        for node in Network1.WholeNodeSeries[Network1.DemandNodeSeries]:
            FlowintoNode = np.sum(self.TempFlow[Network1.WholeNodeSeries[Network1.TranNodeSeries], node])
            FlowintoNode, Network1.DemandValue[temp] = math.floor(FlowintoNode), math.floor(Network1.DemandValue[temp])
            temp += 1
            Network1.SatisfyDemand[-1].append(FlowintoNode)
            FlowintoNode = 0
            Ratio = self.Target.FlowEvolution[0][1][node, :]/np.sum(self.Target.FlowEvolution[0][1][node, :])*self.Target.AdjEvolution[-1][1][node, :]
            Ratio = Ratio/np.sum(Ratio)
            Ratio[np.isnan(Ratio)] = 0
            if(np.sum(self.Target.AdjEvolution[-1][1][node, :]) == 0):
                self.TempFlow[node, :] = np.zeros(len(self.TempFlow[node, :]))
            else:
                for j in range(len(self.TempFlow[node, :])):
                    if(j in Network2.WholeNodeSeries[Network2.SupplyNodeSeries]):
                        self.TempFlow[node, j] = InterNetwork5.PowerConversion*FlowintoNode*Ratio[j]
                    else:
                        self.TempFlow[node, j] = FlowintoNode*Ratio[j]

        Network1.SatisfyDemand[-1] = np.array(Network1.SatisfyDemand[-1]) 
        
        self.Target.FlowEvolution[-1][1] = self.TempFlow
        self.NodeFailIndex.append(copy.copy(self.NodeFailIndex[-1]))
        self.LinkFailIndex.append(copy.copy(self.LinkFailIndex[-1]))
        
        for i in range(len(self.Target.FlowEvolution[0][1])):
            ##Flow too small, cannot function normally
            if(np.sum(self.TempFlow[:, i]) < DefunctionThreshold*np.sum(self.Target.FlowEvolution[0][1][:, i]) and i not in self.NodeFailIndex[-1]):
                self.NodeFailIndex[-1].append(i)

            ##Node: Flow too large, exceeds the threshold
            if(np.sum(self.TempFlow[:, i]) > self.Target.NodeFlowCapacity[i] and i not in self.NodeFailIndex[-1]):
                self.NodeFailIndex[-1].append(i)

            
        
        Temp = self.Target.LinkInNetwork
        ##Link: Flow too large, exceeds the threshold
        for i in range(len(Temp)):
            source = Temp[i][0].WholeNodeSeries[Temp[i][2]]
            sink = Temp[i][1].WholeNodeSeries[Temp[i][3]]
            if(self.TempFlow[source][sink] > self.Target.LinkFlowCapacity[source][sink] and i not in self.LinkFailIndex[-1]):
                self.LinkFailIndex[-1].append(i)
            
    def NetworkFlowRedistribution(self, DefunctionThreshold):
        self.TempFlow = self.Target.FlowEvolution[-1][1]       
        
        #Gas Supply - Gas Tran
        for node in self.Target.SupplyNodeSeries:
            FlowoutNode = np.sum(self.TempFlow[node, self.Target.TranNodeSeries])
            Ratio = self.Target.FlowEvolution[0][1][node, :]/np.sum(self.Target.FlowEvolution[0][1][node, :])*self.Target.AdjEvolution[-1][1][node, :]
            Ratio = Ratio/np.sum(Ratio)
            Ratio[np.isnan(Ratio)] = 0
            if(np.sum(self.Target.AdjEvolution[-1][1][node, :]) == 0):
                self.TempFlow[node, :] = np.zeros(len(self.TempFlow[node, :]))
            else:
                self.TempFlow[node, :] = FlowoutNode*Ratio

        #Supply-Tran-Demand
        for node in self.Target.TranNodeSeries:
            FlowintoNode = np.sum(self.TempFlow[self.Target.SupplyNodeSeries, node])
            Ratio = self.Target.FlowEvolution[0][1][node, :]/np.sum(self.Target.FlowEvolution[0][1][node, :])*self.Target.AdjEvolution[-1][1][node, :]
            Ratio = Ratio/np.sum(Ratio)
            Ratio[np.isnan(Ratio)] = 0
            if(np.sum(self.Target.AdjEvolution[-1][1][node, :]) == 0):
                self.TempFlow[node, :] = np.zeros(len(self.TempFlow[node, :]))
            else:
                self.TempFlow[node, :] = FlowintoNode*Ratio

        temp = 0
        self.Target.SatisfyDemand.append([])
        for node in self.Target.DemandNodeSeries:
            FlowintoNode = np.sum(self.TempFlow[self.Target.TranNodeSeries, node])
            FlowintoNode, self.Target.DemandValue[temp] = math.floor(FlowintoNode), math.floor(self.Target.DemandValue[temp])
            temp += 1
            self.Target.SatisfyDemand[-1].append(FlowintoNode)
            FlowintoNode = 0
            if(np.sum(self.Target.AdjEvolution[-1][1][node, :]) == 0):
                self.TempFlow[node, :] = np.zeros(len(self.TempFlow[node, :]))
            else:
                self.TempFlow[node, :] = FlowintoNode*Ratio

        self.Target.SatisfyDemand[-1] = np.array(self.Target.SatisfyDemand[-1])
        
        self.Target.FlowEvolution[-1][1] = self.TempFlow

        self.NodeFailIndex.append(copy.copy(self.NodeFailIndex[-1]))
        self.LinkFailIndex.append(copy.copy(self.LinkFailIndex[-1]))
        
        for i in range(len(self.Target.FlowEvolution[0][1])):
            ##Flow too small, cannot function normally
            if(np.sum(self.TempFlow[:, i]) < DefunctionThreshold*np.sum(self.Target.FlowEvolution[0][1][:, i]) and i not in self.NodeFailIndex[-1]):
                self.NodeFailIndex[-1].append(i)

            ##Node: Flow too large, exceeds the threshold
            
            if(np.sum(self.TempFlow[:, i]) > self.Target.NodeFlowCapacity[i] and i not in self.NodeFailIndex[-1]):
                self.NodeFailIndex[-1].append(i)
                
        Temp = self.Target.LinkInNetwork

        ##Link: Flow too large, exceeds the threshold
        for i in range(len(Temp)):
            source = Temp[i][2]
            sink = Temp[i][3]
            if(self.TempFlow[source][sink] > self.Target.LinkFlowCapacity[source][sink] and i not in self.LinkFailIndex[-1]):
                self.LinkFailIndex[-1].append(i)  
    
    def Performance(self):
        if(self.Type == 'System'):
            for Network in self.Target.NetworkObject:
                Temp, Tempp = 0, 0
                Network.Ratio.append([])
                for i in range(len(Network.SatisfyDemand[-1])):
                    if((Network.SatisfyDemand[-1][i]/Network.SatisfyDemand[0][i] >= 1) or (Network.SatisfyDemand[-1][i] == 0 and Network.SatisfyDemand[0][i] == 0)):
                        Network.Ratio[-1].append(1)
                    else:
                        Network.Ratio[-1].append(Network.SatisfyDemand[-1][i]/Network.SatisfyDemand[0][i])
                Network.Performance.append(sum(Network.Ratio[-1])/len(Network.Ratio[-1]))
                """
                Network.Performance.append(np.sum(Network.SatisfyDemand[-1])/np.sum(Network.SatisfyDemand[0]))
                """
            self.Target.Performance.append((self.Target.Network1.Performance[-1]+self.Target.Network2.Performance[-1]+self.Target.Network3.Performance[-1])/3)
        else:
            Temp, Tempp = 0, 0
            self.Target.Ratio.append([])
            for i in range(len(self.Target.SatisfyDemand[-1])):
                if((self.Target.SatisfyDemand[-1][i]/self.Target.SatisfyDemand[0][i] >= 1) or (self.Target.SatisfyDemand[-1][i] == 0 and self.Target.SatisfyDemand[0][i] == 0)):
                    self.Target.Ratio[-1].append(1)
                else:
                    self.Target.Ratio[-1].append(self.Target.SatisfyDemand[-1][i]/self.Target.SatisfyDemand[0][i])
            self.Target.Performance.append(sum(self.Target.Ratio[-1])/len(self.Target.Ratio[-1]))
            """
            self.Target.Performance.append(np.sum(self.Target.SatisfyDemand[-1])/np.sum(self.Target.SatisfyDemand[0]))
            """
            
            

#Input Setting
DisrupIntensity = np.arange(0, 10.5, 1)
DisrupLat = np.arange(25, 46, 1)
DisrupLon = np.arange(-100, -79, 1)
"""
Test1
DisrupIntensity = np.arange(0, 10.5, 0.5)
DisrupLat = np.arange(27.5, 43.5, 1)
DisrupLon = np.arange(-97.5, -81.5, 1)
"""
"""
DisrupIntensity = np.arange(0, 10.5, 0.5)
DisrupLat = np.arange(27.5, 42.75, 0.25)
DisrupLon = np.arange(-97.5, -82.25, 0.25)
"""
DefunctionThreshold = 0.5 #The low bound of flow value into the node, which can be the low threshold to decide whether the node is failure or not
LineSegNum = 300 #The number of Line Segments
SimuTime = 1

SystemPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
SystemNodeFailIndex = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
WaterPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
ElectricityPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
GasPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))

SingleSystemPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
SingleWaterPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
SingleElectricityPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
SingleGasPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))

DiffSystemPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
DiffWaterPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
DiffElectricityPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
DiffGasPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))

SystemPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
SingleSystemPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
DiffSystemPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
DiffSystemPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
DiffWaterPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
DiffElectricityPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
DiffGasPerformance = np.array([[[[None]*SimuTime]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
            
            
            
for i in range(len(DisrupIntensity)):
    for j in range(len(DisrupLat)):
        for m in range(len(DisrupLon)):
            Temp = 0
            while(Temp < SimuTime):
                DisrupLatLonLocation = np.array([DisrupLat[j], DisrupLon[m]])
                #Disruption2System Modeling
                Earthquake2System = DisruptionSystem(WaterElecGasNetwork, DisrupLatLonLocation, DisrupIntensity[i], 'System')
                Earthquake2System.DistanceCalculation()
                Earthquake2System.NodeFailProbCalculation()
                Earthquake2System.LinkFailProbCalculation(LineSegNum)
                Earthquake2System.MCFailureSimulation()
                #System Failure Simulation
                while(1):
                    temp = Earthquake2System.Target.Performance[-1]
                    Earthquake2System.Node2LinkFail()
                    Earthquake2System.AdjFlowUpdate()
                    Earthquake2System.SystemFlowRedistribution(DefunctionThreshold)
                    Earthquake2System.Performance()
                    if(temp == Earthquake2System.Target.Performance[-1]):
                        break
                
                SystemNodeFailIndex[i][j][m][Temp] = Earthquake2System.NodeFailIndex[-1]
                #Disruption2Network Modeling
                SingleWater = copy.copy(Water)
                Earthquake2Water = DisruptionSystem(SingleWater, DisrupLatLonLocation, DisrupIntensity[i], 'Network')
                Earthquake2Water.DistanceCalculation()
                Earthquake2Water.NodeFailProbCalculation()
                Earthquake2Water.LinkFailProbCalculation(LineSegNum)
                
                SingleElectricity = copy.copy(Electricity)
                Earthquake2Electricity = DisruptionSystem(SingleElectricity, DisrupLatLonLocation, DisrupIntensity[i], 'Network')
                Earthquake2Electricity.DistanceCalculation()
                Earthquake2Electricity.NodeFailProbCalculation()
                Earthquake2Electricity.LinkFailProbCalculation(LineSegNum)
        
                SingleGas = copy.copy(Gas)
                Earthquake2Gas = DisruptionSystem(SingleGas, DisrupLatLonLocation, DisrupIntensity[i], 'Network')
                Earthquake2Gas.DistanceCalculation()
                Earthquake2Gas.NodeFailProbCalculation()
                Earthquake2Gas.LinkFailProbCalculation(LineSegNum)
        
                #Network Failure Simulation
                Earthquake2Water.NodeFailIndex.append([])
                Earthquake2Electricity.NodeFailIndex.append([])
                Earthquake2Gas.NodeFailIndex.append([])
                Earthquake2Water.LinkFailIndex.append([])
                Earthquake2Electricity.LinkFailIndex.append([])
                Earthquake2Gas.LinkFailIndex.append([])
                
                #Assign Initial Node and Link Failure based on the failure sceneria in System Disruption
                for FailNode in Earthquake2System.NodeFailIndex[0]:
                    if(FailNode in Water.WholeNodeSeries):
                        Index = list(Water.WholeNodeSeries).index(FailNode)
                        Earthquake2Water.NodeFailIndex[-1].append(Water.NodeSeries[Index])
                    elif(FailNode in Electricity.WholeNodeSeries):
                        Index = list(Electricity.WholeNodeSeries).index(FailNode)
                        Earthquake2Electricity.NodeFailIndex[-1].append(Electricity.NodeSeries[Index])    
                    else:
                        Index = list(Gas.WholeNodeSeries).index(FailNode)
                        Earthquake2Gas.NodeFailIndex[-1].append(Gas.NodeSeries[Index])   
        
                for FailLink in Earthquake2System.LinkFailIndex[0]:
                    temp = Earthquake2System.Target.LinkInNetwork[FailLink]
                    for k in range(len(Earthquake2Water.Target.LinkInNetwork)):       
                        if((list(Earthquake2Water.Target.LinkInNetwork[i])) == list(temp)):
                            Earthquake2Water.LinkFailIndex[-1].append(i)
                    
                    for k in range(len(Earthquake2Electricity.Target.LinkInNetwork)):
                        if(list(Earthquake2Electricity.Target.LinkInNetwork[i]) == list(temp)):
                            Earthquake2Electricity.LinkFailIndex[-1].append(i)
                            
                    for k in range(len(Earthquake2Gas.Target.LinkInNetwork)):
                        if(list(Earthquake2Gas.Target.LinkInNetwork[i]) == list(temp)):
                            Earthquake2Gas.LinkFailIndex[-1].append(i)
                #Network Failure Simulation
                while(1):
                    temp = Earthquake2Water.Target.Performance[-1]
                    Earthquake2Water.Node2LinkFail()
                    Earthquake2Water.AdjFlowUpdate()
                    Earthquake2Water.NetworkFlowRedistribution(DefunctionThreshold)
                    Earthquake2Water.Performance()
                    if(temp == Earthquake2Water.Target.Performance[-1]):
                        break
                while(1):
                    temp = Earthquake2Electricity.Target.Performance[-1]
                    Earthquake2Electricity.Node2LinkFail()
                    Earthquake2Electricity.AdjFlowUpdate()
                    Earthquake2Electricity.NetworkFlowRedistribution(DefunctionThreshold)
                    Earthquake2Electricity.Performance()
                    if(temp == Earthquake2Electricity.Target.Performance[-1]):
                        break
                while(1):
                    temp = Earthquake2Gas.Target.Performance[-1]
                    Earthquake2Gas.Node2LinkFail()
                    Earthquake2Gas.AdjFlowUpdate()
                    Earthquake2Gas.NetworkFlowRedistribution(DefunctionThreshold)
                    Earthquake2Gas.Performance()
                    if(temp == Earthquake2Gas.Target.Performance[-1]):
                        break
                    
                WaterPerformance[i][j][m][Temp] = Water.Performance
                ElectricityPerformance[i][j][m][Temp] = Electricity.Performance
                GasPerformance[i][j][m][Temp] = Gas.Performance
                SingleWaterPerformance[i][j][m][Temp] = SingleWater.Performance
                SingleElectricityPerformance[i][j][m][Temp] = SingleElectricity.Performance
                SingleGasPerformance[i][j][m][Temp] = SingleGas.Performance           
                
 
                #OutCome Saving
                print(i, j, m, Temp)
                len1 = len(WaterPerformance[i][j][m][Temp])
                len2 = len(ElectricityPerformance[i][j][m][Temp])
                len3 = len(GasPerformance[i][j][m][Temp])
                len4 = len(SingleWaterPerformance[i][j][m][Temp])
                len5 = len(SingleElectricityPerformance[i][j][m][Temp])
                len6 = len(SingleGasPerformance[i][j][m][Temp])
                
                WaterPerformancelist = list(WaterPerformance[i][j][m][Temp])
                ElectricityPerformancelist = list(ElectricityPerformance[i][j][m][Temp])
                GasPerformancelist = list(GasPerformance[i][j][m][Temp])
                SingleWaterPerformancelist = list(SingleWaterPerformance[i][j][m][Temp])
                SingleElectricityPerformancelist = list(SingleElectricityPerformance[i][j][m][Temp])
                SingleGasPerformancelist = list(SingleGasPerformance[i][j][m][Temp])            
                
                for k in range(20-len1):
                    WaterPerformancelist.append(WaterPerformancelist[-1])
                for k in range(20-len2):
                    ElectricityPerformancelist.append(ElectricityPerformancelist[-1])
                for k in range(20-len3):
                    GasPerformancelist.append(GasPerformancelist[-1])
                for k in range(20-len4):
                    SingleWaterPerformancelist.append(SingleWaterPerformancelist[-1])
                for k in range(20-len5):
                    SingleElectricityPerformancelist.append(SingleElectricityPerformancelist[-1])
                for k in range(20-len6):
                    SingleGasPerformancelist.append(SingleGasPerformancelist[-1])
                
                WaterPerformance[i][j][m][Temp] = np.array(WaterPerformancelist)
                ElectricityPerformance[i][j][m][Temp] = np.array(ElectricityPerformancelist)
                GasPerformance[i][j][m][Temp] = np.array(GasPerformancelist)
                SingleWaterPerformance[i][j][m][Temp] = np.array(SingleWaterPerformancelist)
                SingleElectricityPerformance[i][j][m][Temp] = np.array(SingleElectricityPerformancelist)
                SingleGasPerformance[i][j][m][Temp] = np.array(SingleGasPerformancelist)
                
                DiffWaterPerformance[i][j][m][Temp] = SingleWaterPerformance[i][j][m][Temp] - WaterPerformance[i][j][m][Temp]
                DiffElectricityPerformance[i][j][m][Temp] = SingleElectricityPerformance[i][j][m][Temp] - ElectricityPerformance[i][j][m][Temp]
                DiffGasPerformance[i][j][m][Temp] = SingleGasPerformance[i][j][m][Temp] - GasPerformance[i][j][m][Temp]
                SystemPerformance[i][j][m][Temp] = (WaterPerformance[i][j][m][Temp]+ElectricityPerformance[i][j][m][Temp]+GasPerformance[i][j][m][Temp])/3
                SingleSystemPerformance[i][j][m][Temp] = (SingleWaterPerformance[i][j][m][Temp]+SingleElectricityPerformance[i][j][m][Temp]+SingleGasPerformance[i][j][m][Temp])/3
                DiffSystemPerformance[i][j][m][Temp] = SingleSystemPerformance[i][j][m][Temp] - SystemPerformance[i][j][m][Temp]
                Temp += 1           


Parameter = {'LineSegNum': LineSegNum, 'SimuTime': SimuTime, 'DeFunctionThreshold': DefunctionThreshold, 'DisrupIntensity': DisrupIntensity, 'DisrupLat': DisrupLat, 'DisrupLon': DisrupLon,\
             'Residuality': 0.4, 'ErfaFlow': 30}
np.save(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test2\InfrasDict.npy', InfrasDict)
np.save(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test2\SingleElectricityPerformance.npy', SingleElectricityPerformance)
np.save(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test2\SingleWaterPerformance.npy', SingleWaterPerformance)
np.save(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test2\SingleGasPerformance.npy', SingleGasPerformance)
np.save(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test2\ElectricityPerformance.npy', ElectricityPerformance)
np.save(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test2\WaterPerformance.npy', WaterPerformance)
np.save(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test2\GasPerformance.npy', GasPerformance)
np.save(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test2\SystemPerformance.npy', SystemPerformance)
np.save(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test2\SingleSystemPerformance.npy', SingleSystemPerformance)
np.save(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test2\DiffSystemPerformance.npy', DiffSystemPerformance)
np.save(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test2\DiffWaterPerformance.npy', DiffWaterPerformance)
np.save(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test2\DiffGasPerformance.npy', DiffGasPerformance)
np.save(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test2\DiffElectricityPerformance.npy', DiffElectricityPerformance)   
   


import pickle
pickle_out = open(r"C:\Users\wany105\Desktop\Vulnerability update!!\Test2\Network.pickle","wb")
pickle.dump(Water, pickle_out)
pickle.dump(Gas, pickle_out)
pickle.dump(Electricity, pickle_out)
pickle.dump(ResourceElecWater, pickle_out)
pickle.dump(ResourceGasElec, pickle_out)
pickle.dump(PowerElecWater, pickle_out)
pickle.dump(PowerElecGas, pickle_out)
pickle.dump(WaterElecGasNetwork, pickle_out)
pickle.dump(Parameter, pickle_out)
pickle.dump(Earthquake2System, pickle_out)