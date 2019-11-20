# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 21:22:43 2019

@author: wany105
"""
import numpy as np
import math
import pandas as pd
import networkx as nx
import graphviz as gpz


def GraphvizGraph(A, Class1, Class2, Class3):
    G = gpz.Digraph()
    for i in range(len(A)):
        if(i in Class1):
            G.node(str(i), color = 'red')
        elif(i in Class2):
            G.node(str(i), color = 'blue')
        else:
            G.node(str(i), color = 'orange')
        for j in range(i, len(A)):
            if(A[i,j] == 1):
                G.edge(str(i), str(j))
    G.render('test-output/round-table.gv', view=True)

def VisualGraphlink(Type, Network):
    plt.figure(figsize=(14, 8))
    if(Type == 'local'):
        Base = Basemap(projection = 'merc', resolution = 'l', area_thresh = 1000.0, lat_0=0, lon_0=0, llcrnrlon=Disrupllon, llcrnrlat=Disrupllat, urcrnrlon=Disruprlon, urcrnrlat=Disruprlat)
    elif(Type == 'whole'):
        Base = Basemap(resolution = 'l', area_thresh = 1000.0, lat_0=0, lon_0=0, llcrnrlon=Disrupllon, llcrnrlat=Disrupllat, urcrnrlon=Disruprlon, urcrnrlat=Disruprlat)
    Base.drawcoastlines()
    Base.drawcountries()
    Base.drawmapboundary()
    parallels = np.arange(-90, 90, 10)
    Base.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    merid_values = np.arange(-180, 180., 10)
    meridians = Base.drawmeridians(merid_values,labels=[0,0,0,1],fontsize=10)  
    
    plt.scatter(Network.DemandLon, Network.DemandLat, 30, Network.DemandColor, marker = 'o', label = Network.DemandName)  
    plt.scatter(Network.TranLon, Network.TranLat, 70, Network.TranColor, marker = '*', label = Network.TranName) 
    plt.scatter(Network.SupplyLon, Network.SupplyLat, 100, Network.SupplyColor, marker = '+', label = Network.SupplyName) 
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    
    for j in range(Network.NodeNum):
        for k in range(Network.NodeNum):
            if(Network.AdjMatrix[j][k] == 1):
                plt.plot([Network.NodeLocGeo[j][1], Network.NodeLocGeo[k][1]], [Network.NodeLocGeo[j][0], Network.NodeLocGeo[k][0]], 'black', lw = 0.1)
    plt.show()    
    

class System:
    def __init__(self, Name, Network1, Network2, Network3, InterNetwork1, InterNetwork2, InterNetwork3, InterNetwork4, InterNetwork5):#the Whole System
        self.Name = Name
        self.Network1 = Network1
        self.Network2 = Network2
        self.Network3 = Network3
        self.NetworkObject = [self.Network1, self.Network2, self.Network3]
        
        self.InterNetwork1 = InterNetwork1
        self.InterNetwork2 = InterNetwork2
        self.InterNetwork3 = InterNetwork3
        self.InterNetwork4 = InterNetwork4
        self.InterNetwork5 = InterNetwork5
        self.InterNetworkObject = [self.InterNetwork1, self.InterNetwork2, self.InterNetwork3, self.InterNetwork4, self.InterNetwork5]
        self.ResourceInterNetworkObject = [self.InterNetwork1, self.InterNetwork2]
        self.PowerInterNetworkObject = [self.InterNetwork3, self.InterNetwork4, self.InterNetwork5]
        
        self.NodeNum = self.Network1.NodeNum + self.Network2.NodeNum + self.Network3.NodeNum

        self.Adj = np.zeros([self.Network1.NodeNum + self.Network2.NodeNum + self.Network3.NodeNum, self.Network1.NodeNum + self.Network2.NodeNum + self.Network3.NodeNum])
        self.GeoLocation = np.zeros([self.Network1.NodeNum + self.Network2.NodeNum + self.Network3.NodeNum, 2])
        self.Location = np.zeros([self.Network1.NodeNum + self.Network2.NodeNum + self.Network3.NodeNum, 2])
        self.Flow = np.zeros([self.Network1.NodeNum + self.Network2.NodeNum + self.Network3.NodeNum, self.Network1.NodeNum + self.Network2.NodeNum + self.Network3.NodeNum])
        self.LinkFlowCapacity = np.zeros([self.Network1.NodeNum + self.Network2.NodeNum + self.Network3.NodeNum, self.Network1.NodeNum + self.Network2.NodeNum + self.Network3.NodeNum])
        self.NodeFlowCapacity = np.zeros([self.Network1.NodeNum + self.Network2.NodeNum + self.Network3.NodeNum])
        self.beta = 0.4
        self.erfa = 30
        
        self.Color = []
        self.EdgeSize = []
        self.NodeSize = []
        self.NodeShape = []
        
        
        self.NodeDistance = np.zeros([self.NodeNum, self.NodeNum])
        self.NodeFailTime = np.array([[None]*self.NodeNum]*self.NodeNum)
        self.NodeFailConProb = np.array([[None]*self.NodeNum]*self.NodeNum)
        self.Graph = None

        
    def AdjCalculation(self):
        for Network in self.NetworkObject:
            self.Adj[Network.WholeNodeSeries[0]:(Network.WholeNodeSeries[-1]+1), Network.WholeNodeSeries[0]:(Network.WholeNodeSeries[-1]+1)] = Network.AdjMatrix
        
        self.Adj[self.InterNetwork1.Network1.WholeNodeSeries[self.InterNetwork1.Network1.DemandIniNum]:(self.InterNetwork1.Network1.WholeNodeSeries[self.InterNetwork1.Network1.DemandEndNum]+1),\
                 self.InterNetwork1.Network2.WholeNodeSeries[self.InterNetwork1.Network2.SupplyIniNum]:(self.InterNetwork1.Network2.WholeNodeSeries[self.InterNetwork1.Network2.SupplyEndNum]+1)] = self.InterNetwork1.Adj
        
        self.Adj[self.InterNetwork2.Network1.WholeNodeSeries[self.InterNetwork2.Network1.DemandIniNum]:(self.InterNetwork2.Network1.WholeNodeSeries[self.InterNetwork2.Network1.DemandEndNum]+1), 
                 self.InterNetwork2.Network2.WholeNodeSeries[self.InterNetwork2.Network2.SupplyIniNum]:(self.InterNetwork2.Network2.WholeNodeSeries[self.InterNetwork2.Network2.SupplyEndNum]+1)] = self.InterNetwork2.Adj
          
        self.Adj[self.InterNetwork3.Network1.WholeNodeSeries[self.InterNetwork3.Network1.DemandIniNum]:(self.InterNetwork3.Network1.WholeNodeSeries[self.InterNetwork3.Network1.DemandEndNum]+1), \
                 self.InterNetwork3.Network2.WholeNodeSeries[self.InterNetwork3.Network2.TranIniNum]:(self.InterNetwork3.Network2.WholeNodeSeries[self.InterNetwork3.Network2.TranEndNum]+1)] = self.InterNetwork3.Adj
          
        self.Adj[self.InterNetwork4.Network1.WholeNodeSeries[self.InterNetwork4.Network1.DemandIniNum]:(self.InterNetwork4.Network1.WholeNodeSeries[self.InterNetwork4.Network1.DemandEndNum]+1), \
                 self.InterNetwork4.Network2.WholeNodeSeries[self.InterNetwork4.Network2.TranIniNum]:(self.InterNetwork4.Network2.WholeNodeSeries[self.InterNetwork4.Network2.TranEndNum]+1)] = self.InterNetwork4.Adj
                 
        self.Adj[self.InterNetwork5.Network1.WholeNodeSeries[self.InterNetwork5.Network1.DemandIniNum]:(self.InterNetwork5.Network1.WholeNodeSeries[self.InterNetwork5.Network1.DemandEndNum]+1), \
                 self.InterNetwork5.Network2.WholeNodeSeries[self.InterNetwork5.Network2.SupplyIniNum]:(self.InterNetwork5.Network2.WholeNodeSeries[self.InterNetwork5.Network2.SupplyEndNum]+1)] = self.InterNetwork5.Adj

                 
    def FlowCalculation(self):
        for Network in self.NetworkObject:
            self.Flow[Network.WholeNodeSeries[0]:(Network.WholeNodeSeries[-1] + 1), Network.WholeNodeSeries[0]:(Network.WholeNodeSeries[-1]+1)] = Network.FlowValue    

        
        self.Flow[self.InterNetwork1.Network1.WholeNodeSeries[0]:(self.InterNetwork1.Network1.WholeNodeSeries[-1]+1), \
                 self.InterNetwork1.Network2.WholeNodeSeries[0]:(self.InterNetwork1.Network2.WholeNodeSeries[-1]+1)] = self.InterNetwork1.FlowValue + self.InterNetwork3.FlowValue 
        
        self.Flow[self.InterNetwork2.Network1.WholeNodeSeries[0]:(self.InterNetwork2.Network1.WholeNodeSeries[-1]+1), \
                 self.InterNetwork2.Network2.WholeNodeSeries[0]:(self.InterNetwork2.Network2.WholeNodeSeries[-1]+1)] = self.InterNetwork2.FlowValue
          
        self.Flow[self.InterNetwork4.Network1.WholeNodeSeries[0]:(self.InterNetwork4.Network1.WholeNodeSeries[-1]+1), \
                 self.InterNetwork4.Network2.WholeNodeSeries[0]:(self.InterNetwork4.Network2.WholeNodeSeries[-1]+1)] = self.InterNetwork4.FlowValue

        self.Flow[self.InterNetwork5.Network1.WholeNodeSeries[0]:(self.InterNetwork5.Network1.WholeNodeSeries[-1]+1), \
                 self.InterNetwork5.Network2.WholeNodeSeries[0]:(self.InterNetwork5.Network2.WholeNodeSeries[-1]+1)] = self.InterNetwork5.FlowValue
    
    def LocationCalculation(self):
        self.GeoLocation[self.Network1.NodeSeries[0]:(self.Network1.NodeSeries[-1]+1), 0:2] = self.Network1.NodeLocGeo
        self.Location[self.Network1.NodeSeries[0]:(self.Network1.NodeSeries[-1]+1), 0:2] = self.Network1.NodeLoc
        
        self.GeoLocation[self.Network2.WholeNodeSeries[0]:(self.Network2.WholeNodeSeries[-1]+1), 0:2] = self.Network2.NodeLocGeo
        self.Location[self.Network2.WholeNodeSeries[0]:(self.Network2.WholeNodeSeries[-1]+1), 0:2] = self.Network2.NodeLoc
        
        self.GeoLocation[self.Network3.WholeNodeSeries[0]:(self.Network3.WholeNodeSeries[-1]+1), 0:2] = self.Network3.NodeLocGeo
        self.Location[self.Network3.WholeNodeSeries[0]:(self.Network3.WholeNodeSeries[-1]+1), 0:2] = self.Network3.NodeLoc
        
    def NodeDistanceCal(self):
        for i in range(len(self.GeoLocation)):
            for j in range(i, len(self.GeoLocation)):
                if(i == j):
                    self.NodeDistance[i][j] = self.NodeDistance[j][i] = 0
                else:
                    self.NodeDistance[i][j] = self.NodeDistance[j][i] = np.sqrt(np.dot((self.GeoLocation[i] - self.GeoLocation[j]), \
                                     np.transpose(self.GeoLocation[i] - self.GeoLocation[j])))/1000 #Unit Change
                
        
    def GenerateGraph(self):
        self.Graph = nx.from_numpy_matrix(self.Adj)
        
    def DrawingParameter(self):
        for node in self.Graph.nodes:
            if node in self.Network1.WholeNodeSeries:
                if node in self.Network1.SupplyNodeSeries:
                    self.Color.append(self.Network1.SupplyColor)
                    self.NodeSize.append(100)
                    self.NodeShape.append('+')
                elif node in self.Network1.TranNodeSeries:
                    self.Color.append(self.Network1.TranColor)
                    self.NodeSize.append(100)
                    self.NodeShape.append('*')
                else:
                    self.Color.append(self.Network1.DemandColor)
                    self.NodeSize.append(100)
                    self.NodeShape.append('o')
            elif node in self.Network2.WholeNodeSeries:
                if node in (self.Network2.SupplyNodeSeries + self.Network1.NodeNum):
                    self.Color.append(self.Network2.SupplyColor)
                    self.NodeSize.append(100)
                    self.NodeShape.append('+')
                elif node in (self.Network2.TranNodeSeries + self.Network1.NodeNum):
                    self.Color.append(self.Network2.TranColor)
                    self.NodeSize.append(100)
                    self.NodeShape.append('*')
                else:
                    self.Color.append(self.Network2.DemandColor)
                    self.NodeSize.append(100)
                    self.NodeShape.append('o')
            else:
                if node in (self.Network3.SupplyNodeSeries + self.Network1.NodeNum + self.Network2.NodeNum):
                    self.Color.append(self.Network3.SupplyColor)
                    self.NodeSize.append(100)
                    self.NodeShape.append('+')
                elif node in (self.Network3.TranNodeSeries + self.Network1.NodeNum + self.Network2.NodeNum):
                    self.Color.append(self.Network3.TranColor)
                    self.NodeSize.append(100)
                    self.NodeShape.append('*')
                else:
                    self.Color.append(self.Network3.DemandColor)
                    self.NodeSize.append(100)
                    self.NodeShape.append('o')
        
        for edge in self.Graph.edges:
            self.EdgeSize.append(self.Flow[edge[0]][edge[1]])
            
        self.EdgeSize = np.array(self.EdgeSize)
        hist, bin_edges = np.histogram(self.EdgeSize, bins = 6)
        for i in range(len(self.EdgeSize)):
            j = 0
            while(True):
                j += 1
                if(bin_edges[j] >= self.EdgeSize[i]):
                    break
            self.EdgeSize[i] = j
     
    def FlowCapaCal(self):
        for Network in self.NetworkObject:
            for flow in Network.InitialFlow:
                if(Network.InitialFlow[flow][2] == 0):
                    Network.LinkFlowCapacity[Network.InitialFlow[flow][0]][Network.InitialFlow[flow][1]] = self.erfa
                    self.LinkFlowCapacity[Network.WholeNodeSeries[Network.InitialFlow[flow][0]]][Network.WholeNodeSeries[Network.InitialFlow[flow][1]]] = self.erfa
                else:
                    Network.LinkFlowCapacity[Network.InitialFlow[flow][0]][Network.InitialFlow[flow][1]] = Network.InitialFlow[flow][2]*(1+self.beta)
                    self.LinkFlowCapacity[Network.WholeNodeSeries[Network.InitialFlow[flow][0]]][Network.WholeNodeSeries[Network.InitialFlow[flow][1]]] = Network.InitialFlow[flow][2]*(1+self.beta)
        for Network in self.InterNetworkObject:
            for flow in Network.InitialFlow:
                if(Network.InitialFlow[flow][2] == 0):
                    self.LinkFlowCapacity[Network.Network1.WholeNodeSeries[Network.InitialFlow[flow][0]]][Network.Network2.WholeNodeSeries[Network.InitialFlow[flow][1]]] = self.erfa
                else:
                    self.LinkFlowCapacity[Network.Network1.WholeNodeSeries[Network.InitialFlow[flow][0]]][Network.Network2.WholeNodeSeries[Network.InitialFlow[flow][1]]] = Network.InitialFlow[flow][2]*(1+self.beta)
        for Network in self.NetworkObject:
            for node in Network.NodeSeries:
                if(node in Network.SupplyNodeSeries):
                    Network.NodeFlowCapacity[node] = np.sum(Network.FlowValue[node, :])*(1+self.beta)
                    self.NodeFlowCapacity[Network.WholeNodeSeries[node]] = np.sum(Network.FlowValue[node, :])*(1+self.beta)
                elif(node in Network.TranNodeSeries):
                    Network.NodeFlowCapacity[node] = np.sum(Network.FlowValue[node, :])*(1+self.beta)
                    self.NodeFlowCapacity[Network.WholeNodeSeries[node]] = np.sum(Network.FlowValue[node, :])*(1+self.beta)
                else:
                    Network.NodeFlowCapacity[node] = np.sum(Network.FlowValue[:, node])*(1+self.beta)
                    self.NodeFlowCapacity[Network.WholeNodeSeries[node]] = np.sum(Network.FlowValue[:, node])*(1+self.beta)
    def Draw(self):
        plt.figure(figsize = (15, 15))
        nx.draw(self.Graph, with_labels = True, node_color = self.Color, node_size = self.NodeSize,  width = self.EdgeSize, font_color='white', font_size = 8)
        plt.show()
        
    def LinkSegment(self, m):
        self.LinkNodeCoordinates = []
        self.LinkInNetwork = []
        self.LinkLength = []
        for Network in self.NetworkObject:
            for i in range(Network.NodeNum):
                for j in range(Network.NodeNum):
                    if(Network.AdjMatrix[i][j] == 1):
                        TempNode = []
                        BeginCoordinates = Network.NodeLocGeo[i, :]
                        EndCoordinates = Network.NodeLocGeo[j, :]
                        TempDistance = np.sqrt(np.sum((BeginCoordinates - EndCoordinates)**2))/1000
                        for k in range(m+1):
                            TempNode.append(BeginCoordinates + (EndCoordinates - BeginCoordinates)/m*k)
                        self.LinkNodeCoordinates.append(TempNode)
                        self.LinkInNetwork.append([Network, Network, i, j])
                        self.LinkLength.append(TempDistance)
                                
        for InterNetwork in self.InterNetworkObject:
            for i in range(len(InterNetwork.NodeSeries1)):
                for j in range(len(InterNetwork.NodeSeries2)):
                    if(InterNetwork.Adj[i][j] == 1):
                        TempNode = []
                        BeginCoordinates = InterNetwork.Network1.NodeLocGeo[InterNetwork.NodeSeries1[i], :]
                        EndCoordinates = InterNetwork.Network2.NodeLocGeo[InterNetwork.NodeSeries2[j], :]
                        TempDistance = np.sqrt(np.sum((BeginCoordinates - EndCoordinates)**2))/1000
                        for k in range(m+1):
                            TempNode.append(BeginCoordinates + (EndCoordinates - BeginCoordinates)/m*k)
                        self.LinkNodeCoordinates.append(TempNode)
                        self.LinkInNetwork.append([InterNetwork.Network1, InterNetwork.Network2, i, j])
                        self.LinkLength.append(TempDistance)
                        
        self.LinkNodeCoordinates = np.array(self.LinkNodeCoordinates)
        self.LinkInNetwork = np.array(self.LinkInNetwork)
        self.LinkLength = np.array(self.LinkLength)

        
    def Print(self):
        print(self.Name)
        
        

##Visualize Single Network

    


##Visualize Interdependent Network
#Convert Node Series:


Water.Drawing()
Electricity.Drawing()
Gas.Drawing()
##Visualize Interdependent Network
#Convert Node Series:

WaterElecGasNetwork = System("WaterElecGasNetwork", Water, Electricity, Gas, ResourceElecWater, ResourceGasElec, PowerElecWater, PowerElecGas, CoolingWaterElec)
WaterElecGasNetwork.AdjCalculation()
WaterElecGasNetwork.FlowCalculation()
WaterElecGasNetwork.LocationCalculation()
WaterElecGasNetwork.GenerateGraph()
m = 300 #the number of segments on the link
WaterElecGasNetwork.LinkSegment(m)
WaterElecGasNetwork.DrawingParameter()
WaterElecGasNetwork.Draw()
WaterElecGasNetwork.FlowCapaCal()
WaterElecGasNetwork.NodeDistanceCal()
