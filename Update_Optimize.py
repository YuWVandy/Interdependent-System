# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:46:38 2019

@author: wany105
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:04:52 2019

@author: wany105
"""

runfile("Dataload.py")
runfile("OptFunction.py")

def BaseMapSet(Type):
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
##Generate the real map and plot the Infrastructure Facilities and Dependence
def DrawMap(llon, llat, rlon, rlat, Network, Type):    
    plt.figure(figsize=(14, 8))
    BaseMapSet(Type)
    """
    eq_map.pcolormesh(lon, lat, PDsub, cmap = cmap, norm = norm)
    eq_map.colorbar(mappable=None, location='right', size='5%', pad='2%', fig=None, ax=None)
    """
    Visual(Network)
    plt.show()

def Visual(Network):
    plt.scatter(Network.DemandLon, Network.DemandLat, 30, Network.Color, marker = 'o', label = Network.DemandName)  
    
    plt.scatter(Network.TranLon, Network.TranLat, 30, Network.Color, marker = '*', label = Network.TranName) 
    
    plt.scatter(Network.SupplyLon, Network.SupplyLat, 30, Network.Color, marker = '+', label = Network.SupplyName) 
    
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    
def FeatureScaling(A):
    return ((A-np.min(A))/(np.max(A)-np.min(A)))


#Single Network Class: the Total Number of Facilities and the Number of Demand, Transport and Supply Facilities respectively;
#ResourceNeedPerson: Resource need per Person; TransferFee: Transport fee per Resource; CapacityCofficient: Capacity = the Cofficient*real Flow;
#ResourceTo: Network that the resource goes to; ResourceFrom: Network that the resource comes from
class Infrastructure:
    def __init__(self, Name, SupplyName, TranName, DemandName, NodeNum, DemandNum, SupplyNum, TranNum, ResourceNeedPerson, TransferFee, CapacityCofficient, \
                 Color, FlowCapacity, SubColor, i, DisrupPara, RRPara):
        self.Name = Name
        self.SupplyName = SupplyName
        self.TranName = TranName
        self.DemandName = DemandName
        self.Color = Color
        
        self.DemandLat = np.random.randint(len(LAT), size = DemandNum, dtype = int)
        self.DemandLon = np.random.randint(len(LON), size = DemandNum, dtype = int)
        self.DemandNodeGeo = np.zeros([DemandNum, 2], dtype = int) #Demandnode - DemandNode
        self.DemandNode = np.zeros([DemandNum, 2], dtype = int) #Demandnode - DemandNode
        
        
        self.SupplyLat = np.random.randint(len(LAT), size = SupplyNum, dtype = int)
        self.SupplyLon = np.random.randint(len(LON), size = SupplyNum, dtype = int)
        self.SupplyNodeGeo = np.zeros([SupplyNum, 2], dtype = int) #Supplynode - SupplyNode
        self.SupplyNode = np.zeros([SupplyNum, 2], dtype = int) #Supplynode - SupplyNode
        
        self.TranLat = np.random.randint(len(LAT), size = TranNum, dtype = int)#Trans - Tran
        self.TranLon = np.random.randint(len(LON), size = TranNum, dtype = int)
        self.TranNodeGeo = np.zeros([TranNum,2], dtype = int) #Transnode - TranNode
        self.TranNode = np.zeros([TranNum,2], dtype = int) #Transnode - TranNode
        
        self.NodeLoc = None
        self.NodeLocGeo = None
        
        
        self.NodeNum = NodeNum
        self.DemandNum = DemandNum
        self.SupplyNum = SupplyNum
        self.TranNum = TranNum 
        
        self.SupplyIniNum = 0
        self.SupplyEndNum = self.SupplyIniNum + self.SupplyNum - 1
        self.TranIniNum = self.SupplyEndNum + 1
        self.TranEndNum = self.TranIniNum + self.TranNum - 1
        self.DemandIniNum= self.TranEndNum + 1
        self.DemandEndNum = self.DemandIniNum + self.DemandNum - 1
        self.DemandNodeSeries = np.arange(self.DemandIniNum, self.DemandEndNum + 1, 1)
        self.TranNodeSeries = np.arange(self.TranIniNum, self.TranEndNum + 1, 1)
        self.SupplyNodeSeries = np.arange(self.SupplyIniNum, self.SupplyEndNum + 1, 1)
        self.NodeSeries = np.arange(0, self.NodeNum, 1)
        self.SupplyTranPoiSample = None
        self.TranDemandPoiSample = None
        
        self.AdjMatrix = np.zeros([self.NodeNum, self.NodeNum])
        self.TranDemandAdj = np.zeros([self.TranNum, self.DemandNum])
        self.SupplyTranAdj = np.zeros([self.SupplyNum, self.TranNum])
        
        self.DistanceMatrix = np.zeros([self.NodeNum, self.NodeNum])
        self.TranDemandDist = np.zeros([self.TranNum, self.DemandNum])
        self.SupplyTranDist = np.zeros([self.SupplyNum, self.TranNum])
        
        self.DemandValue = np.zeros([self.DemandNum])
        self.SupplyValue = np.zeros([self.SupplyNum])
        self.TranValue = np.zeros([self.TranNum])
        self.NeedValue = None
        
        self.FlowValue = np.zeros([self.NodeNum, self.NodeNum])
        self.FlowTranDemand = np.zeros([self.TranNum, self.DemandNum])
        self.FlowSupplyTran = np.zeros([self.SupplyNum, self.TranNum])
        
        self.ResourceNeedPerson = ResourceNeedPerson
        self.TransferFee = TransferFee
        self.CapacityCofficient = CapacityCofficient
        self.FlowCapacity = FlowCapacity
        
        self.ResourceToNetwork = None
        self.ResourceFromNetwork = None
        self.PowerToNetwork = None
        self.PowerFromNetwork = None
        
        self.ImitateAdj = None
        self.ParameterLambda = None
        self.ParameterCov_matrix = None
        
        self.graph = None
        self.SupplyColor= SubColor[0]
        self.TranColor = SubColor[1]
        self.DemandColor = SubColor[2]
        self.NodeSize = None
        self.Colormap = None
        self.EdgeSize = None
        self.EdgeColor = InfrasDict['EdgeColor'][i]
        
        self.DisruptDemandLambda = DisrupPara[0][0]
        self.DisruptDemandZeta = DisrupPara[0][1]
        self.DisruptTranLambda = DisrupPara[1][0]
        self.DisruptTranZeta = DisrupPara[1][1]
        self.DisruptSupplyLambda = DisrupPara[2][0]
        self.DisruptSupplyZeta = DisrupPara[2][1]
        
        self.LinkFlowCapacity = np.zeros([self.NodeNum, self.NodeNum])
        self.NodeFlowCapacity = np.zeros([self.NodeNum])
        
        self.LinkRRK = RRPara[0]
        self.LinkRRa = RRPara[1]
        
    
    def Optimize_Demand(self, LATT, LONN, PDSUB, LAT, LON):
        for i in range(len(self.DemandLat)):
            self.DemandNode[i][0], self.DemandNode[i][1] = self.DemandLat[i], self.DemandLon[i]
        self.DemandNode, self.DemandCost = anneal(self.DemandNode, 'Population', LATT, LONN, PDSUB, LAT, LON)
        self.DemandLat = lat[self.DemandNode[:,0]]
        self.DemandLon = lon[self.DemandNode[:,1]]
        
    def Optimize_Trans(self, LATT, LONN, PDSUB, LAT, LON):
        for i in range(len(self.TranLat)):
            self.TranNode[i][0], self.TranNode[i][1] = self.TranLat[i], self.TranLon[i]
        self.TranNode, self.TransCost = anneal(self.TranNode, 'Facility', LATT, LONN, PDSUB, LAT, LON)
        self.TranLat = lat[self.TranNode[:,0]]
        self.TranLon = lon[self.TranNode[:,1]]
        
        
    def Optimize_Supply(self, LATT, LONN, PDSUB, LAT, LON):
        for i in range(len(self.SupplyLat)):
            self.SupplyNode[i][0], self.SupplyNode[i][1] = self.SupplyLat[i], self.SupplyLon[i]
        self.SupplyNode, self.SupplyCost = anneal(self.SupplyNode, 'Facility', LATT, LONN, PDSUB, LAT, LON)
        self.SupplyLat = lat[self.SupplyNode[:,0]]
        self.SupplyLon = lon[self.SupplyNode[:,1]]
        
    def Demand_Value(self, lat, lon, PDsub):
        for i in range(len(lat) - 1):
            for j in range(len(lon) - 1):
                Min_Dist = math.inf
                for k in range(self.DemandNum):
                    temp_X = 0.5*(lat[i] + lat[i+1])
                    temp_Y = 0.5*(lon[j] + lon[j+1])
                    Dist = math.sqrt((temp_X - self.DemandNodeGeo[k][0])**2 + (temp_Y - self.DemandNodeGeo[k][1])**2)
                    if(Dist < Min_Dist):
                        Index = k
                        Min_Dist = Dist
                self.DemandValue[Index] += PDsub[i][j]*self.ResourceNeedPerson
        
        self.NeedValue = np.concatenate((self.SupplyValue, self.TranValue, self.DemandValue))
        
    def Drawing(self):
        VisualGraphlink(Type, self)
        self.Graph = nx.from_numpy_matrix(self.AdjMatrix)
        self.Colormap = []
        self.NodeSize = []
        self.EdgeSize = []
    
        for node in self.Graph.nodes:
            if node in self.SupplyNodeSeries:
                self.NodeSize.append(1000)
                self.Colormap.append(self.SupplyColor)
            elif node in self.TranNodeSeries:
                self.NodeSize.append(500)
                self.Colormap.append(self.TranColor)
            else:
                self.NodeSize.append(200)
                self.Colormap.append(self.DemandColor)
        
            for edge in self.Graph.edges:
                self.EdgeSize.append(self.FlowValue[edge[0]][edge[1]])
        
        self.EdgeSize = np.array(self.EdgeSize)
        hist, bin_edges = np.histogram(self.EdgeSize, bins = 6)
        for i in range(len(self.EdgeSize)):
            j = 0
            while(True):
                j += 1
                if(bin_edges[j] >= self.EdgeSize[i]):
                    break
            self.EdgeSize[i] = j
        
        nx.draw(self.Graph, arrows = True, with_labels = True,\
                node_color = self.Colormap, node_size = self.NodeSize, alpha = 1, width = self.EdgeSize, font_color = 'white', label = {'0':['Water'], '1':['Gas']})
        
        plt.show()
        
    def LinkSegment(self, m, Network):
        self.LinkNodeCoordinates = []
        self.LinkInNetwork = []
        self.LinkLength = []

        for i in range(self.NodeNum):
            for j in range(self.NodeNum):
                if(self.AdjMatrix[i][j] == 1):
                    TempNode = []
                    BeginCoordinates = self.NodeLocGeo[i, :]
                    EndCoordinates = self.NodeLocGeo[j, :]
                    TempDistance = np.sqrt(np.sum((BeginCoordinates - EndCoordinates)**2))/1000
                    for k in range(m+1):
                        TempNode.append(BeginCoordinates + (EndCoordinates - BeginCoordinates)/m*k)
                    self.LinkNodeCoordinates.append(TempNode)
                    self.LinkInNetwork.append([Network, Network, i, j])
                    self.LinkLength.append(TempDistance)
                                
                        
        self.LinkNodeCoordinates = np.array(self.LinkNodeCoordinates)
        self.LinkInNetwork = np.array(self.LinkInNetwork)
        self.LinkLength = np.array(self.LinkLength)
        
    def NetworkResilience(self):
        self.FlowDemandRatio = np.sum(self.FlowValue[:,self.DemandIniNum:(self.DemandEndNum+1)], axis = 0)/self.DemandValue
        

        self.Resilience = np.sum(self.FlowDemandRatio)/len(self.FlowDemandRatio)
        
    def Network_Information(self):
        print('Network Name' + self.Name)
        
        print('Network Supply Node:{}, Number: {}, Location: {}'.format(self.SupplyName, self.SupplyNum, self.SupplyNode))
        print('Network Transport Node:{}, Number: {}, Location: {}'.format(self.TranName, self.TranNum, self.TranNode))
        print('Network Demand Node:{}, Number: {}, Location: {}'.format(self.DemandName, self.DemandNum, self.DemandNode))
        
        print('Network Adjacent Matrix:{}'.format(self.AdjMatrix))
        print('Network TranDemandAdjacent Matrix:{}'.format(self.TranDemandAdj))
        print('Network SupplyTranAdjacent Matrix:{}'.format(self.SupplyTranAdj))
        
        print('Network Distance Matrix:{}'.format(self.DistanceMatrix))
        print('Network Transport Demand Matrix:{}'.format(self.TranDemandDist))
        print('Network Supply Transport Matrix:{}'.format(self.SupplyTranDist))
        
        print('Network Node Demand:{}, Supply:{}, Transport:{}'.format(self.DemandValue, self.SupplyValue, self.TranValue))
        
        print('Network Flow:{}, TranDemandFlow:{}, SupplyTranFlow:{}'.format(self.FlowValue, self.FlowTranDemand, self.FlowSupplyTran))
        
        print('Network ResourcePerPerson:{}, TransferFee:{}, Capacity Cofficient:{}'.format(self.ResourceNeedPerson, self.TransferFee, self.CapacityCofficient))
        
        print('Netowrk Resource goes to {} Network, comes from {} Network'.format(self.ResourceToNetwork, self.ResourceFromNetwork))

#Feature Scaling
LATT = FeatureScaling(lat)
LONN = FeatureScaling(lon)
LATT = FeatureScaling(lat)
LONN = FeatureScaling(lon)
LAT = LATT
LON = LONN
PDSUB = FeatureScaling(PDsub)


InfrasDict = {'Name': ['Water', 'Electricity', 'Gas'], 'SupplyName': ['Pump', 'Power Plant', 'Gas Supply'], 'TranName': ['Storage Tank', 'GateStation', 'Pipeline'],\
              'DemandName': ['Demand Station', 'Substation', 'Demand Station'], 'Color': ['skyblue', 'red', 'green'], 'FacilityTypeNum': [3, 3, 3],\
              'TotalNum':[50, 50, 50], 'DemandNum':[30, 30, 30], 'TranNum':[15, 15, 15], 'SupplyNum':[5, 5, 5],\
              'NodeStartNum':[[0, 5, 20],[0, 5, 20],[0, 5, 20]], 'NodeEndNum':[[4, 19, 49], [4, 19, 49], [4, 19, 49]],\
              'Objective':[], 'InterdependenceRelationship':[], 'ResourceConversion':[], 'ResourceNeedPerson':[0.00001, 0.00001, 0.00001], \
              'TransferFee': [1, 1, 1], 'CapacityCofficient': [2, 2, 2], 'ResourceTo':[], 'ResourceFrom':[], 'FlowCapacity':[1000, 1000, 1000], \
              'DemandNodeLoc':[None]*3, 'TranNodeLoc':[None]*3, 'SupplyNodeLoc':[None]*3, \
              'DisruptPara': [[[np.log(1.5), 0.8],[np.log(1.5), 0.8],[np.log(1.2), 0.6]],[[np.log(1.4),0.4],[np.log(1.3), 0.4],[np.log(1.2), 0.4]],[[np.log(1.5), 0.8],[np.log(1.5), 0.8],[np.log(1.2), 0.6]]], \
              'SubColor': [['navy','deepskyblue','skyblue'], ['maroon','red','sandybrown'], ['darkviolet','lightpink','thistle']],\
              'EdgeColor': [plt.cm.Blues, plt.cm.Reds, plt.cm.PuRd], 'RRPara': [[0.5, 0.002], [0.5, 0.001], [0.5, 0.002]]}         

NetNum = 3
##-----------------------------------------------------------------------------Node Location Optimization based on Population
##DemandNode
for i in range(NetNum):
    exec('{} = Infrastructure("{}", "{}", "{}", "{}", {}, {}, {}, {}, {}, {}, {}, "{}", {}, {}, {}, {}, {})'\
         .format(InfrasDict['Name'][i], InfrasDict['Name'][i], InfrasDict['SupplyName'][i],\
         InfrasDict['TranName'][i], InfrasDict['DemandName'][i], \
         InfrasDict['TotalNum'][i], InfrasDict['DemandNum'][i], InfrasDict['SupplyNum'][i],\
         InfrasDict['TranNum'][i], InfrasDict['ResourceNeedPerson'][i], InfrasDict['TransferFee'][i], InfrasDict['CapacityCofficient'][i], \
         InfrasDict['Color'][i], InfrasDict['FlowCapacity'][i], InfrasDict['SubColor'][i], i, InfrasDict['DisruptPara'][i], InfrasDict['RRPara'][i]))
    exec('InfrasDict["Objective"].append({})'.format(InfrasDict["Name"][i]))
    Network = InfrasDict["Objective"][i]
    
    ##Node Location by Optimization
    ##Demand Node
    Network.Optimize_Demand(LATT, LONN, PDSUB, LAT, LON)
    InfrasDict['DemandNodeLoc'][i] = Network.DemandNode

    
    #Transport Node
    DemandLAT = FeatureScaling(lat[Network.DemandNode[:, 0]])
    DemandLON = FeatureScaling(lon[Network.DemandNode[:, 1]])
    Network.Optimize_Trans(DemandLAT, DemandLON, PDSUB, LAT, LON)
    InfrasDict['TranNodeLoc'][i] = Network.TranNode
    
    #Supply Node
    TranLAT = FeatureScaling(lat[Network.TranNode[:, 0]])
    TranLON = FeatureScaling(lon[Network.TranNode[:, 1]])
    Network.Optimize_Supply(TranLAT, TranLON, PDSUB, LAT, LON)
    InfrasDict['SupplyNodeLoc'][i] = Network.SupplyNode

    Network.DemandNodeGeo = np.array(list(np.c_[Network.DemandLat, Network.DemandLon]))
    Network.TranNodeGeo = np.array(list(np.c_[Network.TranLat, Network.TranLon]))
    Network.SupplyNodeGeo = np.array(list(np.c_[Network.SupplyLat, Network.SupplyLon]))
    Network.NodeLoc = np.array(list(Network.SupplyNode) + list(Network.TranNode) + list(Network.DemandNode))
    Network.NodeLocGeo = np.array(list(np.c_[Network.SupplyLat, Network.SupplyLon]) + list(np.c_[Network.TranLat, Network.TranLon]) \
                           + list(np.c_[Network.DemandLat, Network.DemandLon]))
    Network.GeoLocation = Network.NodeLocGeo

    
    if(Network.Name == "Gas"):
        Network.SupplyValue += math.inf
    Network.Demand_Value(lat, lon, PDsub)
                           
    DrawMap(llon, llat, rlon, rlat, Network, Type)
    
##-----------------------------------------------------------------------------Plot three networks on the map
plt.figure(figsize=(14, 8)) 
BaseMapSet(Type)
for i in InfrasDict['Objective']:
    Visual(i)
plt.show()

runfile("Targeted Sampling Network_ShelbyCounty.py")
Water.WholeNodeSeries = Water.NodeSeries
Electricity.WholeNodeSeries = Electricity.NodeSeries + Water.NodeNum
Gas.WholeNodeSeries = Gas.NodeSeries + Water.NodeNum + Electricity.NodeNum