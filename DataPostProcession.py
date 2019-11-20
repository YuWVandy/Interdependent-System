# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:24:43 2019

@author: wany105
"""


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
import seaborn as sns
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


##Intensity Change Curve: Intensity:[0, 3, 5, 7, 9], Local area: 35, -90
#Intensity change curve

IntenNum = [0, 3, 5, 7, 10]
LocalLatNum = 11
LocalLonNum = 11

AveSystemPerformance = np.zeros([len(IntenNum), 20])
AveSingleSystemPerformance = np.zeros([len(IntenNum), 20])

AveSystemError = []
AveSingleSystemError = []
for i in range(len(IntenNum)):
    AveSystemPerformance[i] = np.average(SystemPerformance[IntenNum[i]][LocalLatNum][LocalLonNum])
    Error = []
    for j in range(20):
        Temp1 = math.inf
        Temp2 = 0
        for k in range(SimuTime):
            if(SystemPerformance[IntenNum[i]][LocalLatNum][LocalLonNum][k][j] <= Temp1):
                Temp1 = SystemPerformance[IntenNum[i]][LocalLatNum][LocalLonNum][k][j]
            if(SystemPerformance[IntenNum[i]][LocalLatNum][LocalLonNum][k][j] >= Temp2):
                Temp2 = SystemPerformance[IntenNum[i]][LocalLatNum][LocalLonNum][k][j]
        Error.append([AveSystemPerformance[i][j] - Temp1, Temp2 - AveSystemPerformance[i][j]])
    AveSystemError.append(Error)
    
for i in range(len(IntenNum)):
    AveSingleSystemPerformance[i] = np.average(SingleSystemPerformance[IntenNum[i]][LocalLatNum][LocalLonNum])
    Error = []
    for j in range(20):
        Temp1 = math.inf
        Temp2 = 0
        for k in range(SimuTime):
            if(SingleSystemPerformance[IntenNum[i]][LocalLatNum][LocalLonNum][k][j] <= Temp1):
                Temp1 = SingleSystemPerformance[IntenNum[i]][LocalLatNum][LocalLonNum][k][j]
            if(SingleSystemPerformance[IntenNum[i]][LocalLatNum][LocalLonNum][k][j] >= Temp2):
                Temp2 = SingleSystemPerformance[IntenNum[i]][LocalLatNum][LocalLonNum][k][j]
        Error.append([AveSingleSystemPerformance[i][j] - Temp1, Temp2 - AveSingleSystemPerformance[i][j]])
    AveSingleSystemError.append(Error)
       
time = np.arange(0,20,1)
AveSystemError = np.array(AveSystemError)
AveSingleSystemError = np.array(AveSingleSystemError)
color = ['red','green','blue','black','pink']
Label = [['I-0','I-3','I-5','I-7','I-10'],['I\'-0','I\'-3','I\'-5','I\'-7','I\'-10']]
fig1 = plt.figure()
for i in range(len(IntenNum)):
    plt.plot(time, AveSystemPerformance[i], color = color[i], linestyle = '-', marker = 'o', label = Label[0][i])
    plt.plot(time, AveSingleSystemPerformance[i], color = color[i], linestyle = '-.', marker = '*', label = Label[1][i])
plt.xlim(0, 10, 1)
plt.xlabel('Time Step(s)', fontweight='bold')
plt.ylabel('System Performance', fontweight='bold')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.grid(True)
plt.show()


fig2 = plt.figure()
for i in range(len(IntenNum)):
    plt.plot(time, (AveSingleSystemPerformance[i] - AveSystemPerformance[i])/AveSingleSystemPerformance[i], color = color[i], linestyle = '-', marker = 'o', label = Label[0][i])
plt.xlim(0, 10, 1)
plt.xlabel('Time Step(s)', fontweight='bold')
plt.ylabel('Interdependence Strength', fontweight='bold')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.grid(True)
plt.show()




#The heat map of the population distribution Global
plt.figure(figsize=(14, 8))     
BaseMapSet(Type)
Disrupeq_map.pcolormesh(Disruplon, Disruplat, DisrupPDsub, cmap = cmap, norm = norm)
Disrupeq_map.colorbar(mappable=None, location='right', size='5%', pad='2%', fig=None, ax=None)
plt.title('The Heat Map of the Population Distribution')
plt.show()
#Localize
plt.figure(figsize=(14, 8))     
BaseMapSet(Type)
Disrupeq_map.pcolormesh(lon, lat, PDsub, cmap = cmap, norm = norm)
Disrupeq_map.colorbar(mappable=None, location='right', size='5%', pad='2%', fig=None, ax=None)
plt.title('The Heat Map of the Population Distribution')
plt.show()


AveSystemVulnerability = np.array([[[None]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
AveSingleSystemVulnerability = np.array([[[None]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
AveInterDependenceStrengthtime = np.array([[[None]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
AveInterDependenceStrength = np.array([[[None]*len(DisrupLat)]*len(DisrupLon)]*len(DisrupIntensity))
for k in range(len(DisrupIntensity)):
    for i in range(len(DisrupLat)):
        for j in range(len(DisrupLon)):
            AveSystemVulnerability[k][i][j] = 1 - np.average(SystemPerformance[k][i][j])
            AveSingleSystemVulnerability[k][i][j] = 1 - np.average(SingleSystemPerformance[k][i][j])
            AveInterDependenceStrengthtime[k][i][j] = (np.average(SingleSystemPerformance[k][i][j]) - np.average(SystemPerformance[k][i][j]))/np.average(SingleSystemPerformance[k][i][j])
            AveInterDependenceStrength[k][i][j] = AveInterDependenceStrengthtime[k][i][j][-1]
            
AveInterDependenceStrengthInsert = np.zeros([len(DisrupIntensity), len(Disruplat), len(Disruplon)])
for k in range(len(DisrupIntensity)):
    for i in range(len(Disruplat)):
        for j in range(len(Disruplon)):
            if(j == 0):
                Temp1 = int(i/4)
                Temp2 = np.mod(i, 4)
                if(i != 80):
                    Diff = (AveInterDependenceStrength[k][Temp1+1][j] - AveInterDependenceStrength[k][Temp1][j])/4*Temp2
                    AveInterDependenceStrengthInsert[k][i][j] = AveInterDependenceStrength[k][Temp1][j] + Diff
                else:
                    AveInterDependenceStrengthInsert[k][i][j] = AveInterDependenceStrength[k][Temp1][j]
            else:
                Temp1 = int(j/4)
                Temp2 = np.mod(j, 4)
                Tempp1 = int(i/4)
                Tempp2 = np.mod(i, 4)
                if(j != 80):
                    Diff = (AveInterDependenceStrength[k][Tempp1][Temp1+1] - AveInterDependenceStrength[k][Tempp1][Temp1])/4*Temp2
                    AveInterDependenceStrengthInsert[k][i][j] = AveInterDependenceStrength[k][Tempp1][Temp1] + Diff
                else:
                    AveInterDependenceStrengthInsert[k][i][j] = AveInterDependenceStrength[k][Tempp1][Temp1]
    
    
plt.figure(figsize=(14, 8))
BaseMapSet(Type)
plt.title('InterDependence Strength with the Earthquake Intensity of {}'.format(0))

Disrupeq_map.pcolormesh(Disruplon, Disruplat, AveInterDependenceStrengthInsert[0])
Disrupeq_map.colorbar(mappable=None, location='right', size='5%', pad='2%', fig=None, ax=None)

plt.show()
#The relationship between conditional probability and distance between two nodes
CaseSystemNodeFail = SystemNodeFailIndex[0][0][0] #Choose a node failure results under an earthquake with certain intensity and location

for j in range(WaterElecGasNetwork.NodeNum):
    for k in range(WaterElecGasNetwork.NodeNum):
        NumFailj = 0
        NumFailjk = 0
        for i in range(len(CaseSystemNodeFail)):
            if(j in CaseSystemNodeFail[i]):
                NumFailj += 1
            if(j in CaseSystemNodeFail[i] and k in CaseSystemNodeFail[i]):
                NumFailjk += 1
        if(NumFailj == 0):
            WaterElecGasNetwork.NodeFailTime[j, k] = [NumFailj, NumFailjk]
            WaterElecGasNetwork.NodeFailConProb[j, k] = 0
        else:
            WaterElecGasNetwork.NodeFailTime[j, k] = [NumFailj, NumFailjk]
            WaterElecGasNetwork.NodeFailConProb[j, k] = NumFailjk/NumFailj

DataFrame = np.array(list(WaterElecGasNetwork.NodeFailConProb), dtype=np.float)
ax = sns.heatmap(DataFrame, xticklabels = 10, yticklabels = 10)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+9.5, top-0.5)

X = []
Y = []
for i in range(WaterElecGasNetwork.NodeNum):
    for j in range(WaterElecGasNetwork.NodeNum):
      X.append(WaterElecGasNetwork.NodeDistance[i][j])
      Y.append(WaterElecGasNetwork.NodeFailConProb[i][j])

plt.scatter(X, Y)        


