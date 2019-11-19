# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 23:54:21 2019

@author: wany105
"""


pickle_in = open(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test1\Network.pickle','rb')
Water = pickle.load(pickle_in)
Gas = pickle.load(pickle_in)
Electricity = pickle.load(pickle_in)
ResourceElecWater = pickle.load(pickle_in)
ResourceGasElec = pickle.load(pickle_in)
PowerElecWater = pickle.load(pickle_in)
PowerElecGas = pickle.load(pickle_in)
WaterElecGasNetwork = pickle.load(pickle_in)
Parameter = pickle.load(pickle_in)
Earthquake2System = pickle.load(pickle_in)
InfrasDict = np.load(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test1\InfrasDict.npy', allow_pickle = True)  
SingleElectricityPerformance = np.load(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test1\SingleElectricityPerformance.npy', allow_pickle = True)
SingleWaterPerformance = np.load(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test1\SingleWaterPerformance.npy', allow_pickle = True)
SingleGasPerformance = np.load(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test1\SingleGasPerformance.npy', allow_pickle = True)
ElectricityPerformance = np.load(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test1\ElectricityPerformance.npy', allow_pickle = True)
WaterPerformance = np.load(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test1\WaterPerformance.npy', allow_pickle = True)
GasPerformance = np.load(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test1\GasPerformance.npy', allow_pickle = True)
SystemPerformance = np.load(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test1\SystemPerformance.npy', allow_pickle = True)
SingleSystemPerformance = np.load(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test1\SingleSystemPerformance.npy', allow_pickle = True)
DiffSystemPerformance = np.load(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test1\DiffSystemPerformance.npy', allow_pickle = True)
DiffWaterPerformance = np.load(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test1\DiffWaterPerformance.npy', allow_pickle = True)
DiffGasPerformance = np.load(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test1\DiffGasPerformance.npy', allow_pickle = True)
DiffElectricityPerformance = np.load(r'C:\Users\wany105\Desktop\Vulnerability update!!\Test1\DiffElectricityPerformance.npy', allow_pickle = True)




        

AveInterDependenceStrengthInsert = np.zeros([len(Disruplat), len(Disruplon)])
for i in range(len(Disruplat)):
    for j in range(len(Disruplon)):
        if(j == 0):
            Temp1 = int(i/4)
            Temp2 = np.mod(i, 4)
            if(i != 80):
                Diff = (AveInterDependenceStrength[Temp1+1][j] - AveInterDependenceStrength[Temp1][j])/4*Temp2
                AveInterDependenceStrengthInsert[i][j] = AveInterDependenceStrength[Temp1][j] + Diff
            else:
                AveInterDependenceStrengthInsert[i][j] = AveInterDependenceStrength[Temp1][j]
        else:
            Temp1 = int(j/4)
            Temp2 = np.mod(j, 4)
            Tempp1 = int(i/4)
            Tempp2 = np.mod(i, 4)
            if(j != 80):
                Diff = (AveInterDependenceStrength[Tempp1][Temp1+1] - AveInterDependenceStrength[Tempp1][Temp1])/4*Temp2
                AveInterDependenceStrengthInsert[i][j] = AveInterDependenceStrength[Tempp1][Temp1] + Diff
            else:
                AveInterDependenceStrengthInsert[i][j] = AveInterDependenceStrength[Tempp1][Temp1]


    plt.figure(figsize=(14, 8))     
    BaseMapSet(Type)

    
    Disrupeq_map.pcolormesh(Disruplon, Disruplat, AveInterDependenceStrengthInsert)
    Disrupeq_map.colorbar(mappable=None, location='right', size='5%', pad='2%', fig=None, ax=None)
    
    plt.show()




