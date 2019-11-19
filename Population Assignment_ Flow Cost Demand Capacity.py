# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:26:16 2019

@author: wany105
"""
from pulp import *
from collections import defaultdict

def NetworkFlowAssign(prob, Obj, NetworkList, InterNetworkList, InitialFlow, temp, Conversion):
    ##Flow within a Network
    for Network in NetworkList:
        for j in range(Network.NodeNum):
            for k in range(Network.NodeNum):
                if(Network.AdjMatrix[j][k] == 1):
                    exec('f{}{}{} = LpVariable("f{}{}{}", 0 , None)'.format(Network.Name, j, k, Network.Name, j, k))
                    
                    exec('prob += f{}{}{} <= {}, "c{}"'.format(Network.Name, j, k, Network.FlowCapacity, temp))
                    temp += 1
                    
                    Network.InitialFlow['f{}{}{}'.format(Network.Name, j, k)] = [j, k, None]
                    if(k >= Network.TranIniNum and k <= Network.TranEndNum):
                        Network.TranFlow[k - Network.TranIniNum] += '-f{}{}{}'.format(Network.Name, j, k)
                    
                    if(Network.Name == 'Electricity' and k>= Network.SupplyIniNum and k <= Network.SupplyEndNum):
                        Network.TranFlow[k - Network.SupplyIniNum] += '-f{}{}{}'.format(Network.Name, j , k)
                    
                    
                    InitialFlow['f{}{}{}'.format(Network.Name, j, k)] = None
                    if(Network.NodeFlowConverge[k] == ""):
                        Network.NodeFlowConverge[k] += 'f{}{}{}'.format(Network.Name, j, k)
                    else:
                        Network.NodeFlowConverge[k] += '+f{}{}{}'.format(Network.Name, j, k)
                    Network.NodeFlowConverge[j] += '-f{}{}{}'.format(Network.Name, j, k)
                    
                    if(Obj == ""):
                        Obj += 'f{}{}{}*{}'.format(Network.Name, j, k, Network.TransferFee)
                    else:
                        Obj += '+ f{}{}{}*{}'.format(Network.Name, j, k, Network.TransferFee)
    
    ##Flow between Network
    for InterNetwork in InterNetworkList:
        for j in range(len(InterNetwork.NodeSeries1)):
            for k in range(len(InterNetwork.NodeSeries2)):
                if(InterNetwork.Adj[j][k] == 1):
                    exec('f{}{}{}{} = LpVariable("f{}{}{}{}", 0 , None)'\
                         .format(InterNetwork.Network1.Name, InterNetwork.Network2.Name, InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k], InterNetwork.Network1.Name, InterNetwork.Network2.Name, InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k]))
                    
                    exec('prob += f{}{}{}{} <= {}, "c{}"'.format(InterNetwork.Network1.Name, InterNetwork.Network2.Name, InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k], InterNetwork.FlowCapacity, temp))
                    temp += 1
                    if(InterNetwork.Type == 'Resource'):
                        InterNetwork.InitialFlow['f{}{}{}{}'.format(InterNetwork.Network1.Name, InterNetwork.Network2.Name, InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k])] = [InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k], None]
                        InitialFlow['f{}{}{}{}'.format(InterNetwork.Network1.Name, InterNetwork.Network2.Name, InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k])] = None
                        if(InterNetwork.Network2.NodeFlowConverge[InterNetwork.NodeSeries2[k]] == ""):
                            InterNetwork.Network2.NodeFlowConverge[InterNetwork.NodeSeries2[k]] += 'f{}{}{}{}*{}'.format(InterNetwork.Network1.Name, InterNetwork.Network2.Name, InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k], Conversion[InterNetworkList.index(InterNetwork)])
                        else:
                            InterNetwork.Network2.NodeFlowConverge[InterNetwork.NodeSeries2[k]] += '+f{}{}{}{}*{}'.format(InterNetwork.Network1.Name, InterNetwork.Network2.Name, InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k],Conversion[InterNetworkList.index(InterNetwork)])
                        InterNetwork.Network1.NodeFlowConverge[InterNetwork.NodeSeries1[j]] += '-f{}{}{}{}'.format(InterNetwork.Network1.Name, InterNetwork.Network2.Name, InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k])
                    else:
                        exec('prob += f{}{}{}{} + {}*{} == 0, "c{}"'.format(InterNetwork.Network1.Name, InterNetwork.Network2.Name, InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k], InterNetwork.Network2.TranFlow[k] , Conversion[InterNetworkList.index(InterNetwork)], temp))
                        temp += 1
                        InterNetwork.InitialFlow['f{}{}{}{}'.format(InterNetwork.Network1.Name, InterNetwork.Network2.Name, InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k])] = [InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k], None]
                        InitialFlow['f{}{}{}{}'.format(InterNetwork.Network1.Name, InterNetwork.Network2.Name, InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k])] = None
                        InterNetwork.Network1.NodeFlowConverge[InterNetwork.NodeSeries1[j]] += '-f{}{}{}{}'.format(InterNetwork.Network1.Name, InterNetwork.Network2.Name, InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k])
                        
                    if(Obj == ""):
                        Obj += 'f{}{}{}{}*{}'.format(InterNetwork.Network1.Name, InterNetwork.Network2.Name, InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k], InterNetwork.TransferFee)
                    else:
                        Obj += '+ f{}{}{}{}*{}'.format(InterNetwork.Network1.Name, InterNetwork.Network2.Name, InterNetwork.NodeSeries1[j], InterNetwork.NodeSeries2[k], InterNetwork.TransferFee)
                    
    for Network in NetworkList:
        for j in range(Network.NodeNum):
            if((Network.Name != 'Gas') or (j not in range(Network.SupplyNum))):
                exec('prob += ' + Network.NodeFlowConverge[j] + ' == {}, "c{}"'.format(Network.NeedValue[j], temp))
                temp += 1
    
    exec('prob += ' + Obj + ', "obj"')
    
    return prob, InitialFlow


def NetworkProbSet(prob, NetworkList, InterNetworkList, Conversion):
    Obj = ""
    temp = 1
    InitialFlow = dict()
    
    for Network in NetworkList:
        Network.InitialFlow = dict()
        Network.NodeFlowConverge = [""]*Network.NodeNum
        Network.TranFlow = [""]*Network.TranNum
    
    prob, InitialFlow = NetworkFlowAssign(prob, Obj, NetworkList, InterNetworkList, InitialFlow, temp, Conversion)
    return prob, InitialFlow

def PostProcess(prob, InitialFlow):
    for v in prob.variables():
        print(v.name, "=", v.varValue)
        InitialFlow[v.name] = v.varValue

        for Network in SystemNetwork :
            try:
                Network.InitialFlow[v.name][2] = v.varValue
            except:
                continue
    
    for Network in SystemNetwork:
        for FlowID in Network.InitialFlow:
                Network.FlowValue[Network.InitialFlow[FlowID][0]][Network.InitialFlow[FlowID][1]] = Network.InitialFlow[FlowID][2]        


prob = LpProblem('InitialFlow', LpMinimize)
prob, InitialFlow = NetworkProbSet(prob, InfrasDict["Objective"], InterInfrasDict['Objective'], InterInfrasDict['Conversion'])
prob.solve()
print("Status:", LpStatus[prob.status])
PostProcess(prob, InitialFlow)


        




"""
for v in prob.variables():
    print(v.name, "=", v.varValue)
    InitialFlow[v.name] = v.varValue
    for i in range(NetNum):
        Network = InfrasDict['Objective'][i]
        try:
            Network.InitialFlow[v.name][2] = v.varValue
            break
        except:
            continue
        
    for i in range(ResourceInterNetworkNum):
        Network = InterInfrasDict['ResourceObjective'][i]
        try:
            Network.InitialFlow[v.name][2] = v.varValue
            break
        except:
            continue
        
    for i in range(PowerInterNetworkNum):
        Network = InterInfrasDict['PowerObjective'][i]
        try:
            Network.InitialFlow[v.name][2] = v.varValue
            break
        except:
            continue

for i in range(NetNum):
    Network = InfrasDict['Objective'][i]
    for FlowID in Network.InitialFlow:
        Network.FlowValue[Network.InitialFlow[FlowID][0]][Network.InitialFlow[FlowID][1]] = Network.InitialFlow[FlowID][2]
        
for i in range(ResourceInterNetworkNum):
    Network = InterInfrasDict['ResourceObjective'][i]
    for FlowID in Network.InitialFlow:
        Network.FlowValue[Network.InitialFlow[FlowID][0]][Network.InitialFlow[FlowID][1]] = Network.InitialFlow[FlowID][2]

for i in range(PowerInterNetworkNum):
    Network = InterInfrasDict['PowerObjective'][i]
    for FlowID in Network.InitialFlow:
        Network.FlowValue[Network.InitialFlow[FlowID][0]][Network.InitialFlow[FlowID][1]] = Network.InitialFlow[FlowID][2]

for i in range(NetNum):
    Network = InfrasDict['Objective'][i]
    Network.NetworkResilience()

del prob
"""