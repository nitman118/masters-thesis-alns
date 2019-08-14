# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:20:19 2019

@author: nitis
"""
import random

#orig = [ d for i,d in enumerate(orig) if d not in orig[i+1:]]


defaultHyperparams = {'psi1':0.9,
'psi2':0.6,
'psi3':0.3,
'lambdaP':0.5,
'a':2,
'b':5,
'alpha':0.5,
'q':2,
'p':0.8,
'phi':1,
'chi':1,
'psi':1,
'r':2,
'k':4
}

iterationPerExperiment = 5
runtimePerIteration = 5 #seconds

def createRandomParameterSet(params, n_size):
    '''
    creates a list of parameter sets that is passed to scheduler object
    '''
    paramSets = []
    for i in range(n_size):
        paramSet={}
        pSet = []
        for p in params:
            pSet.append(random.choice(p))
        paramSet['psi1']=pSet[0]
        paramSet['psi2']=pSet[1]
        paramSet['psi3']=pSet[2]
        paramSet['lambdaP']=pSet[3]
        paramSet['a']=pSet[4]
        paramSet['b']=pSet[5]
        paramSet['q']=pSet[6]
        paramSet['p']=pSet[7]
        paramSet['phi']=pSet[8]
        paramSet['chi']=pSet[9]
        paramSet['psi']=pSet[10]
        paramSet['r']=pSet[11]
        paramSet['k']=pSet[12]
        paramSet['alnsMethod']=pSet[13]
        paramSets.append(paramSet)
    paramSets = [p for i,p in enumerate(paramSets) if p not in paramSets[i+1:]]
    return paramSets
        




params= [
        [0.2,0.5,0.7,0.9,1,3,5,8,10,100],#psi1-ALNS parameter for rewarding best solution
          [0.2,0.5,0.7,0.9,1,3,5,8,10,100], #psi2-ALNS parameter for rewarding better solution
          [0.2,0.5,0.7,0.9,1,3,5,8,10,100], #psi3-ALNS parameter for rewarding a solution
          [0.2,0.5,0.7,0.9,0.99], #lambdaP-Step size of ALNS
          [1,2,3,4,5,6,7,8,9,10],#a-number of times customer methods are run before using station methods
          [1,2,3,4,5,6,7,8,9,10],#b- number of times station methods are run on finding a new best solution through customer method
          [1,2,3,4,5,6,7,8,9,10],#q-number of tasks to remove via shaw destroy
          [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],#p-randomness in choosing related jobs, lower values corresponds to more randomness
          [0.2,0.3,0.5,0.7,0.9,1.5,2,5,10,100],#phi-parameter to find relatedness due to distance
          [0.2,0.3,0.5,0.7,0.9,1.5,2,5,10,100], #chi-parameter to find relatedness due to time window
          [0.2,0.3,0.5,0.7,0.9,1.5,2,5,10,100], #psi-parameter to find relatedness due to agv capability
          [1,2,3,4,5,6,7,8,9,10], #r-number of random tasks to remove
          [1,2,3,4,5,6,7,8,9,10], #k in k-regret customer destroy method
          [4] # alns methods, 0 - default, 1-normalized, 2 - adaptive family, 3-explore&exploit, 4- with LP
        ]

paramSets = createRandomParameterSet(params1,10)

for parameterSet in paramSets:    
    for iteration in range(iterationPerExperiment):
        pass
    pass

    
        

