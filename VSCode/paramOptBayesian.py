# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 23:00:03 2019

@author: nitis
"""

from hyperopt import fmin, tpe, Trials, hp, STATUS_OK
import numpy as np
from scheduler import Scheduler
import matplotlib.pyplot as plt
import statistics

Hyperparams = {'psi1':hp.uniform('psi1',0.1,0.9),
'psi2':hp.uniform('psi2',0.1,0.9),
'psi3':hp.uniform('psi3',0.1,0.9),
'lambdaP':hp.uniform('lambdaP',0.1,0.99),
'b':hp.quniform('b',1,10,1),
'q':hp.quniform('q',2,10,1),
'p':hp.uniform('p',0.1,0.9),
'phi':hp.uniform('phi',0.1,5),
'chi':hp.uniform('chi',0.1,5),
'psi':hp.uniform('psi',0.1,5),
'r':hp.quniform('r',1,10,1),
'k':hp.quniform('k',2,6,1)
}



def hyperopt(param_space,num_eval, numRuns):
    
    def objective_function(params):
        main_path=r'Scenarios\ParameterOpt'
        scenario =r'1800-0.2-60-1'
        bestScores = []
        solutions = []
        hyperParamSets =[]
        for i in range(numRuns):
            scheduler = Scheduler(layoutFile=rf'{main_path}\{scenario}\DistanceMatrix\dm.csv', agvFile=rf'{main_path}\{scenario}\AGV\agvs.xlsx', \
                                  requestFile=rf'{main_path}\{scenario}\TRs\trs.xlsx', \
                                stationFile=rf'{main_path}\{scenario}\Station\stations.xlsx',hyperparams=params)
    
            solution =scheduler.solve(15)
            solutions.append(solution)
            hyperParamSets.append(params)
            bestScores.append(solution.get('bestScores')[-1][1])
        bestScore = statistics.mean(bestScores)
        return {'loss':bestScore, 'status':STATUS_OK, 'solution':solutions, 'paramSet':hyperParamSets, 'bestScores':bestScores}
        
    
    trials =Trials()

    best_param = fmin(objective_function, Hyperparams,algo=tpe.suggest,max_evals=num_eval,trials=trials,\
                      rstate=np.random.RandomState(1))
    loss = [x['result']['loss'] for x in trials.trials]
    history=[(x['result']['paramSet'], x['result']['solution'], x['result']['loss'], x['result']['bestScores']) for x in trials.trials]
    
    
    print('optimization completed / Running experiment with best parameters')
    scheduler = Scheduler(layoutFile=rf'{main_path}\{scenario}\DistanceMatrix\dm.csv', agvFile=rf'{main_path}\{scenario}\AGV\agvs.xlsx', \
                                  requestFile=rf'{main_path}\{scenario}\TRs\trs.xlsx', \
                                stationFile=rf'{main_path}\{scenario}\Station\stations.xlsx',hyperparams=best_param)
    
    solution =scheduler.solve(15)
    
    
    return best_param,loss, solution, history

bestParam, loss, solution, history = hyperopt(Hyperparams,150,10)

print('--------------------------------------------------------')
print('numIter:',solution.get('numIter'))
print('destroyN:',solution.get('stationDestroyN'))
print('destroyB:',solution.get('stationDestroyB'))
print('repairN',solution.get('stationRepairN'))
print('repairB',solution.get('stationRepairB'))
print('customerRepairN', solution.get('customerRepairN'))
print('customerDestroyN', solution.get('customerDestroyN'))
print('customerRepairB', solution.get('customerRepairB'))
print('customerDestroyB', solution.get('customerDestroyB'))
bestScores = solution.get('bestScores')
# scores.sort(reverse=True)
# plt.plot([x[0] for x in scores],[x[1] for x in scores])
plt.plot([x[0] for x in solution.get('bestScores')],[x[1] for x in solution.get('bestScores')])
plt.xticks([x[0] for x in solution.get('bestScores')],[x[2] for x in solution.get('bestScores')], rotation ='vertical')
plt.plot([x[0] for x in solution.get('scores')],[x[1] for x in solution.get('scores')])
plt.ylabel('Score of solution')
plt.xlabel('Iteration Number')
print(f'Best solution found at time:{bestScores[-1][2]:.3f} seconds')


h = [h[0] for h in history]
df = pd.DataFrame(h)
df['score']=[h[2] for h in history]
with open('export_df.csv', 'a') as f:
             df.to_csv(f, header=False)








    