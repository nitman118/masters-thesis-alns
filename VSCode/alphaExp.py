# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:31:46 2019

@author: nitis
"""

import statistics
import pandas as pd
import os
import random
from shutil import copyfile
from scheduler import Scheduler
import copy

defaultHyperparams = {'psi1':0.87,
'psi2':0.65,
'psi3':0.66,
'lambdaP':0.65,
'a':6,
'b':4,
'alpha':1,
'q':4,
'p':0.6,
'phi':2.2,
'chi':2.21,
'psi':2.75,
'r':4,
'k':6,
'alnsMethod':4
}

def runAlphaExp(main_path,alpha, numRuns=5,exp_duration=15):
    cols = ['scenario_name','ho','tw','n','agvcase', 'upper_threshold','alpha']
     
    for run in range(numRuns):
        cols.append(f'tardiness{run}')
        cols.append(f'tardinessCost{run}')
        cols.append(f'lt{run}')
        cols.append(f'ult{run}')
        cols.append(f'totalTravelTime{run}')
        cols.append(f'totalTravelTimeCost{run}')
        cols.append(f'score{run}')
        cols.append(f'sla{run}')
        
    resultDF = pd.DataFrame(columns=cols)
    
    scenarios = os.listdir(main_path)
    for scenario in scenarios:
        splitScenario = scenario.split('-')
        row={'scenario_name':scenario,'ho':splitScenario[0],'tw':splitScenario[1],\
             'n':splitScenario[2],'agvcase':splitScenario[3], 'upper_threshold':splitScenario[4],'alpha':alpha}
        print(f'Scenario:{scenario}')
        for run in range(numRuns):
            newParam = copy.deepcopy(defaultHyperparams)
            newParam['alpha']=alpha
            scheduler = Scheduler(layoutFile=rf'{main_path}\{scenario}\DistanceMatrix\dm.csv', agvFile=rf'{main_path}\{scenario}\AGV\agvs.xlsx', \
                              requestFile=rf'{main_path}\{scenario}\TRs\trs.xlsx', \
                            stationFile=rf'{main_path}\{scenario}\Station\stations.xlsx',hyperparams=newParam)
            
            scheduler.solve(exp_duration)
            tardiness,tardinessCost,totalTravel,lt,ult,totalTravelTime,totalTravelTimeCost,sla=scheduler.getScheduleKPI()
            row[f'tardiness{run}']=tardiness
            row[f'tardinessCost{run}']=tardinessCost
            row[f'totalTravel{run}']=totalTravel
            row[f'lt{run}']=lt
            row[f'ult{run}']=ult
            row[f'totalTravelTime{run}']=totalTravelTime
            row[f'totalTravelTimeCost{run}']=totalTravelTimeCost
            row[f'score{run}']=scheduler.getScoreALNS()
            row[f'sla{run}']=sla
            
        #sample mean
        row[f'Av_tardiness'] = sum([row.get(f'tardiness{x}') for x in range(numRuns)])/numRuns
        row[f'Av_tardinessCost'] = sum([row.get(f'tardinessCost{x}') for x in range(numRuns)])/numRuns
        row[f'Av_totalTravel'] = sum([row.get(f'totalTravel{x}') for x in range(numRuns)])/numRuns
        row[f'Av_lt'] = sum([row.get(f'lt{x}') for x in range(numRuns)])/numRuns
        row[f'Av_ult'] = sum([row.get(f'ult{x}') for x in range(numRuns)])/numRuns
        row[f'Av_totalTravelTime'] = sum([row.get(f'totalTravelTime{x}') for x in range(numRuns)])/numRuns
        row[f'Av_totalTravelTimeCost'] = sum([row.get(f'totalTravelTimeCost{x}') for x in range(numRuns)])/numRuns
        row[f'Av_score'] = sum([row.get(f'score{x}') for x in range(numRuns)])/numRuns
        row[f'Av_sla']= sum([row.get(f'sla{x}') for x in range(numRuns)])/numRuns
        #sample standard deviation
        row[f'Std_tardiness'] = statistics.stdev([row.get(f'tardiness{x}') for x in range(numRuns)])
        row[f'Std_tardinessCost'] = statistics.stdev([row.get(f'tardinessCost{x}') for x in range(numRuns)])
        row[f'Std_totalTravel'] = statistics.stdev([row.get(f'totalTravel{x}') for x in range(numRuns)])
        row[f'Std_lt'] = statistics.stdev([row.get(f'lt{x}') for x in range(numRuns)])
        row[f'Std_ult'] = statistics.stdev([row.get(f'ult{x}') for x in range(numRuns)])
        row[f'Std_totalTravelTime'] = statistics.stdev([row.get(f'totalTravelTime{x}') for x in range(numRuns)])
        row[f'Std_totalTravelTimeCost'] = statistics.stdev([row.get(f'totalTravelTimeCost{x}') for x in range(numRuns)])
        row[f'Std_score'] = statistics.stdev([row.get(f'score{x}') for x in range(numRuns)])
        row[f'Std_sla']=statistics.stdev([row.get(f'sla{x}') for x in range(numRuns)])
        row[f'Min_tardiness'] = min([row.get(f'tardiness{x}') for x in range(numRuns)])
        row[f'Min_tardinessCost'] = min([row.get(f'tardinessCost{x}') for x in range(numRuns)])
        row[f'Min_totalTravel'] = min([row.get(f'totalTravel{x}') for x in range(numRuns)])
        row[f'Min_lt'] = min([row.get(f'lt{x}') for x in range(numRuns)])
        row[f'Min_ult'] = min([row.get(f'ult{x}') for x in range(numRuns)])
        row[f'Min_totalTravelTime'] = min([row.get(f'totalTravelTime{x}') for x in range(numRuns)])
        row[f'Min_totalTravelTimeCost'] = min([row.get(f'totalTravelTimeCost{x}') for x in range(numRuns)])
        row[f'Min_score'] = min([row.get(f'score{x}') for x in range(numRuns)])
        row[f'Max_sla']=max([row.get(f'sla{x}') for x in range(numRuns)])
        resultDF= resultDF.append(row ,ignore_index=True)
    return resultDF
    

path=r'Scenarios\ExperimentSet6\run2\sourceFiles'

alphas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
df = pd.DataFrame(columns = ['tardinessCost', 'travelTimeCost'])
for alpha in alphas:
    resultDF = runAlphaExp(path,alpha,5)
    row={'alpha':alpha,'tardCost':resultDF['Av_tardinessCost'],'travelCost':resultDF['Av_totalTravelTimeCost'] }
    df=df.append(row, ignore_index=True)
    

df.to_excel(rf'Scenarios\ExperimentSet6\run2\results.xlsx')