# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:51:04 2019

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
'alpha':0.5,
'q':4,
'p':0.6,
'phi':2.2,
'chi':2.21,
'psi':2.75,
'r':4,
'k':6,
'alnsMethod':4
}
        
def createSensitivityAnalysisFiles(main_path, agvFilePath):
    agvdf = pd.read_excel(agvFilePath) 
    numPDNodes = 41
    reqCap = ['A,B', 'A,C', 'C,D', 'E']#'A,B' = lift heavy load, 'A,C'=Lift-light, 'C,D'=tow-light load, 'E'=use robot arm
    fos = 3 # factor of safety for time-window
    requestCosts = [1] # 1 - non-critical task, 10-critical tasks
    maxLayoutDistance = 308 * fos #meters
    minSpeed = 1 #m/sec
    timeHorizons = [1800,3600] # 30 mins, 60 mins, 2 hours, [1800,3600,7200]
    bnc = [60]
    tws= [0.2,0.8] # prob of tight time-windows, [0.2,0.5,0.8]
    numJobs = [20,40] # [20,40,60]
    agvFleetCases = [3,6]
    alphaRange = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for ho in timeHorizons:
        for tw in tws:
            for n in numJobs:
                df = pd.DataFrame(columns = ['Id','ArrivalTime','source','target','ept','ldt','capability','requestCost'])
                for r in range(n):
                    index = r #id
                    ept = random.randint(0,ho)
                    arrTime=ept
                    isTightTW = random.random()<tw
                    if isTightTW:
                        ldt = int(ept+(maxLayoutDistance/minSpeed)*(0.5+0.5*random.random()))
                    else:
                        ldt = int(ept+(maxLayoutDistance/minSpeed)*(1+random.random()))
                    cap = random.choice(reqCap)
                    source = random.randint(0,numPDNodes)
                    target = random.randint(0,numPDNodes)
                    requestCost = random.choice(requestCosts)

                    while (target - source)==0:
                        target = random.randint(0,numPDNodes)

                    row = {'Id':index,'ArrivalTime':arrTime,'source':source,'target':target,'ept':ept,'ldt':ldt,'capability':cap, 'requestCost':requestCost}
                    df=df.append(row, ignore_index=True)
                df=df.sort_values(by=['ept'])
                for agvcase in agvFleetCases:
                    for nc in bnc:
                        os.mkdir(rf'{main_path}\{ho}-{tw}-{n}-{agvcase}-{nc}')
                        os.mkdir(rf'{main_path}\{ho}-{tw}-{n}-{agvcase}-{nc}\DistanceMatrix')
                        os.mkdir(rf'{main_path}\{ho}-{tw}-{n}-{agvcase}-{nc}\TRs')
                        os.mkdir(rf'{main_path}\{ho}-{tw}-{n}-{agvcase}-{nc}\AGV')
                        os.mkdir(rf'{main_path}\{ho}-{tw}-{n}-{agvcase}-{nc}\Station')
                        copyfile(rf'Scenarios\SourceFiles\dm.csv',rf'{main_path}\{ho}-{tw}-{n}-{agvcase}-{nc}\DistanceMatrix\dm.csv')
                        copyfile(rf'Scenarios\SourceFiles\stations.xlsx',rf'{main_path}\{ho}-{tw}-{n}-{agvcase}-{nc}\Station\stations.xlsx')
                        df.to_excel(rf'{main_path}\{ho}-{tw}-{n}-{agvcase}-{nc}\TRs\trs.xlsx', index = False)
                        newAgvDf = copy.deepcopy(agvdf.iloc[:agvcase])
                        newAgvDf.loc[:,'upperThreshold']=nc
                        newAgvDf.to_excel(rf'{main_path}\{ho}-{tw}-{n}-{agvcase}-{nc}\AGV\agvs.xlsx', index = False)
                    

def runSensitivityAnalysis(main_path, numRuns=5, exp_duration=15):   
    cols = ['scenario_name','ho','tw','n','agvcase', 'upper_threshold']
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
             'n':splitScenario[2],'agvcase':splitScenario[3], 'upper_threshold':splitScenario[4]}
        print(f'Scenario:{scenario}')
        for run in range(numRuns):
            scheduler = Scheduler(layoutFile=rf'{main_path}\{scenario}\DistanceMatrix\dm.csv', agvFile=rf'{main_path}\{scenario}\AGV\agvs.xlsx', \
                              requestFile=rf'{main_path}\{scenario}\TRs\trs.xlsx', \
                            stationFile=rf'{main_path}\{scenario}\Station\stations.xlsx',hyperparams=defaultHyperparams)
            
            solution = scheduler.solve(exp_duration)
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
            bestTime = solution.get('bestScores')[-1][2]
            row[f'bestTime{run}']=bestTime
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
        row[f'Av_solutionTime']= sum([row.get(f'bestTime{x}') for x in range(numRuns)])/numRuns
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
        row[f'Std_solutionTim']=statistics.stdev([row.get(f'bestTime{x}') for x in range(numRuns)])
        #minimum values
        row[f'Min_tardiness'] = min([row.get(f'tardiness{x}') for x in range(numRuns)])
        row[f'Min_tardinessCost'] = min([row.get(f'tardinessCost{x}') for x in range(numRuns)])
        row[f'Min_totalTravel'] = min([row.get(f'totalTravel{x}') for x in range(numRuns)])
        row[f'Min_lt'] = min([row.get(f'lt{x}') for x in range(numRuns)])
        row[f'Min_ult'] = min([row.get(f'ult{x}') for x in range(numRuns)])
        row[f'Min_totalTravelTime'] = min([row.get(f'totalTravelTime{x}') for x in range(numRuns)])
        row[f'Min_totalTravelTimeCost'] = min([row.get(f'totalTravelTimeCost{x}') for x in range(numRuns)])
        row[f'Min_score'] = min([row.get(f'score{x}') for x in range(numRuns)])
        row[f'Max_sla']=max([row.get(f'sla{x}') for x in range(numRuns)])
        row[f'Min_solutionTime']=min([row.get(f'bestTime{x}') for x in range(numRuns)])

        resultDF= resultDF.append(row ,ignore_index=True)
    
    resultDF.to_excel(rf'{main_path}\results.xlsx')    
    
    return resultDF

def runSensitivityExp3(main_path, numRuns=5, exp_duration=15):
    cols = ['scenario_name','ho','tw','n','agvcase', 'upper_threshold']
    results = []
    scenarios = os.listdir(main_path)
    
    for scenario in scenarios:
        for run in range(numRuns):
            scheduler = Scheduler(layoutFile=rf'{main_path}\{scenario}\DistanceMatrix\dm.csv', agvFile=rf'{main_path}\{scenario}\AGV\agvs.xlsx', \
                              requestFile=rf'{main_path}\{scenario}\TRs\trs.xlsx', \
                            stationFile=rf'{main_path}\{scenario}\Station\stations.xlsx',hyperparams=defaultHyperparams)
            
            scheduler.solve(exp_duration)
            results.append(scheduler.getScheduleCostAnalysis())
    
    
    return results
        

path=r'Scenarios\Exact\spyder\sourceFiles'
agvFilePathHomo =r'Scenarios\SourceFiles\agvsHomogeneous.xlsx' #agv file for homogeneous fleet
agvFilePathHomoVarCost = r'Scenarios\SourceFiles\agvsHomogeneousVarCost.xlsx' 
agvFilePathHetero = r'Scenarios\SourceFiles\agvs.xlsx'

#createSensitivityAnalysisFiles(path, agvFilePath=agvFilePathHetero)
results = runSensitivityAnalysis(path,5,15)
#results3=runSensitivityExp3(path, 10, 15)
