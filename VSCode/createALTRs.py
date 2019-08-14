# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:09:40 2019
@author: nitis
"""
import pandas as pd
import os
from shutil import copyfile
import random

path_from=rf'Scenarios\ExperimentSet5\run1(alpha-1)\spyder\sourceFiles'
path_to=rf'Scenarios\ExperimentSet5\run1(alpha-1)\AL\sourceFiles'

def createFiles(path_from,path_to, agvCharacterStart=3):
    '''
    @path_to : destination path for file
    '''
    scenario_folders = os.listdir(path_from)
    excludedNames=['results.xlsx']
    scenario_folders=[s for s in scenario_folders if s not in excludedNames]
    scenario_folders_filtered = set(['-'.join(f.split('-')[:agvCharacterStart]) for f in scenario_folders])
    scenario_folders_filtered=list(scenario_folders_filtered)
    scenario_folders_filtered.sort()
#    print(scenario_folders_filtered)
    for folder in scenario_folders_filtered:
        selected_folder = random.choice([s for s in scenario_folders if folder in s])
        copyfile(rf'{path_from}\{selected_folder}\TRs\trs.xlsx', rf'{path_to}\{folder}.xlsx')
    df = pd.DataFrame({'scenario_name':scenario_folders_filtered})
    df=df.sort_values(by=['scenario_name'])
    df.to_excel(rf'{path_to}\demandSummary.xlsx', index=True, index_label='ID')
    
createFiles(path_from,path_to,3)
        
        
    