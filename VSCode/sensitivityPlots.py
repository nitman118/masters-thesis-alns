# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 15:31:27 2019

@author: nitis
"""

import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_it(**kwargs):
    path = kwargs.get('path')
    ticks = kwargs.get('ticks')
    xlabel = kwargs.get('xlabel')
    ylabel = kwargs.get('ylabel')
    titles = kwargs.get('pltTitles')
    figText = kwargs.get('figText')
    
    x = kwargs.get('x')
    y1 = kwargs.get('Matheuristic')
    y2 = kwargs.get('EDDBID')
    
    numPlots  = len(y1)
    fig, ax = plt.subplots(nrows=1, ncols=numPlots, sharey='row', sharex = 'col' ,figsize=(16,2.5))
    
    for i in range(numPlots):
        ax[i].plot(x[i], y1[i],label ='Matheuristic', marker = 'o')
        ax[i].plot(x[i], y2[i],label ='EDDBID', marker='o')
        ax[i].set_xlabel(xlabel)
        ax[i].set_xticks(ticks)
        ax[i].set_title(titles[i])
        h,l = ax[i].get_legend_handles_labels()
    ax[0].set_ylabel(ylabel)
    fig.text(1, 0.5,figText)
    fig.legend(h,l)
    fig.savefig(rf"{path}", bbox_inches = 'tight')
    
    
   
def sensitivityPlots(data, main_path):
    H= list(data['H'].unique())
    TW = list(data['TW'].unique())
    N=list(data['N'].unique())
    A = list(data['A'].unique())
    U = list(data['U'].unique())
    toPlot = [H, TW, N, A, U]
    lenParameters = int(len(toPlot))
    kwds = ['H', 'TW', 'N', 'A', 'U']

    for i in range(lenParameters):
        fvary0 = kwds[(i+1)%lenParameters]
        folder_name0 = f'_{fvary0}'
        os.mkdir(rf'{main_path}\{folder_name0}')
        for ch1 in toPlot[i%lenParameters]:
            kwd1 = kwds[toPlot.index(toPlot[i%lenParameters])]
            fvary1 = kwds[(i+1)%lenParameters]
            folder_name1 = '-'.join(sorted(f'{kwd1}_{ch1}'.split('-')))+f'_{fvary1}'
            os.mkdir(rf'{main_path}\{folder_name0}\{folder_name1}')
            for ch2 in toPlot[(i+1)%lenParameters]:
                kwd1 = kwds[toPlot.index(toPlot[i%lenParameters])]
                kwd2 = kwds[toPlot.index(toPlot[(i+1)%lenParameters])]
                fvary2 = kwds[(i+3)%lenParameters]
                folder_name2 = '-'.join(sorted(f'{kwd1}_{ch1}-{kwd2}_{ch2}'.split('-')))+f'_{fvary2}'
#                os.mkdir(rf'{main_path}\{folder_name0}\{folder_name1}\{folder_name2}')
                x = []
                matheuristic= []
                eddbid = []
                pltTitles = []
                path = rf'{main_path}\{folder_name0}\{folder_name1}\{folder_name2}.png'
                
                figText = f'{kwd1}={ch1}\n{kwd2}={ch2}'
                for ch3 in toPlot[(i+2)%lenParameters]:
                    kwd1 = kwds[toPlot.index(toPlot[i%lenParameters])]
                    kwd2 = kwds[toPlot.index(toPlot[(i+1)%lenParameters])]
                    kwd3 = kwds[toPlot.index(toPlot[(i+2)%lenParameters])]
                    fvary3 = kwds[(i+3)%lenParameters]
                    folder_name3 = '-'.join(sorted(f'{kwd1}_{ch1}-{kwd2}_{ch2}-{kwd3}_{ch3}'.split('-')))+f'_{fvary3}'
#                    os.mkdir(rf'{main_path}\{folder_name0}\{folder_name1}\{folder_name2}\{folder_name3}')
                    
                    for ch4 in toPlot[(i+3)%5]:
                        kwd1 = kwds[toPlot.index(toPlot[i%lenParameters])]
                        kwd2 = kwds[toPlot.index(toPlot[(i+1)%lenParameters])]
                        kwd3 = kwds[toPlot.index(toPlot[(i+2)%lenParameters])]
                        kwd4 = kwds[toPlot.index(toPlot[(i+3)%lenParameters])]
                        toPlotDf = data[(data[kwd1]==ch1)&(data[kwd2]==ch2)&(data[kwd3]==ch3)&(data[kwd4]==ch4)]
                        xParam = kwds[(i+4)%lenParameters]
                        x.append(toPlotDf[xParam])
                        xlabel = xParam
                        matheuristic.append(toPlotDf['Avg'])
                        eddbid.append(toPlotDf['EDDBID'])
                        ticks = toPlotDf[xParam]
                        pltTitles.append(f'{kwd3}:{ch3}, {kwd4}:{ch4}')
                        
                    plot_it(x=x,Matheuristic=matheuristic, EDDBID = eddbid, xlabel = xlabel,\
                            ylabel = 'Average Score, Avg', ticks = ticks, path = path, pltTitles=pltTitles, \
                            figText = figText)

main_path = r'Scenarios\ExperimentSet1\Run2\Results\Plots2'
data = pd.read_excel(r'Scenarios\ExperimentSet1\Run2\Results\run2ResultsCombined.xlsx')
data=data.sort_values(by=['H', 'TW', 'N', 'A', 'U'])
sensitivityPlots(data, main_path)