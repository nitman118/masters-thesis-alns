# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 16:59:25 2019

@author: nitis
"""

import matplotlib.pyplot as plt

x = [5,6,7]
y1= [6,7,8]
y2 = [9,10,11]

plt.style.use('default')
fig, ax = plt.subplots(nrows = 1, ncols = 2, sharey='row', sharex = 'col', figsize=(3,1.5))
textstr = 'some text\nsomeother'
for i in range(2):
    ax[i].plot(x,y1, label = 'one', marker='o')
    ax[i].plot(x,y2, label = 'two', marker='o')
#    ax[i].text()
    h,l = ax[i].get_legend_handles_labels()
    
fig.text(1, 0.5,textstr)
fig.legend(h,l)  
