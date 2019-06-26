from time import *
from scheduler import Scheduler
import matplotlib.pyplot as plt

if __name__== '__main__':
    start=time()
    scheduler = Scheduler(layoutFile='outputDM.csv', agvFile='agvs.xlsx', requestFile='transportOrders2.xlsx', \
                        stationFile='stations.xlsx')
    scheduler.createGreedySequence()
    scheduler.createTaskSequence()
    scheduler.solveLP(printOutput=False)

    end = time()
    
    # scores.sort(reverse=True)
    # plt.plot([x[0] for x in scores],[x[1] for x in scores])
    plt.plot([1,2,3],[4,5,6])
    plt.ylabel('Score of solution')
    plt.xlabel('Iteration Number')
    plt.show()
    # sortedScore = sorted(scores,reverse=True)
    # plt.plot(sortedScore)
    # plt.ylabel('Score of solution')
    # plt.xlabel('Iteration Number')