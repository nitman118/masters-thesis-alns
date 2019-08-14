from scheduler import Scheduler
import matplotlib.pyplot as plt

defaultHyperparams = {'psi1':0.67,
'psi2':0.5,
'psi3':0.51,
'lambdaP':0.65,
'a':6,
'b':4,
'alpha':0.8,
'q':3,
'p':0.6,
'phi':2.95,
'chi':1.6,
'psi':0.71,
'r':3,
'k':3,
'alnsMethod':4
}

scheduler = Scheduler(layoutFile='outputDM.csv', agvFile='agvs.xlsx', \
                      requestFile='transportOrders2.xlsx', \
                    stationFile='stations.xlsx',hyperparams=defaultHyperparams)

solution =scheduler.solve(15)
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

plt.plot([x[0] for x in solution.get('bestScores')],[x[1] for x in solution.get('bestScores')])
plt.xticks([x[0] for x in solution.get('bestScores')],[x[2] for x in solution.get('bestScores')], rotation ='vertical')
plt.plot([x[0] for x in solution.get('scores')],[x[1] for x in solution.get('scores')])
plt.ylabel('Score of solution')
plt.xlabel('Iteration Number')
print(f'Best solution found at time:{bestScores[-1][2]:.3f} seconds')
print(scheduler.getScheduleKPI())

#plt.show()
#
#plt.figure()
#plt.plot(solution.get('scorealns'), label='alns')
#plt.legend()
#plt.show()
#plt.figure()
#plt.plot(solution.get('scorenormal'),label='normal')
#plt.legend()
#
#plt.show()
#
#plt.figure()
#plt.plot([a-b for a,b in zip(solution.get('scorenormal'),solution.get('scorealns'))])
