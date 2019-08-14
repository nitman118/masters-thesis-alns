from scheduler import Scheduler
import matplotlib.pyplot as plt

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

main_path=r'Scenarios\Exact\spyder\sourceFiles'
scenario =r'1800-0.2-7-3-60'

scheduler = Scheduler(layoutFile=rf'{main_path}\{scenario}\DistanceMatrix\dm.csv', agvFile=rf'{main_path}\{scenario}\AGV\agvs.xlsx', \
                              requestFile=rf'{main_path}\{scenario}\TRs\trs.xlsx', \
                            stationFile=rf'{main_path}\{scenario}\Station\stations.xlsx',hyperparams=defaultHyperparams)

solution =scheduler.solve(5)
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
