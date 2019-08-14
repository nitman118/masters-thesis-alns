numPDNodes = 41
reqCap = ['A,B','A,C','C,D','E']#'A,B' = lift heavy load, 'A,C'=Lift-light, 'C,D'=tow-light load, 'E'=use robot arm
fos = 3 # factor of safety for time-window
requestCosts = [1] # 1 - non-critical task, 10-critical tasks
maxLayoutDistance = 308 * fos #meters
minSpeed = 1 #m/sec
timeHorizons = [1800,3600] # 30 mins, 60 mins, 2 hours, [1800,3600,7200]
bnc = [60,80,100]
tws= [0.2,0.5,0.8] # prob of tight time-windows, [0.2,0.5,0.8]
numJobs = [20,40,60] # [20,40,60]
agvFleetCases = [3,6,9]