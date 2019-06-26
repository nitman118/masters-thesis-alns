import copy
from task import Task
import random

class ALNSStationDestroyAndRepairMethods():
    
    def __init__(self):
        self.agvInfo={}
    
    def setAGVInfo(self,agv):
        self.agvInfo['agvId']=agv.agvId
        self.agvInfo['charge']=agv.charge
        self.agvInfo['startNode']=agv.startNode
        self.agvInfo['release']=(0, agv.startNode) # (time, node)
        self.agvInfo['state']=agv.state
    
    def getCurrentReleaseNode(self):
        '''
        returns agv's current release node
        '''
        return self.agvInfo['release'][1]
    
    def setCurrentReleaseNode(self, releaseNode):
        '''
        sets current release node
        '''
        self.agvInfo['release']= (self.agvInfo['release'][0], releaseNode)
    
    def getCurrentReleaseTime(self):
        '''
        returns current release time
        '''
        return self.agvInfo['release'][0]
    
    def setCurrentReleaseTime(self, releaseTime):
        '''
        set current release time
        '''
        self.agvInfo['release'] = (releaseTime,self.agvInfo['release'][1])
    
    def getState(self):
        '''
        gets state of agv
        '''
        return self.agvInfo['state']
        
    def setState(self,state):
        '''
        sets state of agv
        '''
        self.agvInfo['state']=state
        
    def setCharge(self, charge):
        '''
        Sets the agv charge
        '''
        self.agvInfo['charge'] = min(100,charge)
        
    def getCharge(self):
        '''
        Returns AGV Charge in %age
        '''
        return self.agvInfo['charge']
    
    def getStartNode(self,agv):
        '''
        Returns agv's initial/start node'''
        return self.agvInfo['startNode']
        
        
    def destroyRandomCharge(self,agv, scheduler):
        '''
        randomly remove a charging task from sequence of tasks of a random agv
        '''
        tList = scheduler.taskList.get(agv.agvId)
        chargeTasks = list(filter(lambda x:x.taskType=='C', tList))
        if chargeTasks:
            taskToRemove = random.choice(chargeTasks)
            tList.remove(taskToRemove)
            
    def destroyAllCharge(self,agv,scheduler):
        '''
        destroy all the charging tasks of a random agv
        '''
        tList = scheduler.taskList.get(agv.agvId)
        tList[:] = list(filter(lambda x:x.taskType=='TO', tList))
        
        
    def destroyWorstCharge(self,agv, scheduler):
        '''
        Destroys the worst charge task from a set of charge tasks
        '''
        
        tList = scheduler.taskList.get(agv.agvId) # get the taskList of the agv
        
        chargeTasks = list(filter(lambda x:x.taskType=='C' or x.taskType=='NC',tList)) # find all charge tasks
        
        if chargeTasks: # if there are charge tasks
            toRemove = None # initialize empty object
            minCharge = 1000
            
            for c,chargeTask in enumerate(chargeTasks): 
                indexOfChargeTask = chargeTask.taskIndex # find index of charge task
                indexOfNextTask = indexOfChargeTask+1 # index of next task is required to calculate gain in charge
                try:
                    currentTask = scheduler.getTaskfromTaskScheduleByIndex(index=indexOfChargeTask, agv=agv) 
                    nextTask = scheduler.getTaskfromTaskScheduleByIndex(index = indexOfNextTask,agv=agv)
                    deltaCharge = nextTask['B']-currentTask['B']
                    if deltaCharge<minCharge: #gain in battery level during the trip is lesser than minimum
                        minCharge = deltaCharge
                        toRemove = chargeTask
                except (IndexError, TypeError) as e:
                    #if the index error occurs, since the charge task is the last task of the taskList, ignore that \
                    # charge task for evaluation
                    pass
                
            if toRemove:
                tList.remove(toRemove)
    
    
    def repairInsertAllCharge(self,agv,taskList,scheduler):
        '''
        method that modifies taskList by inserting a charge task after every task in a given taskList
        '''
        #TODO : maybe if the agv is in 'C' state, modify the repair to have a charge request first?
        taskList[:]=list(filter(lambda x:x.taskType=='TO',taskList)) # this way reference to original object is maintained
           
        taskCopy = copy.deepcopy(taskList) # create a copy
        
        taskList.clear() #clear original list
        
        for t in taskCopy:
            taskList.append(t)
            cTask = Task(999,"C",'X','X')
            taskList.append(cTask)
        
            
    def repairInsertCCharge(self,agv,taskList,scheduler,threshold,ncc=False):
        '''
        this greedy repair method repairs the given taskList by inserting critical charge tasks wherever required
        '''
        scheduler.setAGVInfo() # reset the agv info of scheduler
        self.setAGVInfo(agv) # make a new agv info object
        taskCopy = copy.deepcopy(taskList) # create a taskList copy from which tasks are picked
        taskCopyLen= len(taskCopy) # used in if condition later
        taskList.clear() # delete current tasks and start repairing
#         breakpoint()
        for t,task in enumerate(taskCopy):
            '''
            condition 1: if there is sufficient charge and agv is not in charge state and the current task is of type TO
            elif->if: ensures that the loop is continued from the next loop iteration only and only if the next task is a 
            charge task and the current task is also not a TO, other
            '''
            if self.getCharge() >= threshold and self.getState()=='N' and task.taskType=='TO':
                self.addTaskToTaskList(agv,task,taskList,scheduler)
            
            elif self.getCharge()<threshold and self.getState()=='N' or task.taskType!='TO':
                
                if t<taskCopyLen-1 and taskCopy[t+1].taskType !='TO' and task.taskType!='TO': # or in other words next task is some charging task
                    if ncc:
                        threshold = agv.LOWER_THRESHOLD
                    continue
                else:
                    #since the next task is a TO, add a charge task
                    self.addChargeTask(agv,task,taskList,scheduler)
                
            if self.getState()=='C' and task.taskType=='TO':
                #update the charge at time of new request, if sufficient charge is present, change state, add the task
                self.addTaskToTaskList(agv,task,taskList,scheduler)            
     
    def repairInsertNCCharge(self,agv,taskList,scheduler):
        '''
        this function inserts NC tasks greedily
        '''
        self.repairInsertCCharge(agv,taskList,scheduler,threshold=agv.UPPER_THRESHOLD)
        
        
    def repairInsertNCandCCharge(self, agv, taskList,scheduler):
        '''
        this function repairs the schedule by inserting one NC task and the rest C tasks
        '''
        self.repairInsertCCharge(agv,taskList,scheduler,threshold=agv.UPPER_THRESHOLD,ncc=True)
        
        
    def addChargeTask(self,agv,task,taskList,scheduler):
        
        if task.taskType =='TO':
            
            chargeTask = Task(999,'C','X','X')
            taskList.append(chargeTask)
            self.setState('C') #charging
            nearestChargeNode=scheduler.getNearestChargeLocation(self.getCurrentReleaseNode())
            travelDist = scheduler.layout.getDistanceFromNode(self.getCurrentReleaseNode(),nearestChargeNode)
            drivingTime = travelDist/agv.speed # time spent in driving
            
            self.setCharge(self.getCharge() - (agv.dischargeRate * drivingTime))
            self.setCurrentReleaseNode(nearestChargeNode)
            self.setCurrentReleaseTime(self.getCurrentReleaseTime()+drivingTime)
        
        else:
            
            travelDist = scheduler.layout.getDistanceFromNode(self.getCurrentReleaseNode(),task.dest)
            drivingTime = travelDist/agv.speed # time spent in driving
            taskList.append(task)
            self.setState('C') #charging
            self.setCharge(self.getCharge() - (agv.dischargeRate * drivingTime))
            self.setCurrentReleaseNode(task.dest)
            self.setCurrentReleaseTime(self.getCurrentReleaseTime()+drivingTime)
        
     
    
    def addTaskToTaskList(self,agv,task,taskList, scheduler):
        
        travelDist = scheduler.layout.getDistanceFromNode(self.getCurrentReleaseNode(), task.source) + \
        scheduler.layout.getDistanceFromNode(task.source,task.dest)
        
        srcStation = list(filter(lambda x:x.nodeId==task.source,scheduler.stations))[0] # src station
        dstStation = list(filter(lambda x:x.nodeId==task.dest,scheduler.stations))[0] # destination station
        drivingTime = travelDist/agv.speed # time spent in driving
        
        if self.getState()=='N':
            travelTime = drivingTime+srcStation.mhTime+dstStation.mhTime
            taskList.append(task)
            self.setCurrentReleaseNode(task.dest)
            self.setCurrentReleaseTime(max(self.getCurrentReleaseTime(),task.ept)+travelTime)
            self.setCharge(self.getCharge() - (agv.dischargeRate * drivingTime))
            self.setState('N')
        
        elif self.getState()=='C':
            minChargeTime = (agv.LOWER_THRESHOLD - self.getCharge())/agv.chargeRate # time to reach LOWER_THRESHOLD charge level
            minChargeAbsTime = self.getCurrentReleaseTime() + max(0,minChargeTime)  # the absolutime time (in sec) at which AGV becomes 30% charged
            
            if task.ept >= minChargeAbsTime :
                # add the task but update charge based on delta
                chargeTime = (task.ept - self.getCurrentReleaseTime())
                self.setCharge(self.getCharge() + (chargeTime * agv.chargeRate)) # charge after charging till task's ept
                self.setCurrentReleaseTime(self.getCurrentReleaseTime()+chargeTime)
                travelTime = drivingTime + srcStation.mhTime + dstStation.mhTime # total time including material handling
#                 agv.taskList.append(task) # REMOVE LATER
                taskList.append(task)
                self.setCurrentReleaseNode(task.dest)
                self.setCurrentReleaseTime(max(self.getCurrentReleaseTime(),task.ept)+travelTime)
                self.setCharge(self.getCharge() - (agv.dischargeRate * drivingTime) )
                self.setState('N')
                
            elif task.ept < minChargeAbsTime:
                self.setCurrentReleaseTime(minChargeAbsTime)
                travelTime = drivingTime + srcStation.mhTime + dstStation.mhTime # total time including material handling
#                 agv.taskList.append(task) #REMOVE LATER
                taskList.append(task)
                self.setCurrentReleaseNode(task.dest)
                self.setCurrentReleaseTime(max(self.getCurrentReleaseTime(),task.ept)+travelTime)
                self.setCharge(self.getCharge() - (agv.dischargeRate * drivingTime) )
                self.setState('N')
        
        
    