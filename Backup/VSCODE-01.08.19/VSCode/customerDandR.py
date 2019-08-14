import numpy as np
import random
import copy
from task import Task

class ALNSCustomerDestroyAndRepairMethods():
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.agvTaskList = list()
        self.setAGVRequestList()
        self.destroyedRequests = list()
        self.agvSqScore=dict()
        self.q=int(self.scheduler.hyperparams.get('q')) #no. of tasks to remove/destroy
        self.p=self.scheduler.hyperparams.get('p') #probability / randomness
        self.phi=self.scheduler.hyperparams.get('phi')
        self.chi=self.scheduler.hyperparams.get('chi')
        self.psi=self.scheduler.hyperparams.get('psi')
        self.r = int(self.scheduler.hyperparams.get('r',2))
        self.k=int(self.scheduler.hyperparams.get('k',4))
        
        
        
    def setAGVRequestList(self):
        '''
        reset this whenever any of the function is called
        '''
        self.agvTaskList=list() #initialize a new list
        for l in list(self.scheduler.taskList.values()):
            for r in l:
                self.agvTaskList.append(r)
    
    
    def removeTasksFromSchedulerTaskList(self,D):
        '''
        remove requests in D from scheduler's taskList
        '''
        for key,val in self.scheduler.taskList.items():
            self.scheduler.taskList[key]=[v for v in val if v not in D]
        #to keep track of destroyed requests
        self.destroyedRequests=D
        #check for consecutive charge tasks and remove one of them, is handled in scheduler createTaskSequence
#         self.checkConsecutiveChargeTasks()
        
    def destroyWorstTardinessCustomers(self):
        '''
        remove the task that has highest tardiness, from the current schedule
        '''
        tasksInSchedule = list()
        self.setAGVRequestList()
        
        for agvList in self.scheduler.taskSchedule.values():
            for taskSchedule in agvList:
                tasksInSchedule.append(taskSchedule)
            
        tasksInSchedule=list(filter(lambda x:x['taskType']=='TO',tasksInSchedule)) #keep only TOs
        
        tasksInSchedule.sort(key=lambda x:x['D']-x['ldt'], reverse=True) #sort based on tasks that have max tardiness
        
        q = min(self.q,len(tasksInSchedule)) #protection against 0 tasks?
        
        taskIdsToRemove = [x['taskId'] for x in tasksInSchedule[:q]]
        
        D=[x for x in self.agvTaskList if x.getTaskId() in taskIdsToRemove]
        
        self.removeTasksFromSchedulerTaskList(D)
        
    
    
    def destroyShawDistance(self):
        self.shawRemoval(phi=None, chi=0, psi=0)
        
    def destroyShawTimeWindow(self):
        self.shawRemoval(phi=0, chi=None, psi=0)
        
    def destroyShawCapability(self):
        self.shawRemoval(phi=0, chi=0, psi=None)
        
    def shawRemoval(self,phi=None, chi=None, psi=None):
        '''
        shaw removal heuristic
        a low value of p corresponds to much randomness
        '''
        self.setAGVRequestList()
        agvTos=list(filter(lambda x:x.taskType=='TO',self.agvTaskList))
        randomRequest=np.random.choice(agvTos)
        D = [randomRequest]
        
        
        
        phi = phi if phi==0 else self.phi
        chi = chi if chi==0 else self.chi
        psi = psi if psi==0 else self.psi
        
       
        
        while len(D)<self.q:
            r=np.random.choice(D)# randomly select a request from D
            L=[req for req in agvTos if req not in D]
            L.sort(key=lambda x:self.shawRelatednessFunction(r,x,phi, chi, psi)) # do pairwise comparison and sort based on shaw
            # relatedness function
            y = random.random()
            pos = int((y**self.p)*len(L))
            D.append(L[pos])
        self.removeTasksFromSchedulerTaskList(D)
        
   

    def shawRelatednessFunction(self, task1, task2, phi, chi, psi):
        '''
        this function calculates a relatedness measure between requests
        '''
        #distance relatedness
        distance = self.scheduler.layout.getNormalizedDistanceFromNode(task1.source, task2.source)+\
        self.scheduler.layout.getNormalizedDistanceFromNode(task1.dest, task2.dest)
        #time-window relatedness
        timeWindows = abs(self.scheduler.getNormalizedEPT(task1)-self.scheduler.getNormalizedEPT(task2))+\
        abs(self.scheduler.getNormalizedLDT(task1)-self.scheduler.getNormalizedLDT(task2)) 
        #capability relatedness
        agvs1 = {agv for agv in self.scheduler.agvs if self.scheduler.checkCap(task1,agv)} # define a set
        agvs2 = {agv for agv in self.scheduler.agvs if self.scheduler.checkCap(task2,agv)} # define a set
        
        intersectionLen = len(agvs1.intersection(agvs2)) #set1 Intersect set2
        minLen = min(len(agvs1), len(agvs2))
        capability = (1-(intersectionLen/minLen))
        
        #measure relatedness based on phi, chi and psi
        R = phi*(distance) + chi*(timeWindows) + psi*(capability)
        
        #return value
        return R
    
    def destroyRandomTasks(self):
        '''
        destroys q tasks at random from agv tasklist
        '''
        self.setAGVRequestList()
        agvTos=list(filter(lambda x:x.taskType=='TO',self.agvTaskList))
        r=min(self.r,len(agvTos)) # to ensure feasibility
        D=random.sample(agvTos,r) # select q TOs randomly from task list, without replacement
        self.removeTasksFromSchedulerTaskList(D)

        
    def repairInsertRandomTasks(self):
        '''
        inserts destroyed tasks randomly in scheduler's tasklist 
        '''
        modifiedAgvs = set()
        for task in self.destroyedRequests:
            agv_count = [agv for agv in self.scheduler.agvs if self.scheduler.checkCap(task,agv)]
            randomAGV = random.choice(agv_count)
            taskList = self.scheduler.taskList.get(randomAGV.agvId)
            randomPosition = int(random.random()*len(taskList))
            taskList.insert(randomPosition,task)
            modifiedAgvs.add(randomAGV)
            #store the agv whose tasklist is changed
        #CHECK FEASIBILITY OF SCHEDULE by passing agvs that were modified
        for agv in modifiedAgvs:
            self.makeTaskListFeasible(agv)
        
    def makeTaskListFeasible(self, agv):
        '''
        returns true if agv schedule is feasible, else false
        '''
        isFeasible=True
#         breakpoint()
        currentNode=self.scheduler.getStartNode(agv)
        currentCharge=self.scheduler.getCharge(agv)
        tempTaskList=copy.deepcopy(self.scheduler.taskList[agv.agvId])
        for task,i in zip(tempTaskList,range(len(tempTaskList))):
            if not isFeasible:
                dists = [self.scheduler.layout.getDistanceFromNode(currentNode,station.getNode()) 
                         for station in self.scheduler.chargingStations]
                optIndex = dists.index(min(dists))
                nearestChargeNode = self.scheduler.chargingStations[optIndex].getNode()
                chargeTask = Task(999,'C','X','X')
                self.scheduler.taskList[agv.agvId].insert(i,chargeTask)
                currentCharge=100
                currentNode=nearestChargeNode
            prevToCurr=agv.getDischargeRate()*(self.scheduler.layout.getDistanceFromNode(currentNode,task.source)/agv.speed)
            currToCurr=agv.getDischargeRate()*(self.scheduler.layout.getDistanceFromNode(task.source,task.dest)/agv.speed)
            currentNode=task.dest
            currentCharge=currentCharge-prevToCurr-currToCurr if task.taskType!='C' else 100
            if i<len(tempTaskList)-1:
                nextDisc=agv.getDischargeRate()*(self.scheduler.layout.getDistanceFromNode(task.dest,tempTaskList[i+1].source)/agv.speed)
                nextCharge=currentCharge-nextDisc
            else:
                nextCharge=currentCharge
            isFeasible=False if nextCharge<30 else True
    
    def repairInsertGreedyTasks(self):
        '''
        greedily inserts tasks into agv tasklists
        '''
        modifiedAgvs=set()
        self.setAgvScore()
        while len(self.destroyedRequests)>0:
            scores=[]
            for task in self.destroyedRequests:
                agv_count = [agv for agv in self.scheduler.agvs if self.scheduler.checkCap(task,agv)]
                for agv in agv_count:
                    pos,cost = self.findGreedyPosition(agv, task) #TODO
                    scores.append((cost,agv,pos,task))
            scores.sort(key=lambda x:x[0]) #find argmin cost
            agv,pos,task=scores[0][1],scores[0][2],scores[0][3]
            self.scheduler.taskList.get(agv.agvId).insert(pos,task)
            modifiedAgvs.add(agv)
            self.updateAgvScore(agv)
            self.destroyedRequests.remove(task)
        for agv in modifiedAgvs:
            self.makeTaskListFeasible(agv)
            
                
    def findGreedyPosition(self, agv, task, isKregret=False):
        '''
        Returns the best position and cost in terms of unloaded travel cost and tardiness for inserting the task
        in the tasklist of a particular agv
        '''
        agvTaskListCopy = copy.deepcopy(self.scheduler.taskList.get(agv.agvId))
        scores=[]
        unchangedAgvScore=self.getUnchangedAgvScore(agv)
        for pos in range(len(agvTaskListCopy)+1):
            score = unchangedAgvScore + self.createGreedySchedule(agv,task,pos)
            scores.append((score,agv,task,pos))
#             if score>prevScore:
#                 break
        scores.sort(key=lambda x:x[0])
        return (scores[0][3],scores[0][0]) if not isKregret else scores
            
    def createGreedySchedule(self, agv,task,pos):
        taskListCopy = copy.deepcopy(self.scheduler.taskList)
        if pos<len(taskListCopy.get(agv.agvId)):
            taskListCopy.get(agv.agvId).insert(pos, task)
        else:
            taskListCopy.get(agv.agvId).append(task)
        return self.calculateGreedyScore(agv,taskListCopy.get(agv.agvId))
            
            
    def calculateGreedyScore(self,agv,taskListCopy, alpha=0.5):
        tardinessTime=0
        totalUnloadedTime=0
        a = agv
        chargeRate = a.getChargeRate()
        dischargeRate = a.getDischargeRate()
        agvSpeed = a.getSpeed()
        currentNode = a.startNode
        currentCharge = a.charge
        runTime=0
        for n,task in enumerate(taskListCopy):
            if task.taskType=='C':
                nearestChargeNode = self.scheduler.getNearestChargeLocation(currentNode)      
                unloadedTT=self.scheduler.layout.getDistanceFromNode(currentNode,nearestChargeNode)*(1/agvSpeed)
                currentCharge-=dischargeRate*unloadedTT
                reqdCharge=0
                currentNodeC=nearestChargeNode
                for t in range(n+1, len(taskListCopy)):
                    if taskListCopy[t].taskType!='C':
                        unloadedTTC=self.scheduler.layout.getDistanceFromNode(currentNodeC,taskListCopy[t].source)*(1/agvSpeed)
                        loadedTTC=self.scheduler.layout.getDistanceFromNode(taskListCopy[t].source,taskListCopy[t].dest)*(1/agvSpeed)
                        currentNodeC=taskListCopy[t].dest
#                             print(f"CURRENTNODE:{currentNodeC}")
                        reqdCharge+=dischargeRate*(unloadedTTC+loadedTTC)
                    else: break;
                runTime+=unloadedTT+min(reqdCharge,100-currentCharge)/chargeRate
                currentNode=nearestChargeNode
                currentCharge+=reqdCharge
                currentCharge=min(currentCharge,100)
            else:
                unloadedTT=self.scheduler.layout.getDistanceFromNode(currentNode,task.source)*(1/agvSpeed)
                loadedTT=self.scheduler.layout.getDistanceFromNode(task.source,task.dest)*(1/agvSpeed)
                currentCharge-=dischargeRate*(unloadedTT+loadedTT)
                if runTime<task.ept:
                    currentCharge+=(task.ept-unloadedTT-runTime)/chargeRate if n>0 and taskListCopy[n-1].taskType=='C' else currentCharge
                    currentCharge=min(currentCharge,100)
                    runTime=task.ept-unloadedTT
                runTime+=unloadedTT+self.scheduler.getMhtById(task.source)+loadedTT+self.scheduler.getMhtById(task.dest)
                currentNode=task.dest
                tardinessTime+=max(0,runTime-task.ldt)
            totalUnloadedTime+=(unloadedTT*a.travelCost)
        return alpha*totalUnloadedTime+(1-alpha)*tardinessTime
    
    def repairKRegret(self):
        
        modifiedAgvs=set()
        self.setAgvScore()
        while len(self.destroyedRequests)>0:
            scores=[]
            for task in self.destroyedRequests:
                agv_count = [agv for agv in self.scheduler.agvs if self.scheduler.checkCap(task,agv)]
                tempScore=[]
                for agv in agv_count:
                    tempScore.extend(self.findGreedyPosition(agv,task,isKregret=True))
                tempScore.sort(key=lambda x:x[0])
                s=sum([tempScore[i+1][0]-tempScore[0][0] for i in range(min(self.k-1,len(tempScore)-1))])
                scores.append([s,tempScore[0][1],tempScore[0][2],tempScore[0][3]])
            scores.sort(key=lambda x:x[0], reverse=True) #find argmin cost
            agv,task,pos=scores[0][1],scores[0][2],scores[0][3]
            self.scheduler.taskList.get(agv.agvId).insert(pos,task)
            modifiedAgvs.add(agv)
            self.updateAgvScore(agv)
            self.destroyedRequests.remove(task)
        for agv in modifiedAgvs:
            self.makeTaskListFeasible(agv)
        
    def repairGreedyEDDInsert(self):
        '''
        this algorithm will insert destroyed requests greedily into agv based on EDD of the destroyed jon
        AGV<- TO1,TO2,C,TO3...
        To Insert : TOX
        IS TO1.edd > TOX.edd ? Insert at TO1's index else check for TO2.....
        '''
        modifiedAgvs=set()
        self.setAgvScore()
        while len(self.destroyedRequests)>0:
            scores=[]
            for task in self.destroyedRequests:
                agv_count = [agv for agv in self.scheduler.agvs if self.scheduler.checkCap(task,agv)]
                for agv in agv_count:
                    pos,cost = self.findGreedyEDDPosition(agv, task) #TODO
                    scores.append((cost,agv,pos,task))
            scores.sort(key=lambda x:x[0]) #find argmin cost
            agv,pos,task=scores[0][1],scores[0][2],scores[0][3]
            self.scheduler.taskList.get(agv.agvId).insert(pos,task)
            modifiedAgvs.add(agv)
            self.updateAgvScore(agv)
            self.destroyedRequests.remove(task)
        for agv in modifiedAgvs:
            self.makeTaskListFeasible(agv)
            
    def updateAgvScore(self, agv):
        self.agvSqScore[agv.agvId] = self.calculateGreedyScore(agv,self.scheduler.taskList.get(agv.agvId))
            
    def setAgvScore(self):
        self.agvSqScore={}
        for a in self.scheduler.agvs:
            self.agvSqScore[a.agvId]=self.calculateGreedyScore(a,self.scheduler.taskList.get(a.agvId))
    
    def getUnchangedAgvScore(self,agv):
        return sum([self.agvSqScore[a] for a in self.agvSqScore.keys() if a!=agv.agvId])
         
    def findGreedyEDDPosition(self, agv, task, isKregret=False):
        '''
        Returns the best position and cost in terms of unloaded travel cost and tardiness for inserting the task
        in the tasklist of a particular agv
        '''
        agvTaskListCopy = copy.deepcopy(self.scheduler.taskList.get(agv.agvId))
        scores=[]
        unchangedAgvScore=self.getUnchangedAgvScore(agv)
        inserted=False
        for pos in range(len(agvTaskListCopy)):
            taskListCopy = copy.deepcopy(self.scheduler.taskList)
            if taskListCopy.get(agv.agvId)[pos].taskType!='C' and taskListCopy.get(agv.agvId)[pos].getLDT() >= task.getLDT():
                taskListCopy.get(agv.agvId).insert(pos, task)
                score = unchangedAgvScore + self.calculateGreedyScore(agv,taskListCopy.get(agv.agvId))
                scores.append((score,agv,task,pos))
                inserted=True
                break
        if not inserted:
            taskListCopy = copy.deepcopy(self.scheduler.taskList)
            taskListCopy.get(agv.agvId).append(task)
            score = unchangedAgvScore + self.calculateGreedyScore(agv,taskListCopy.get(agv.agvId))
            scores.append((score,agv,task,taskListCopy.get(agv.agvId).index(task))) #if not inserted pos = last element or size
        scores.sort(key=lambda x:x[0])
        return (scores[0][3],scores[0][0]) if not isKregret else scores
    
  