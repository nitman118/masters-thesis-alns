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
        
    def setAGVRequestList(self):
        for l in list(self.scheduler.taskList.values()):
            for r in l:
                self.agvTaskList.append(r)
    
#     def checkConsecutiveChargeTasks(self):
        
#         for key,val in self.scheduler.taskList.items():
#             toRemove=[]
#             for i in range(len(val)-1):
#                 if val[i].taskType=='C'and val[i+1].taskType=='C' and i!=0:
#                     toRemove.append(val[i+1])
#                     val[i].source = val[i-1].dest
#                 elif val[i].taskType=='C' and val[i+1].taskType=='C' and i==0:
#                     toRemove.append(val[i+1])
#                     val[i].source = self.scheduler.getStartNode(agvs[key])
                    
#             val =[v for v in val if v not in toRemove]
    
    
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
        
    def destroyShawDistance(self):
        self.shawRemoval(q=2,p=0.6,phi=1, chi=0, psi=0)
        
    def destroyShawTimeWindow(self):
        self.shawRemoval(q=2,p=0.6,phi=0, chi=1, psi=0)
        
    def destroyShawCapability(self):
        self.shawRemoval(q=2,p=0.6,phi=0, chi=0, psi=1)
        
    def shawRemoval(self,q=2,p=0.6,phi=1, chi=1, psi=1):
        '''
        shaw removal heuristic
        a low value of p corresponds to much randomness
        '''
        self.setAGVRequestList()
        agvTos=list(filter(lambda x:x.taskType=='TO',self.agvTaskList))
        
        randomRequest=np.random.choice(agvTos)
#         agvId=np.random.randint(low=0,high=len(self.agvTaskLists))
#         randomRequest = np.random.choice(list(filter(lambda x:x.taskType=='TO',self.agvTaskLists[agvId])))
        
        D = [randomRequest]
        while len(D)<q:
            r=np.random.choice(D)# randomly select a request from D
            L=[req for req in agvTos if req not in D]
            L.sort(key=lambda x:self.shawRelatednessFunction(r,x,phi=1, chi=1, psi=1)) # do pairwise comparison and sort based on shaw
            # relatedness function
            y = random.random()
            pos = int((y**p)*len(L))
            D.append(L[pos])
            
        #remove requests in D from scheduler's taskList
        print(D)
        self.removeTasksFromSchedulerTaskList(D)
        
   

    def shawRelatednessFunction(self, task1, task2, phi=1, chi=1, psi=1):
        '''
        this function calculates a relatedness measure between requests
        '''
        #distance relatedness
        distance = self.scheduler.layout.getNormalizedDistanceFromNode(task1.source, task2.source)+\
        self.scheduler.layout.getNormalizedDistanceFromNode(task1.dest, task2.dest)
        #time-window relatedness
        timeWindows = abs(self.scheduler.getNormalizedEPT(task1)-self.scheduler.getNormalizedEPT(task2))+abs(self.scheduler.getNormalizedLDT(task1)-self.scheduler.getNormalizedLDT(task2)) 
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
    
    def destroyRandomTasks(self,q=0.05):
        '''
        destroys q tasks at random from agv tasklist
        '''
        self.setAGVRequestList()
        agvTos=list(filter(lambda x:x.taskType=='TO',self.agvTaskList))
        q=int(q*len(agvTos))
        D=random.sample(agvTos,q) # select q TOs randomly from task list
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
            prevToCurr=agv.getDischargeRate()*self.scheduler.layout.getDistanceFromNode(currentNode,task.source)
            currToCurr=agv.getDischargeRate()*self.scheduler.layout.getDistanceFromNode(task.source,task.dest)
            currentNode=task.dest
            currentCharge=currentCharge-prevToCurr-currToCurr if task.taskType!='C' else 100
            isFeasible=False if currentCharge<30 else True            
