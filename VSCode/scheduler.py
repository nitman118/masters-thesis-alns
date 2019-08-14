import numpy as np
import pandas as pd
from pandas import read_excel
from sklearn.preprocessing import normalize # for normalizing distance matrix
import matplotlib.pyplot as plt
from csv import reader
import functools # for reduce
from gurobipy import *
import random 
import copy # to deep copy from taskSequence to taskSchedule
from time import perf_counter
#OBJECT IMPORTS
from agv import AGV
from layout import Layout
from station import Station
from task import Task,TransportOrder
from stationDandR import ALNSStationDestroyAndRepairMethods
from customerDandR import ALNSCustomerDestroyAndRepairMethods

class Scheduler():
    def __init__(self, layoutFile, agvFile, requestFile, stationFile, hyperparams=None):
        self.hyperparams=hyperparams
        self.agvs =[] # field responsible to keep track of agvs
        self.agvsInfo=dict() # field to keep track of agv charge, location etc...
        self.stations=[] # field responsible to keep track of stations
        self.layout = None 
        self.transportOrders = list()
        self.chargingStations = list()
        self.nearestChargeFromStation=dict()
        self.taskList = dict() # this contains a processed list of TOs and Chargetasks, produced during greedy sequencing
        self.taskSequence = dict() # it is made AFTER all tos and charge tasks are assigned i.e. after greedy sequencing, taskSequence is sent for scheduling
        self.taskSchedule = dict() # order of execution -> create taskList->create taskSequence->schedule via LP->taskSchedule
        self.normalizedT=dict() #to store normalized EPT values
        self.createLayout(layoutFile=layoutFile)
        self.createAGVs(agvFile=agvFile)
        self.createStations(stationFile=stationFile)
        self.createRequests(requestFile=requestFile)
        self.setAGVInfo() # create a JSON like agv information object
        self.stationALNSMethods = ALNSStationDestroyAndRepairMethods(self) # instantiating ALNS repair method class
        self.setNearestChargeLocation() # keep a dictionary of from to relationship b/w source station to charging stations
        self.customerALNSMethods = ALNSCustomerDestroyAndRepairMethods(self)
        
        

    def setAGVInfo(self):
        '''
        Keeps a record of agv location charge etc.
        '''
        for a,agv in enumerate(self.agvs):
            self.agvsInfo[a]={}
            self.agvsInfo[a]['charge']=agv.charge
            self.agvsInfo[a]['startNode']=agv.startNode
            self.agvsInfo[a]['release']=(0, agv.startNode) #(time, location)
            self.agvsInfo[a]['state']=agv.state
            
    def getCurrentReleaseNode(self, agv):
        '''
        returns agv's current release node
        '''
        return self.agvsInfo.get(agv.agvId)['release'][1]
    
    def setCurrentReleaseNode(self,agv, releaseNode):
        '''
        sets the release node of an agv
        '''
        self.agvsInfo.get(agv.agvId)['release']=(self.agvsInfo.get(agv.agvId)['release'][0],releaseNode)
        
    def getCurrentReleaseTime(self,agv):
        
        '''
        returns the time at which AGV leaves/can leave the release node
        '''
        return self.agvsInfo.get(agv.agvId)['release'][0]
    
    def setCurrentReleaseTime(self,agv,releaseTime):
        '''
        sets agv release time
        '''    
        self.agvsInfo.get(agv.agvId)['release']=(releaseTime,self.agvsInfo.get(agv.agvId)['release'][1])

    
    def setState(self,agv,state):
        '''
        Sets the state of agv to Normal ='N' or Charging = 'C'
        '''
        self.agvsInfo.get(agv.agvId)['state'] = state
        
    def getState(self,agv):
        '''
        Returns state of agv, 'N'=Normal, 'C'=Charging
        '''
        return self.agvsInfo.get(agv.agvId)['state']
        
    def setCharge(self, agv,charge):
        '''
        Sets the agv charge
        '''
        self.agvsInfo.get(agv.agvId)['charge'] = min(100,charge)
        
    def getCharge(self,agv):
        '''
        Returns AGV Charge in %age
        '''
        return self.agvsInfo.get(agv.agvId)['charge']
    
    def getStartNode(self,agv):
        '''
        Returns agv's initial/start node'''
        if type(agv)==int:
            agv = self.getAgvById(agv)
        return self.agvsInfo.get(agv.agvId)['startNode']
        
        
    def createLayout(self, layoutFile):
        '''
        Creates digital layout inside scheduler
        '''
        self.layout = Layout(fileName=layoutFile)
        
    def createAGVs(self, agvFile):
        '''
        Creates agvs
        '''
        df =read_excel(agvFile)
        #TODO - modify lower threshold of agv based on 2*max distance in the layout and discharge rate of agvs
        for index,row in df.iterrows():
            agv = AGV(agvId=row['agvidf'], startNode=row['startNode'],caps= row['capability'], speed=row['speed'], \
                      charge=row['charge'],dischargeRate= row['dischargeRate'], chargeRate = row['chargingRate'],\
                      travelCost = row['travelCost'], low = row['lowerThreshold'], up = row['upperThreshold'], taskList=None)
            self.agvs.append(agv)
        for agv in self.agvs:
            safety_charge = (self.layout.getMaxDistance()/agv.speed)*agv.dischargeRate
            agv.LOWER_THRESHOLD+=safety_charge # to ensure feasibility of LP, may not be required though

            
    def createStations(self, stationFile):
        '''
        Creates stations
        '''
        df = read_excel(stationFile)
        for index, row in df.iterrows():
            station = Station(nodeId=row['id'], name=row['pointidf'], stType=row['type'], mhTime=row['mhtime'])
            self.stations.append(station)
            if station.getType() == 'C':
                self.chargingStations.append(station)
        
    
    def createRequests(self, requestFile):
        '''
        Creates TOs that need to be scheduled
        '''
        df =read_excel(requestFile)
        epts=[]
        ldts=[]
        for index,row in df.iterrows():
            transportOrder = TransportOrder(taskId=row['Id'], taskType='TO', source=row['source'],\
                                            dest= row['target'],ept= row['ept'], ldt=row['ldt'], cap=row['capability'], requestCost = row['requestCost'])
            self.transportOrders.append(transportOrder)
            epts.append(row['ept'])
            ldts.append(row['ldt'])
            self.normalizedT[row['Id']]=(row['ept'],row['ldt'])
            
        minEptLi, maxEptLi = min(epts),max(epts)
        minLdtLi, maxLdtLi = min(ldts),max(ldts)
        #create normalized ept and ldt values to be used in shaw relatedness function
        for r,val in self.normalizedT.items():
            self.normalizedT[r]=((val[0]-minEptLi)/(maxEptLi-minEptLi), (val[1]-minLdtLi)/(maxLdtLi-minLdtLi)) 
               
            
    def getNormalizedEPT(self,task):
        '''
        returns normalized ept
        '''
        return self.normalizedT.get(task.taskId)[0]
    
    def getNormalizedLDT(self,task):
        '''
        returns normalized ept
        '''
        return self.normalizedT.get(task.taskId)[1]
    
    def getAgvById(self,agvId):
        '''
        returns agv object by id
        '''
        return next((agv for agv in self.agvs if agv.agvId==agvId),None) #returns agv or None if not found
    
    def getMhtById(self,nodeId):
        '''
        returns mh time for node
        '''
        return next(st.getMHT() for st in self.stations if st.nodeId==nodeId)
        
    
            
#'''
#HEREUNDER LIES THE GREEDY SEQUENCING HEURISTIC
#'''

    def checkCap(self,to, agv):
        '''
        This function checks if the agv has required capabilities to do the task
        '''
        toCap = to.cap.split(',')
        
        b=[str(c) in agv.caps for c in toCap]
        return functools.reduce(lambda x,y: x and y,b) # reduce does pairwise comparison of 2 objects in a list
        

    def createGreedySequence(self):
        '''
        This function creates a greedy sequence of tasks to be performed, calls createTaskSequence to make a sequence list 
        with unloaded travel etc.
        '''
        #initialize the taskList dict object of scheduler
        self.taskList = dict()
        for a,agv in enumerate(self.agvs):
            self.taskList[agv.agvId]=[]
        
        self.transportOrders.sort(key = lambda x: x.ldt) # sort based on delivery time
        #assign tasks to AGVs and keep checking for expected charge after finishing the task
        for to in self.transportOrders:
            agv_count = [agv for agv in self.agvs if self.checkCap(to,agv)]
            # assert that there is at least one agv with desired capability
            assert (len(agv_count)>0), ("There should always be at least one agv with desired capability")
            if len(agv_count)==1:
                #add the task to AGV
                self.addGreedyTask(agv_count[0],to)
            elif len(agv_count)>1:
                scores=[]
                for agv in agv_count:
                    score = self.getScore(agv,to)
                    scores.append(score)
                optAGV = scores.index(min(scores))
                self.addGreedyTask(agv_count[optAGV],to)
        
                
#         #AFTER HAVING CREATED A TASK LIST, add normalized ept and ldt values for shaw removal
#         eptLi=[]
#         ldtLi=[]
#         self.normalizedT=dict.fromkeys(self.taskList,[]) # create a dictonay with keys as from tasklist and value as empty
#         #list
#         for key,val in self.taskList.items(): # for agv and tasklist 
            
#             eptLi=[e.getEPT() for e in val if e.taskType=='TO'] #create a list of all epts
#             ldtLi=[e.getLDT() for e in val if e.taskType=='TO'] #create a list of all ldts
            
#             minEptLi, maxEptLi = min(eptLi),max(eptLi)
#             minLdtLi, maxLdtLi = min(ldtLi),max(ldtLi)
            
#             eptLi=[(e-minEptLi)/(maxEptLi-minEptLi) for e in eptLi] #normalize
#             ldtLi=[(e-minLdtLi)/(maxLdtLi-minLdtLi) for e in ldtLi] #normalize
            
#             normalT=self.normalizedT.get(key)
#             c=0
#             for v in val:
#                 if v.taskType=='TO':
#                     normalT.append((v.getTaskId(),eptLi[c],ldtLi[c]))
#                     c+=1
                
#         print(self.normalizedT)
        
            
                
    
    
    def getScore(self,agv,task):
        alpha = self.hyperparams.get('alpha',0.5)
        srcStation = list(filter(lambda x:x.nodeId==task.source,self.stations))[0] # src station
        dstStation = list(filter(lambda x:x.nodeId==task.dest,self.stations))[0] # destination station
        if self.getCharge(agv)<agv.LOWER_THRESHOLD:
            #unloaded travel Cost
            dists = [self.layout.getDistanceFromNode(self.getCurrentReleaseNode(agv),station.getNode())\
                     for station in self.chargingStations]
            optIndex = dists.index(min(dists))
            nearestChargeNode = self.chargingStations[optIndex].getNode()
            distScore = agv.travelCost*((self.layout.getDistanceFromNode(nearestChargeNode,task.source)+ \
                         self.layout.getDistanceFromNode(task.source,task.dest))/agv.getSpeed())
            
            #tardiness score
            tardScore=0
            travelDist = self.layout.getDistanceFromNode(self.getCurrentReleaseNode(agv),nearestChargeNode)
            drivingTime = travelDist/agv.speed # time spent in driving
            minChargeTime = (agv.LOWER_THRESHOLD - self.getCharge(agv))/agv.chargeRate
            absRelTime = self.getCurrentReleaseTime(agv)+minChargeTime+drivingTime
            dist = self.layout.getDistanceFromNode(nearestChargeNode,task.source)+\
                self.layout.getDistanceFromNode(task.source,task.dest)
            
            if absRelTime>task.ept:
                time = absRelTime+dist/agv.speed +srcStation.mhTime + dstStation.mhTime
            else:
                time = task.ept+dist/agv.speed +srcStation.mhTime + dstStation.mhTime
            
            tardiness = max(0,time-task.ldt)
            
            tardScore = task.requestCost*tardiness
            
            return alpha*tardScore + (1-alpha)*distScore
        
        else:
            #dist score
            distScore = agv.travelCost*((self.layout.getDistanceFromNode(self.getCurrentReleaseNode(agv),task.source)+self.layout.getDistanceFromNode(task.source,task.dest))/agv.getSpeed())
            #tardiness score
            tardScore=0
            dist = self.layout.getDistanceFromNode(self.getCurrentReleaseNode(agv),task.source)+\
                self.layout.getDistanceFromNode(task.source,task.dest)
            
            absRelTime = max(self.getCurrentReleaseTime(agv),task.ept)
            
            time = absRelTime+dist/agv.speed+srcStation.mhTime+dstStation.mhTime
            
            tardiness = max(0,time-task.ldt)
            
            tardScore = task.requestCost*tardiness
            
            return alpha*tardScore + (1-alpha)*distScore
        
    
    def addChargeTask(self,agv):
        '''
        Adds a charge task to the agv's tasklist
        '''
        dists = [self.layout.getDistanceFromNode(self.getCurrentReleaseNode(agv),station.getNode()) for station in self.chargingStations]
        optIndex = dists.index(min(dists))
        nearestChargeNode = self.chargingStations[optIndex].getNode()
        chargeTask = Task(999,'C','X','X')
        
#         agv.taskList.append(chargeTask) #REMOVE LATER
        self.taskList.get(agv.agvId).append(chargeTask)
        self.setState(agv,'C') #charging
        
        travelDist = self.layout.getDistanceFromNode(self.getCurrentReleaseNode(agv),nearestChargeNode)
        drivingTime = travelDist/agv.speed # time spent in driving
        self.setCharge(agv, self.getCharge(agv) - (agv.dischargeRate * drivingTime) )
        
        self.setCurrentReleaseNode(agv,nearestChargeNode)
        self.setCurrentReleaseTime(agv,self.getCurrentReleaseTime(agv)+drivingTime)

        
    def addTaskToTaskList(self,agv,task):
        #add a TO to agv's task list 
        travelDist = self.layout.getDistanceFromNode(self.getCurrentReleaseNode(agv),task.source) + \
                        self.layout.getDistanceFromNode(task.source, task.dest)
            
        srcStation = list(filter(lambda x:x.nodeId==task.source,self.stations))[0] # src station
        dstStation = list(filter(lambda x:x.nodeId==task.dest,self.stations))[0] # destination station
        drivingTime = travelDist/agv.speed # time spent in driving
        
        if self.getState(agv)=='N':
            travelTime = drivingTime + srcStation.mhTime + dstStation.mhTime # total time including material handling
#             agv.taskList.append(task) #REMOVE LATER
            self.taskList.get(agv.agvId).append(task)
            self.setCurrentReleaseNode(agv,task.dest)
            self.setCurrentReleaseTime(agv,max(self.getCurrentReleaseTime(agv),task.ept)+travelTime)
            self.setCharge(agv, self.getCharge(agv) - (agv.dischargeRate * drivingTime) )
            self.setState(agv,'N')
        
        elif self.getState(agv)=='C':
            minChargeTime = (agv.LOWER_THRESHOLD - self.getCharge(agv))/agv.chargeRate # time to reach LOWER_THRESHOLD charge level
            minChargeAbsTime = self.getCurrentReleaseTime(agv) + max(0,minChargeTime)  # the absolutime time (in sec) at which AGV becomes 30% charged
            
            if task.ept >= minChargeAbsTime :
                # add the task but update charge based on delta
                chargeTime = (task.ept - self.getCurrentReleaseTime(agv))
                self.setCharge(agv,self.getCharge(agv) + (chargeTime * agv.chargeRate)) # charge after charging till task's ept
                self.setCurrentReleaseTime(agv,self.getCurrentReleaseTime(agv)+chargeTime)
                travelTime = drivingTime + srcStation.mhTime + dstStation.mhTime # total time including material handling
#                 agv.taskList.append(task) # REMOVE LATER
                self.taskList.get(agv.agvId).append(task)
                self.setCurrentReleaseNode(agv,task.dest)
                self.setCurrentReleaseTime(agv,max(self.getCurrentReleaseTime(agv),task.ept)+travelTime)
                self.setCharge(agv, self.getCharge(agv) - (agv.dischargeRate * drivingTime) )
                self.setState(agv,'N')
                
            elif task.ept < minChargeAbsTime:
                #
                self.setCurrentReleaseTime(agv,minChargeAbsTime)
                travelTime = drivingTime + srcStation.mhTime + dstStation.mhTime # total time including material handling
#                 agv.taskList.append(task) # REMOVE LATER
                self.taskList.get(agv.agvId).append(task)
                self.setCurrentReleaseNode(agv,task.dest)
                self.setCurrentReleaseTime(agv,max(self.getCurrentReleaseTime(agv),task.ept)+travelTime)
                self.setCharge(agv, self.getCharge(agv) - (agv.dischargeRate * drivingTime) )
                self.setState(agv,'N')
                
         
        
    def addGreedyTask(self,agv,task):

        if self.getCharge(agv)>=agv.LOWER_THRESHOLD and self.getState(agv) =='N':
            #add the task
            self.addTaskToTaskList(agv,task)
            
        elif self.getCharge(agv)<agv.LOWER_THRESHOLD and self.getState(agv)=='N':
            # add a charge task
            self.addChargeTask(agv)
            
        if self.getCharge(agv)<agv.LOWER_THRESHOLD and self.getState(agv)=='C':
            #update the charge at time of new request, if sufficient charge is present, change state, add the task
            self.addTaskToTaskList(agv,task)
            
    
    def writeToTaskSequence(self, agvId, task):
        '''
        write a task of an agv to scheduler's task sequence 
        '''
        taskDict={}
        taskDict['taskIndex']=task.taskIndex
        taskDict['taskId']=task.taskId
        taskDict['taskType']=task.taskType
        taskDict['source']=task.source
        taskDict['dest']=task.dest       
        try:
            taskDict['mhTime'] = list(filter(lambda x:x.nodeId==task.source,self.stations))[0].getMHT()
        except IndexError:
            print(f'id of task{task}:{id(task)}')
            
        if task.taskType=='TO':
            taskDict['ept']=task.ept
            taskDict['ldt']=task.ldt
            taskDict['requestCost']=task.requestCost
        self.taskSequence.get(agvId).append(taskDict)
#         print(self.taskList)
    
    def setNearestChargeLocation(self):
        '''
        creates a dictionary of nearest charge locations from nodes in a layout
        '''
        stations = [s.getNode() for s in self.stations]
        for s in stations:
            dists = [self.layout.getDistanceFromNode(s, station.getNode()) for station in self.chargingStations]
            optIndex = dists.index(min(dists))
            nearestChargeNode = self.chargingStations[optIndex].getNode()
            self.nearestChargeFromStation[s] = nearestChargeNode
            
            
    def getNearestChargeLocation(self, stationNode):
        '''
        returns nodeId of nearest charge location
        '''
        return self.nearestChargeFromStation.get(stationNode)
    
    def fixChargeOrganiseTaskList(self):
        '''
        Checks for multiple charge tasks and assigns source and destination of charge tasks
        '''
         #check tasklist and remove duplicate charge tasks and add charge task source and destination
        
        
        for agv,tl in self.taskList.items():
            tasklist= self.taskList.get(agv)
            agv_startNode = self.getStartNode(self.getAgvById(agv))
            if agv_startNode in [x.getNode() for x in self.chargingStations]:
                ct = Task(999,'C',agv_startNode,agv_startNode)
                tasklist.insert(0,ct)
            taskListLen=len(tasklist)-1
            toRemove=[]
            for t in range(taskListLen):
                if tasklist[t].taskType=='C' and t!=0:
                    if tasklist[t+1].taskType=='C':
                        toRemove.append(tasklist[t+1])
#                     print(f'id of task inside createTask{tasklist[t]}:{id(tasklist[t])}')
                    tasklist[t].source = tasklist[t-1].dest
#                     print(tasklist[t].source)
                    tasklist[t].dest = self.getNearestChargeLocation(tasklist[t].source)
                elif tasklist[t].taskType=='C' and t==0:
                    if tasklist[t+1].taskType=='C':
                        toRemove.append(tasklist[t+1])
                    tasklist[t].source = self.getStartNode(self.agvs[agv])
                    tasklist[t].dest = self.getStartNode(self.agvs[agv])
                elif t==taskListLen-1 and tasklist[t+1].taskType=='C':
                    tasklist[t+1].source = tasklist[t].dest
                    tasklist[t+1].dest = self.getNearestChargeLocation(tasklist[t+1].source)
                    
            tasklist[:] =[t for t in tasklist if t not in toRemove]
            
        
    def createTaskSequence(self):
        '''
        function that converts scheduler's taskList into a scheduler's taskSequence (postprocessing - add ut tasks)
        and checks for multiple charge tasks and assigns source and destination of charge tasks
        '''
        for agv in self.agvs:
            self.taskList.get(agv.agvId).append(Task(999,"C",'X','X'))
       
        self.fixChargeOrganiseTaskList()
        #Create task Sequence
        self.taskSequence={}
        #create keys in taskList for AGVs
        for agv in self.agvs:
            for task in self.taskList.get(agv.agvId):
                task.taskIndex = self.taskList.get(agv.agvId).index(task) #assigning index
        #create agv keys
        for agv in self.agvs:
            taskListLength = len(self.taskList.get(agv.agvId)) # length of tasks in task list
            self.taskSequence[agv.agvId]=[] # assign an empty list to each agv in task sequence
            if len(self.taskList.get(agv.agvId))>0 and self.taskList.get(agv.agvId)[0].taskType !='C':
                ut = Task(998,'UT',self.getStartNode(agv), self.taskList.get(agv.agvId)[0].source) 
                #create a UT(Unloaded Travel)
                #task, id and index dont matter'''
                self.writeToTaskSequence(agv.agvId,ut)
            for t,task in enumerate(self.taskList.get(agv.agvId)): # for rest of the tasks
                self.writeToTaskSequence(agv.agvId,task)
                if t< taskListLength-1 and self.taskList.get(agv.agvId)[t+1].taskType=='TO': # if not end of list and 
                    # next task is some kind of TO and not charge
                    ut = Task(998,'UT',task.dest, self.taskList.get(agv.agvId)[t+1].source)  
                    self.writeToTaskSequence(agvId=agv.agvId,task=ut)
                    
        
                
        
                                
#'''
#HEREUNDER LIES THE LP FORMULATION / could be another class
#'''   
    
    
    def solveLP(self,printOutput=False, requireConflictFree=False):
        numAGVs=len(self.agvs)
        REQ =[] #List consisting of number of requests
        SRC=[] #List containing SRC of requests
        DEST=[]
        EPT=[]
        LDT=[]
        MHT=[]
        cr=[]
        dcr=[]
        sp=[]
        REQID=[]
        B0=[]
        #keep track of tasks (UT, TO , C) in agvs
        for a,agv in enumerate(self.agvs):
            REQ.append(len(self.taskSequence.get(a)))
            sp.append(agv.getSpeed())
            cr.append(agv.getChargeRate())
            dcr.append(agv.getDischargeRate())
            B0.append(agv.getInitialCharge())
            SRC.append([])
            DEST.append([])
            EPT.append([])
            LDT.append([])
            MHT.append([])
            REQID.append([])
            
            for r in range(REQ[a]):
                REQID[a].append(self.taskSequence.get(a)[r]['taskId'])
                SRC[a].append(self.taskSequence.get(a)[r]['source'])
                DEST[a].append(self.taskSequence.get(a)[r]['dest'])
                MHT[a].append(self.taskSequence.get(a)[r].get('mhTime') or 0)
                EPT[a].append(self.taskSequence.get(a)[r].get('ept') or 0)
                LDT[a].append(self.taskSequence.get(a)[r].get('ldt') or 0)
                
        
        C = [x.getNode() for x in self.chargingStations] # list of charging station nodeIDs
        
        
        #GUROBI
        m = Model('Scheduling')
        m.setParam('OutputFlag',printOutput)
        #minimization model
        m.modelSense = GRB.MINIMIZE
        #decision variables
        #variable for lateness of request r
        Z = {(r,v):m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, name = f"Z_{r}_{v}")
             for v in range(numAGVs)  for r in range(REQ[v])} # 0 becuase we do not consider earliness

        #variable to represent the time instance at which AGV reaches 'source' of a request r
        S = {(r,v):m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, name = f"S_{r}_{v}")
             for v in range(numAGVs)  for r in range(REQ[v])}

        #variable to represent the time instance at which AGV reaches 'destination' of a request r
        D = {(r,v):m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, name = f"D_{r}_{v}") 
             for v in range(numAGVs)  for r in range(REQ[v])}

        #variable to represent the battery status at the beginning of request r
        B = {(r,v):m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub=100.0, name = f"B_{r}_{v}")
             for v in range(numAGVs)  for r in range(max(1,REQ[v]))} # we use max to avoid error becuae of zero tasklist length

        #battery can reach 0% only at charging stations
        for v in range(numAGVs):
            for r in range(REQ[v]):
                if REQID[v][r]<998:
                    B[r,v].lb=30.0 # gurobi syntax to showcase indices
        
        m.update()
        
#         for v in range(numAGVs):
#             print(REQ[v])
#             for r in range(REQ[v]):
#                 print(r,v)
        #objective function
        #Objective 1 - To minimize lateness
        obj_lateness = quicksum(Z[r,v] for v in range(numAGVs) for r in range(REQ[v]))

        # #Objective 2 - To maximize charging duration and minimize parking duration at source and destination nodes
#         obj_charging = grb.quicksum(0 if SRC[v][r] in C else 1*(S[r+1,v]-S[r,v]) for r in range(REQ[v]-1) for v in range(numAGVs))

        #Objective 3 - To minimize Unloaded Travel Time
#         obj_unloaded = grb.quicksum(0 if REQID[v][r]<998 else 1*(D[r,v] - S[r,v]) for r in range(REQ[v]) for v in range(numAGVs)) 
        # becoz that's where unloaded travel occurs
        
        #SET OBJECTIVE
        m.setObjective(obj_lateness)
        #adaption for online +obj_charging
        #Constraints

        #Constraint 1 - trip time should be less than Destination time instance
        for v in range(numAGVs):
            for r in range(REQ[v]):
                i=SRC[v][r]
                j=DEST[v][r]
                dist = self.layout.getDistanceFromNode(i,j)
                m.addConstr(S[r,v]+(MHT[v][r]+(1/sp[v])*dist)<=D[r,v],name=f"Headway_{r}_{v}")
                
        #Constraint 2 - Destination of request should equal source of next request
        for v in range(numAGVs):
            for r in range(REQ[v]-1):
                #i=SRC[v][r]
                #j=DEST[v][r]
                m.addConstr(D[r,v]==S[r+1,v], name=f"Dest_{r}_{v}=Src{r+1}_{v}")
                
        #Constraint 3 - A job cannot be picked up before EPT
        for v in range(numAGVs):
            for r in range(REQ[v]):
                m.addConstr(EPT[v][r]<=S[r,v],name=f"S_{r}_{v}>=EPT_{r}_{v}")
                
        #Constraint 4 - To represent lateness Z = D - LDT 
        for v in range(numAGVs):
            for r in range(REQ[v]):
                if LDT[v][r]>0:
                    m.addConstr(Z[r,v]>=D[r,v]- LDT[v][r], name = f"Z_{r}_{v}>=D_{r}_{v}-LDT_{r}_{v}") # where Z[r] represents slack, when Z{r} is -ve, it will be 0 due to lower bound
        #+MHT[v][r]
        #Constraint 5 - Battery initialization at starting nodes to 100%
        for v1 in range(numAGVs):
            m.addConstr(B[0,v1]==B0[v1], name=f"B_0_init_{v1}") 

        #Constraint 6 - Battery discharging and charging
        for v in range(numAGVs):
            for r in range(REQ[v]-1):
                i=SRC[v][r]
                j=SRC[v][r+1]
                dist = self.layout.getDistanceFromNode(i,j)
                if i in C:
                    b = m.addVar()
                    m.addConstr(b==B[r,v]+((S[r+1,v]-S[r,v]-(1/sp[v])*dist)*cr[v])-((1/sp[v])*dist*dcr[v]), name=f"b_{r}_{v}")
                    m.addGenConstrMin(B[r+1,v],[b,100], name=f"B_{r}_{v}") # charge cannot be greater than 100%
                else:
                    m.addConstr((B[r+1,v]==B[r,v]-((1/sp[v])*dist*dcr[v])), name=f"B_{r}_{v}")
                    
        #Constraint 7 - Check for conflicts
        if requireConflictFree:
            for v1 in range(numAGVs-1):
                for v2 in range(v1+1,numAGVs): 
                    for r1 in range(REQ[v1]-1):
                        for r2 in range(REQ[v2]-1):
                            if SRC[v1][r1]==SRC[v2][r2]:
                                y=m.addVar(vtype=GRB.BINARY)
                                dist = self.layout.getDistanceFromNode(source=SRC[v1][r1],dest=DEST[v1][r1])
                                m.addConstr(S[r2,v2]+10000*y>=D[r1,v1]-(dist/sp[v1])+0.1, name = f"ConflictS1_{v1}_{r1}_{v2}_{r2}")
                                dist = self.layout.getDistanceFromNode(source=SRC[v2][r2],dest=DEST[v2][r2])
                                m.addConstr(S[r1,v1]>=D[r2,v2]-(dist/sp[v2])+0.1-(1-y)*10000, name = f"ConflictS2_{v1}_{r1}_{v2}_{r2}")
                            elif DEST[v1][r1]==DEST[v2][r2]:
                                y=m.addVar(vtype=GRB.BINARY)
                                dist = self.layout.getDistanceFromNode(source=SRC[v1][r1+1],dest=DEST[v1][r1+1])
                                m.addConstr(D[r2,v2]+10000*y>=D[r1+1,v1]-(dist/sp[v1])+0.1, name = f"ConflictD1_{v1}_{r1}_{v2}_{r2}")
                                dist = self.layout.getDistanceFromNode(source=SRC[v2][r2+1],dest=DEST[v2][r2+1])
                                m.addConstr(D[r1,v1]>=D[r2+1,v2]-(dist/sp[v2])+0.1-(1-y)*10000, name = f"ConflictD2_{v1}_{r1}_{v2}_{r2}")
                        
                            
                            
#                            if LDT[v1][r1]<LDT[v2][r2]:
#                                dist = self.layout.getDistanceFromNode(source=SRC[v1][r1],dest=SRC[v1][r1+1])
#                                m.addConstr(S[r2,v2]>=S[r1+1,v1]-((1/sp[v1])*dist)+1,name=f"ConflictS_v{v1}{r1}_v{v2}{r2}")
#                            elif LDT[v1][r1]>LDT[v2][r2]:
#                                dist = self.layout.getDistanceFromNode(source=SRC[v2][r2],dest=SRC[v2][r2+1])
#                                m.addConstr(S[r1,v1]>=S[r2+1,v2]-((1/sp[v2])*dist)+1,name=f"Conflict_v{v1}{r1}_v{v2}{r2}")
#                        elif DEST[v1][r1]==DEST[v2][r2] and REQID[v1][r1]<998 and REQID[v2][r2]<998:
#                            if LDT[v1][r1]<LDT[v2][r2]:
#                                dist = self.layout.getDistanceFromNode(source=DEST[v1][r1],dest=DEST[v1][r1+1])
#                                m.addConstr(D[r2,v2]>=D[r1+1,v1]-((1/sp[v1])*dist)+1,name=f"ConflictD_v{v1}{r1}_v{v2}{r2}")
#                            elif LDT[v1][r1]>LDT[v2][r2]:
#                                dist = self.layout.getDistanceFromNode(source=DEST[v2][r2],dest=DEST[v2][r2+1])
#                                m.addConstr(D[r1,v1]>=D[r2+1,v2]-((1/sp[v2])*dist)+1,name=f"Conflict_v{v1}{r1}_v{v2}{r2}")
    
    
        #optimize model
        m.optimize()
        status = m.status
        if status == GRB.Status.UNBOUNDED:
            print('The model cannot be solved because it is unbounded')
            return False
        elif status == GRB.Status.OPTIMAL:
            self.createTaskSchedule(S=S, D=D, B=B)
            m.reset()
            return True
#             print('The optimal objective is %g' % m.objVal)

        elif status == GRB.Status.INF_OR_UNBD or status== GRB.Status.INFEASIBLE:
            print(f'Optimization was stopped with status {status}')
            
            print(self.taskList)
            print('-------------------------------------------------------')
            print(self.taskSequence)
            
            # do IIS
            print('The model is infeasible; computing IIS')
            m.computeIIS()
            if m.IISMinimal:
                print('IIS is minimal\n')
            else:
                print('IIS is not minimal\n')
            print('\nThe following constraint(s) cannot be satisfied:')
            for c in m.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)
            return False
            
    def createTaskSchedule(self,S,D,B):
        '''
        creates Task schedule
        '''
        self.taskSchedule=copy.deepcopy(self.taskSequence)
        
        for v,a in enumerate(self.agvs):
            for r,req in enumerate(self.taskSchedule.get(v)):
                req['S']=S[r,v].x # time at which agv v reaches the source of request r
                req['D']=D[r,v].x # time at which agv v reaches the dest of request r
                req['B']=B[r,v].x # battery level at start of request r of agv v
                
            
#'''
#HEREUNDER LIES THE ALNS IMPLEMENTATION
#'''

    def destroyRandomCharge(self,agv):
        '''
        randomly remove a charging task from sequence of tasks of a random agv
        '''
        start=perf_counter()
        self.stationALNSMethods.destroyRandomCharge(agv=agv)
        end = perf_counter()
        return end-start

    def destroyAllCharge(self,agv):
        '''
        destroy all the charging tasks of a random agv
        '''
        start=perf_counter()
        self.stationALNSMethods.destroyAllCharge(agv=agv)
        end=perf_counter()
        return end-start
        
    def destroyWorstCharge(self,agv):
        '''
        Destroys the worst charge task from a set of charge tasks
        '''
        start=perf_counter()
        self.stationALNSMethods.destroyWorstCharge(agv=agv)
        end=perf_counter()
        return end-start
                      
    
    def repairInsertNCCharge(self,agv):
        '''
        repair sequence by introducing Non-Critical charge after tasks in a random agv
        this function should assign tasks with a charge threshold of 60%, however, it is not a critical 
        '''
        start=perf_counter()
        self.stationALNSMethods.repairInsertNCCharge(agv=agv,taskList=self.taskList.get(agv.agvId))
        end=perf_counter()
        return end-start
    
    def repairInsertNCandCCharge(self, agv):
        '''
        repair sequence by introducing ONE NC charge followed by greedily placing C charge
        '''
        start=perf_counter()
        self.stationALNSMethods.repairInsertNCandCCharge(agv=agv,taskList=self.taskList.get(agv.agvId))
        end=perf_counter()
        return end-start
    
    def repairInsertCCharge(self, agv):
        '''
        repair sequence by introducing C charge greedily
        '''
        start=perf_counter()
        self.stationALNSMethods.repairInsertCCharge(agv=agv,taskList=self.taskList.get(agv.agvId), \
                                                    threshold=agv.LOWER_THRESHOLD)
        end=perf_counter()
        return end-start
        
    
    def repairInsertAllCharge(self,agv):
        '''
        repair sequence by introducing charge task after every ask in an agv
        '''
        start=perf_counter()
        self.stationALNSMethods.repairInsertAllCharge(agv=agv,taskList=self.taskList.get(agv.agvId))
        end=perf_counter()
        return end-start
    
    def destroyShawDistance(self):
        start=perf_counter()
        self.customerALNSMethods.destroyShawDistance()
        end=perf_counter()
        return end-start
        
    def destroyShawTime(self):
        start=perf_counter()
        self.customerALNSMethods.destroyShawTimeWindow()
        end=perf_counter()
        return end-start
        
    def destroyShawCapability(self):
        start=perf_counter()
        self.customerALNSMethods.destroyShawCapability()
        end=perf_counter()
        return end-start
        
    def destroyShaw(self):
        start=perf_counter()
        self.customerALNSMethods.shawRemoval()
        end=perf_counter()
        return end-start
        
    def destroyRandomTasks(self):
        start=perf_counter()
        self.customerALNSMethods.destroyRandomTasks()
        end=perf_counter()
        return end-start
    
    def destroyWorstTardinessCustomers(self):
        start=perf_counter()
        self.customerALNSMethods.destroyWorstTardinessCustomers()
        end=perf_counter()
        return end-start
        
        
    def repairInsertRandomTasks(self):
        start=perf_counter()
        self.customerALNSMethods.repairInsertRandomTasks()
        end=perf_counter()
        return end-start
        
    def repairInsertGreedyTask(self):
        start=perf_counter()
        self.customerALNSMethods.repairInsertGreedyTasks()
        end=perf_counter()
        return end-start
        
    def repairKRegret(self):
        start=perf_counter()
        self.customerALNSMethods.repairKRegret()
        end=perf_counter()
        return end-start
        
    def repairGreedyEDDInsert(self):
        start=perf_counter()
        self.customerALNSMethods.repairGreedyEDDInsert()
        end=perf_counter()
        return end-start
        
    def alns(self, solTime):
        '''
        Adaptive Large Neighborhood Search
        TODO: See the effect of adding a Deep Neural Net to initialize initial weights of destroy and repair methods
        '''
        print('method0 executed')
        psi1 = self.hyperparams.get('psi1',0.9) # if new solution is new global best
        psi2 = self.hyperparams.get('psi2',0.6) # if new solution is better than current solution but not the best
        psi3 = self.hyperparams.get('psi3',0.3) # if new solution is accepted
        lambdaP = self.hyperparams.get('lambdaP',0.5) # lambda parameter to cont
        bestTL = copy.deepcopy(self.taskList)
        bestTaskList = copy.deepcopy(self.taskList)
        bestScore = self.getScoreALNS()
        currentScore = bestScore
        #print(f'Best Score at the start:{bestScore}')
        '''
        Initialize set of destroy and repair methods, initialize weights of respective methods
        '''
        customerDestroy = [self.destroyShaw,self.destroyShawDistance,self.destroyShawTime,self.destroyShawCapability,\
                          self.destroyRandomTasks, self.destroyWorstTardinessCustomers]
        customerRepair = [self.repairKRegret,self.repairInsertGreedyTask,self.repairGreedyEDDInsert]
        customerRhoD=[1/len(customerDestroy) for i in range(len(customerDestroy))]
        customerDestroyN = [0 for i in range(len(customerDestroy))]
        customerDestroyB= [0 for i in range(len(customerDestroy))]
        customerRepairN = [0 for i in range(len(customerRepair))]
        customerRepairB = [0 for i in range(len(customerRepair))]
        customerRhoR=[1/len(customerRepair) for i in range(len(customerRepair))]
        stationDestroy = [self.destroyRandomCharge,self.destroyAllCharge] # destroy methods
        stationRepair = [self.repairInsertCCharge,self.repairInsertNCCharge,self.repairInsertAllCharge,\
                  self.repairInsertNCandCCharge] # repair methods
        stationRhoD=[1/len(stationDestroy) for i in range(len(stationDestroy))] # weight vector of destroy methods
        stationDestroyN=[0 for i in range(len(stationDestroy))] # number of times station destroy is chosen
        stationDestroyB=[0 for i in range(len(stationDestroy))] # number of times station destroy leads to best solution
        stationRhoR=[1/len(stationRepair) for i in range(len(stationRepair))] # weight vector of repair methods
        stationRepairN=[0 for i in range(len(stationRepair))] # number of times station reair is chosen
        stationRepairB=[0 for i in range(len(stationRepair))] # number of times it leads to best solution
        numIter=0
        scores=[]
        scores.append((numIter,bestScore)) # first observation
        bestScores=[]
        startTime = perf_counter()
        bestScores.append((numIter,bestScore,perf_counter()-startTime))
        infeasibleCount=0
        a1=int(self.hyperparams.get('a',2))
        while perf_counter()-startTime<=solTime: # solTime is passed as the time available to provide a solution
            isStationMethod=True
            self.setAGVInfo()# reset the information about AGVs
            if numIter% a1==0:
                agv = np.random.choice(self.agvs) 
                selD = np.random.choice(stationDestroy,p=stationRhoD)
                stationIndexD = stationDestroy.index(selD) # index of destroy method
                stationDestroyN[stationIndexD]+=1
                selR = np.random.choice(stationRepair,p=stationRhoR) 
                stationIndexR = stationRepair.index(selR) # index of repair method
                stationRepairN[stationIndexR]+=1
                selD(agv) # destroy agv sequence
                selR(agv) # repair agv sequence
                isStationMethod=True
            else:
                selD = np.random.choice(customerDestroy,p=customerRhoD)
                selR = np.random.choice(customerRepair, p=customerRhoR)
                selD()
                customerIndexD=customerDestroy.index(selD)
                customerDestroyN[customerIndexD]+=1
                selR()
                customerIndexR=customerRepair.index(selR)
                customerRepairN[customerIndexR]+=1
                isStationMethod=False
            if True:
                self.createTaskSequence()
                xtl=self.taskList
                self.solveLP()
                newScore = self.getScoreALNS()
                scores.append((numIter,newScore))
                
                if newScore<bestScore:
                    bestTL=copy.deepcopy(xtl)#copy tasklist
                    bestScore=newScore
                    bestScores.append((numIter,bestScore,perf_counter()-startTime))
                    if isStationMethod:
                        stationDestroyB[stationIndexD]+=1
                        stationRepairB[stationIndexR]+=1
                        self.updateWeightVectors(rhoD=stationRhoD,rhoR=stationRhoR,indexD=stationIndexD, indexR=stationIndexR, lambdaX=lambdaP, psiX=psi1)
                    else:
                        customerDestroyB[customerIndexD]+=1
                        customerRepairB[customerIndexR]+=1
                        self.updateWeightVectors(rhoD=customerRhoD,rhoR=customerRhoR,indexD=customerIndexD, indexR=customerIndexR, lambdaX=lambdaP, psiX=psi1)        
                elif newScore < currentScore:
                    currentScore=newScore
                    if isStationMethod:
                        self.updateWeightVectors(rhoD=stationRhoD,rhoR=stationRhoR,indexD=stationIndexD, indexR=stationIndexR, lambdaX=lambdaP, psiX=psi2)
                    else:
                        self.updateWeightVectors(rhoD=customerRhoD,rhoR=customerRhoR,indexD=customerIndexD, indexR=customerIndexR, lambdaX=lambdaP, psiX=psi2)
                else:
                    currentScore=newScore
                    if isStationMethod:
                        self.updateWeightVectors(rhoD=stationRhoD,rhoR=stationRhoR,indexD=stationIndexD, indexR=stationIndexR, lambdaX=lambdaP, psiX=psi3)
                    else:
                        self.updateWeightVectors(rhoD=customerRhoD,rhoR=customerRhoR,indexD=customerIndexD, indexR=customerIndexR, lambdaX=lambdaP, psiX=psi3)
                         
            else:
                infeasibleCount+=1   
            numIter+=1
        self.taskList=bestTL
        self.createTaskSequence()
        self.solveLP()
        
        solution={'numIter':numIter,
                  'stationDestroyN':stationDestroyN,
                  'stationRepairN':stationRepairN,
                  'scores':scores,
                  'bestScores':bestScores,
                  'stationDestroyB':stationDestroyB,
                  'stationRepairB':stationRepairB,
                  'customerDestroyN':customerDestroyN,
                  'customerRepairN':customerRepairN,
                  'customerDestroyB':customerDestroyB,
                  'customerRepairB':customerRepairB,
                  'methodN':None
                  }
                  
        
        return solution

    def alns1(self, solTime):
        '''
        Adaptive Large Neighborhood Search
        TODO: See the effect of adding a Deep Neural Net to initialize initial weights of destroy and repair methods
        '''
        print('method1 executed')
        psi1 = 0.9 # if new solution is new global best
        psi2 = 0.6 # if new solution is better than current solution but not the best
        psi3 = 0.3 # if new solution is accepted
        lambdaP = 0.5 # lambda parameter to cont
        bestTL = copy.deepcopy(self.taskList)
        bestTaskList = copy.deepcopy(self.taskList)
        bestScore = self.getGreedyTaskListScore()
        currentScore = bestScore
        #print(f'Best Score at the start:{bestScore}')
        '''
        Initialize set of destroy and repair methods, initialize weights of respective methods
        '''
        customerDestroy = [self.destroyShaw,self.destroyShawDistance,self.destroyShawTime,self.destroyShawCapability,\
                          self.destroyRandomTasks,self.destroyWorstTardinessCustomers]
        customerRepair = [self.repairKRegret,self.repairInsertGreedyTask,self.repairGreedyEDDInsert]
        customerRhoD=[1/len(customerDestroy) for i in range(len(customerDestroy))]
        customerDestroyN = [0 for i in range(len(customerDestroy))]
        customerDestroyB= [0 for i in range(len(customerDestroy))]
        customerRepairN = [0 for i in range(len(customerRepair))]
        customerRepairB = [0 for i in range(len(customerRepair))]
        customerRhoR=[1/len(customerRepair) for i in range(len(customerRepair))]
        stationDestroy = [self.destroyRandomCharge,self.destroyAllCharge] # destroy methods
        stationRepair = [self.repairInsertCCharge,self.repairInsertNCCharge,self.repairInsertAllCharge,\
                  self.repairInsertNCandCCharge] # repair methods
        stationRhoD=[1/len(stationDestroy) for i in range(len(stationDestroy))] # weight vector of destroy methods
        stationDestroyN=[0 for i in range(len(stationDestroy))] # number of times station destroy is chosen
        stationDestroyB=[0 for i in range(len(stationDestroy))] # number of times station destroy leads to best solution
        stationRhoR=[1/len(stationRepair) for i in range(len(stationRepair))] # weight vector of repair methods
        stationRepairN=[0 for i in range(len(stationRepair))] # number of times station reair is chosen
        stationRepairB=[0 for i in range(len(stationRepair))] # number of times it leads to best solution
        numIter=0
        scores=[]
        scores.append((numIter,bestScore)) # first observation
        bestScores=[]
        startTime = perf_counter()
        bestScores.append((numIter,bestScore,perf_counter()-startTime))
        infeasibleCount=0
        while perf_counter()-startTime<=solTime: # solTime is passed as the time available to provide a solution
            isStationMethod=True
            self.setAGVInfo()# reset the information about AGVs
            if numIter%3==0:
                agv = np.random.choice(self.agvs) 
                selD = np.random.choice(stationDestroy,p=stationRhoD)
                stationIndexD = stationDestroy.index(selD) # index of destroy method
                stationDestroyN[stationIndexD]+=1
                selR = np.random.choice(stationRepair,p=stationRhoR) 
                stationIndexR = stationRepair.index(selR) # index of repair method
                stationRepairN[stationIndexR]+=1
                runtimeD=selD(agv) # destroy agv sequence
                runtimeR=selR(agv) # repair agv sequence
                isStationMethod=True
            else:
                selD = np.random.choice(customerDestroy,p=customerRhoD)
                selR = np.random.choice(customerRepair, p=customerRhoR)
                runtimeD=selD()
                customerIndexD=customerDestroy.index(selD)
                customerDestroyN[customerIndexD]+=1
                runtimeR=selR()
                customerIndexR=customerRepair.index(selR)
                customerRepairN[customerIndexR]+=1
                isStationMethod=False
            if True:
                self.fixChargeOrganiseTaskList()#fix tasklist
                xtl=self.taskList
                newScore = self.getGreedyTaskListScore()
                scores.append((numIter,newScore))
                
                if newScore<bestScore:
                    bestTL=copy.deepcopy(xtl)#copy tasklist
                    bestScore=newScore
                    bestScores.append((numIter,bestScore,perf_counter()-startTime))
                    if isStationMethod:
                        stationDestroyB[stationIndexD]+=1
                        stationRepairB[stationIndexR]+=1
                        self.updateWeightVectors(rhoD=stationRhoD,rhoR=stationRhoR,indexD=stationIndexD, indexR=stationIndexR, lambdaX=lambdaP, psiX=psi1, runtimeD=runtimeD, runtimeR=runtimeR)
                    else:
                        customerDestroyB[customerIndexD]+=1
                        customerRepairB[customerIndexR]+=1
                        self.updateWeightVectors(rhoD=customerRhoD,rhoR=customerRhoR,indexD=customerIndexD, indexR=customerIndexR, lambdaX=lambdaP, psiX=psi1,runtimeD=runtimeD, runtimeR=runtimeR)        
                elif newScore < currentScore:
                    currentScore=newScore
                    if isStationMethod:
                        self.updateWeightVectors(rhoD=stationRhoD,rhoR=stationRhoR,indexD=stationIndexD, indexR=stationIndexR, lambdaX=lambdaP, psiX=psi2,runtimeD=runtimeD, runtimeR=runtimeR)
                    else:
                        self.updateWeightVectors(rhoD=customerRhoD,rhoR=customerRhoR,indexD=customerIndexD, indexR=customerIndexR, lambdaX=lambdaP, psiX=psi2,runtimeD=runtimeD, runtimeR=runtimeR)
                else:
                    currentScore=newScore
                    if isStationMethod:
                        self.updateWeightVectors(rhoD=stationRhoD,rhoR=stationRhoR,indexD=stationIndexD, indexR=stationIndexR, lambdaX=lambdaP, psiX=psi3,runtimeD=runtimeD, runtimeR=runtimeR)
                    else:
                        self.updateWeightVectors(rhoD=customerRhoD,rhoR=customerRhoR,indexD=customerIndexD, indexR=customerIndexR, lambdaX=lambdaP, psiX=psi3,runtimeD=runtimeD, runtimeR=runtimeR)
                         
            else:
                infeasibleCount+=1   
            numIter+=1
        
        self.taskList=bestTL
        self.createTaskSequence()
        self.solveLP()
        solution={'numIter':numIter,
                  'stationDestroyN':stationDestroyN,
                  'stationRepairN':stationRepairN,
                  'scores':scores,
                  'bestScores':bestScores,
                  'stationDestroyB':stationDestroyB,
                  'stationRepairB':stationRepairB,
                  'customerDestroyN':customerDestroyN,
                  'customerRepairN':customerRepairN,
                  'customerDestroyB':customerDestroyB,
                  'customerRepairB':customerRepairB,
                  'methodN':None
                  }
        return solution
    
    def alns2(self, solTime):
        '''
        Adaptive Large Neighborhood Search
        TODO: See the effect of adding a Deep Neural Net to initialize initial weights of destroy and repair methods
        '''
        print('method2 executed')
        psi1 = 0.9 # if new solution is new global best
        psi2 = 0.6 # if new solution is better than current solution but not the best
        psi3 = 0.3 # if new solution is accepted

        lambdaP = 0.5 # lambda parameter to cont
        
        
        bestTL = copy.deepcopy(self.taskList)
        bestScore = self.getScoreALNS()
        
        currentScore = bestScore
        #print(f'Best Score at the start:{bestScore}')
        '''
        Initialize set of destroy and repair methods, initialize weights of respective methods
        '''
        M = ['S','C'] # which method to choose
        rhoM=[0.5,0.5] # initial weights to choose family
        methodN=[0,0] #to keep track of how many times 'S' and 'C' were run
        
        customerDestroy = [self.destroyShaw,self.destroyShawDistance,self.destroyShawTime,self.destroyShawCapability,\
                          self.destroyRandomTasks,self.destroyWorstTardinessCustomers]
        customerRepair = [self.repairKRegret,self.repairInsertGreedyTask, self.repairGreedyEDDInsert]
        
        customerRhoD=[1/len(customerDestroy) for i in range(len(customerDestroy))]
        customerDestroyN = [0 for i in range(len(customerDestroy))]
        customerDestroyB= [0 for i in range(len(customerDestroy))]
        customerRepairN = [0 for i in range(len(customerRepair))]
        customerRepairB = [0 for i in range(len(customerRepair))]
        
        customerRhoR=[1/len(customerRepair) for i in range(len(customerRepair))]
    
        stationDestroy = [self.destroyRandomCharge,self.destroyAllCharge, self.destroyWorstCharge] # destroy methods
        stationRepair = [self.repairInsertCCharge,self.repairInsertNCCharge,self.repairInsertAllCharge,\
                  self.repairInsertNCandCCharge] # repair methods
        stationRhoD=[1/len(stationDestroy) for i in range(len(stationDestroy))] # weight vector of destroy methods
        stationDestroyN=[0 for i in range(len(stationDestroy))] # number of times station destroy is chosen
        stationDestroyB=[0 for i in range(len(stationDestroy))] # number of times station destroy leads to best solution
        
        stationRhoR=[1/len(stationRepair) for i in range(len(stationRepair))] # weight vector of repair methods
        stationRepairN=[0 for i in range(len(stationRepair))] # number of times station reair is chosen
        stationRepairB=[0 for i in range(len(stationRepair))] # number of times it leads to best solution
        
        numIter=0
        scores=[]
        scores.append((numIter,bestScore)) # first observation
        bestScores=[]
        startTime = perf_counter()
        bestScores.append((numIter,bestScore,perf_counter()-startTime))
        infeasibleCount=0
        while perf_counter()-startTime<=solTime: # solTime is passed as the time available to provide a solution
            chosenMethod = np.random.choice(M,p=rhoM)
            indexM = M.index(chosenMethod) # index of chosen method
            self.setAGVInfo()# reset the information about AGVs
            if chosenMethod=='S':
                agv = np.random.choice(self.agvs) 
                selD = np.random.choice(stationDestroy,p=stationRhoD)
                stationIndexD = stationDestroy.index(selD) # index of destroy method
                stationDestroyN[stationIndexD]+=1
                selR = np.random.choice(stationRepair,p=stationRhoR) 
                stationIndexR = stationRepair.index(selR) # index of repair method
                stationRepairN[stationIndexR]+=1
                selD(agv) # destroy agv sequence
                selR(agv) # repair agv sequence
                isStationMethod=True
            else:
                selD = np.random.choice(customerDestroy,p=customerRhoD)
                selR = np.random.choice(customerRepair, p=customerRhoR)
                selD()
                customerIndexD=customerDestroy.index(selD)
                customerDestroyN[customerIndexD]+=1
                selR()
                customerIndexR=customerRepair.index(selR)
                customerRepairN[customerIndexR]+=1
                isStationMethod=False
            if True: #because we do not do LP at each stage
                self.fixChargeOrganiseTaskList()#fix tasklist
                xtl=self.taskList
                self.createTaskSequence()
                self.solveLP()
                newScore = self.getScoreALNS()
                scores.append((numIter,newScore))
                if newScore<bestScore:
                    bestTL=copy.deepcopy(xtl)#copy tasklist
                    bestScore=newScore
                    bestScores.append((numIter,bestScore,perf_counter()-startTime))
                    self.updateWeightVectors(rhoD= rhoM, rhoR=rhoM, indexD=indexM, indexR=indexM, lambdaX=lambdaP, psiX=psi1)
                    methodN[indexM]+=1
                    if isStationMethod:
                        self.updateWeightVectors(rhoD=stationRhoD,rhoR=stationRhoR,indexD=stationIndexD, indexR=stationIndexR, lambdaX=lambdaP, psiX=psi1)
                        stationDestroyB[stationIndexD]+=1
                        stationRepairB[stationIndexR]+=1
                    else:
                        self.updateWeightVectors(rhoD=customerRhoD,rhoR=customerRhoR,indexD=customerIndexD, indexR=customerIndexR, lambdaX=lambdaP, psiX=psi1)
                        customerDestroyB[customerIndexD]+=1
                        customerRepairB[customerIndexR]+=1
                elif newScore < currentScore:
                    currentScore=newScore
                    self.updateWeightVectors(rhoD= rhoM, rhoR=rhoM, indexD=indexM, indexR=indexM, lambdaX=lambdaP, psiX=psi2)
                    methodN[indexM]+=1
#                     scores.append((numIter,currentScore))
                    if isStationMethod:
                        self.updateWeightVectors(rhoD=stationRhoD,rhoR=stationRhoR,indexD=stationIndexD, indexR=stationIndexR, lambdaX=lambdaP, psiX=psi2)
                    else:
                        self.updateWeightVectors(rhoD=customerRhoD,rhoR=customerRhoR,indexD=customerIndexD, indexR=customerIndexR, lambdaX=lambdaP, psiX=psi2)
                else:
                    currentScore=newScore
                    self.updateWeightVectors(rhoD= rhoM, rhoR=rhoM, indexD=indexM, indexR=indexM, lambdaX=lambdaP, psiX=psi3)
                    methodN[indexM]+=1
                    if isStationMethod:
                        self.updateWeightVectors(rhoD=stationRhoD,rhoR=stationRhoR,indexD=stationIndexD, indexR=stationIndexR, lambdaX=lambdaP, psiX=psi3)
                    else:
                        self.updateWeightVectors(rhoD=customerRhoD,rhoR=customerRhoR,indexD=customerIndexD, indexR=customerIndexR, lambdaX=lambdaP, psiX=psi3)
            else:
                infeasibleCount+=1
                
            numIter+=1
        
        self.taskList=bestTL
        self.createTaskSequence()
        self.solveLP()
        solution={'numIter':numIter,
                  'stationDestroyN':stationDestroyN,
                  'stationRepairN':stationRepairN,
                  'scores':scores,
                  'bestScores':bestScores,
                  'stationDestroyB':stationDestroyB,
                  'stationRepairB':stationRepairB,
                  'customerDestroyN':customerDestroyN,
                  'customerRepairN':customerRepairN,
                  'customerDestroyB':customerDestroyB,
                  'customerRepairB':customerRepairB,
                  'methodN':methodN
                  }
        return solution
    
    
    def alns3(self, solTime):
        '''
        Adaptive Large Neighborhood Search
        TODO: See the effect of adding a Deep Neural Net to initialize initial weights of destroy and repair methods
        '''
        print('method3 executed')
        psi1 = 0.9 # if new solution is new global best
        psi2 = 0.6 # if new solution is better than current solution but not the best
        psi3 = 0.3 # if new solution is accepted

        lambdaP = 0.5 # lambda parameter to cont
        
        
        bestTL = copy.deepcopy(self.taskList)
        bestScore = self.getGreedyTaskListScore()
        
        currentScore = bestScore
        #print(f'Best Score at the start:{bestScore}')
        '''
        Initialize set of destroy and repair methods, initialize weights of respective methods
        '''
        customerDestroy = [self.destroyShaw,self.destroyShawDistance,self.destroyShawTime,self.destroyShawCapability,\
                          self.destroyRandomTasks,self.destroyWorstTardinessCustomers]
        customerRepair = [self.repairKRegret,self.repairInsertGreedyTask,self.repairGreedyEDDInsert]
        
        customerRhoD=[1/len(customerDestroy) for i in range(len(customerDestroy))]
        customerDestroyN = [0 for i in range(len(customerDestroy))]
        customerDestroyB= [0 for i in range(len(customerDestroy))]
        customerRepairN = [0 for i in range(len(customerRepair))]
        customerRepairB = [0 for i in range(len(customerRepair))]
        
        customerRhoR=[1/len(customerRepair) for i in range(len(customerRepair))]
    
        stationDestroy = [self.destroyRandomCharge,self.destroyAllCharge] # destroy methods
        stationRepair = [self.repairInsertCCharge,self.repairInsertNCCharge,self.repairInsertAllCharge,\
                  self.repairInsertNCandCCharge] # repair methods
        stationRhoD=[1/len(stationDestroy) for i in range(len(stationDestroy))] # weight vector of destroy methods
        stationDestroyN=[0 for i in range(len(stationDestroy))] # number of times station destroy is chosen
        stationDestroyB=[0 for i in range(len(stationDestroy))] # number of times station destroy leads to best solution
        
        stationRhoR=[1/len(stationRepair) for i in range(len(stationRepair))] # weight vector of repair methods
        stationRepairN=[0 for i in range(len(stationRepair))] # number of times station reair is chosen
        stationRepairB=[0 for i in range(len(stationRepair))] # number of times it leads to best solution
        
        numIter=0
        scores=[]
        scores.append((numIter,bestScore)) # first observation
        bestScores=[]
        startTime = perf_counter()
        bestScores.append((numIter,bestScore,perf_counter()-startTime))
        infeasibleCount=0
        s_n=int(self.hyperparams.get('b',5))
        #DEBUG
        scorealns=[]
        scorenormal=[]
        while perf_counter()-startTime<=solTime: # solTime is passed as the time available to provide a solution
            self.setAGVInfo()# reset the information about AGVs
            selD = np.random.choice(customerDestroy,p=customerRhoD) #select destroy method
            selR = np.random.choice(customerRepair, p=customerRhoR) #select repair method
            selD() #destroy
            customerIndexD=customerDestroy.index(selD) #record index of destroy method
            customerDestroyN[customerIndexD]+=1 #inc. by 1
            selR() # repair
            customerIndexR=customerRepair.index(selR) #record repair method index
            customerRepairN[customerIndexR]+=1 # inc.
            self.fixChargeOrganiseTaskList()#fix tasklist  
            newScore = self.getGreedyTaskListScore()
            scores.append((numIter,newScore))
            
            if newScore<bestScore:
                bestTL=copy.deepcopy(self.taskList)#copy tasklist
                bestScore=newScore
                bestScores.append((numIter,bestScore,perf_counter()-startTime))
                self.updateWeightVectors(rhoD= customerRhoD, rhoR=customerRhoR, indexD=customerIndexD, indexR=customerIndexR, lambdaX=lambdaP, psiX=psi1)
                customerDestroyB[customerIndexD]+=1
                customerRepairB[customerIndexR]+=1
                bestFound = False
                s_i=0
                stationxtl = copy.deepcopy(self.taskList)
                while bestFound or s_i<=s_n:
                    self.setAGVInfo()# reset the information about AGVs
                    agv = np.random.choice(self.agvs) 
                    selD = np.random.choice(stationDestroy,p=stationRhoD)
                    stationIndexD = stationDestroy.index(selD) # index of destroy method
                    stationDestroyN[stationIndexD]+=1
                    selR = np.random.choice(stationRepair,p=stationRhoR) 
                    stationIndexR = stationRepair.index(selR) # index of repair method
                    stationRepairN[stationIndexR]+=1
                    selD(agv) # destroy agv sequence
                    selR(agv) # repair agv sequence
                    isStationMethod=True
                    self.fixChargeOrganiseTaskList()#fix tasklist
                    newScore = self.getGreedyTaskListScore()
                    scores.append((numIter,newScore))
                    if newScore < bestScore:
                        s_i=0
                        bestTL=copy.deepcopy(self.taskList)#copy tasklist
                        stationxtl = copy.deepcopy(self.taskList)
                        bestScore=newScore
                        stationDestroyB[stationIndexD]+=1
                        stationRepairB[stationIndexR]+=1
                        bestScores.append((numIter,bestScore,perf_counter()-startTime))
                        self.updateWeightVectors(rhoD= stationRhoD, rhoR=stationRhoR, indexD=stationIndexD, indexR=stationIndexR, lambdaX=lambdaP, psiX=psi1)
                        bestFound=True
                        
                    elif newScore<currentScore:
                        currentScore=newScore
                        bestFound=False
                        self.updateWeightVectors(rhoD= stationRhoD, rhoR=stationRhoR, indexD=stationIndexD, indexR=stationIndexR, lambdaX=lambdaP, psiX=psi2)
                        self.taskList = copy.deepcopy(stationxtl)
                        s_i+=1
                        
                    else:
                        bestFound=False
                        s_i+=1
                        self.taskList = copy.deepcopy(stationxtl)
                        self.updateWeightVectors(rhoD= stationRhoD, rhoR=stationRhoR, indexD=stationIndexD, indexR=stationIndexR, lambdaX=lambdaP, psiX=psi3)           
            elif newScore<currentScore:
                currentScore=newScore
                self.updateWeightVectors(rhoD=customerRhoD,rhoR=customerRhoR,indexD=customerIndexD, indexR=customerIndexR, lambdaX=lambdaP, psiX=psi2)
            else:
                currentScore=newScore
                self.updateWeightVectors(rhoD=customerRhoD,rhoR=customerRhoR,indexD=customerIndexD, indexR=customerIndexR, lambdaX=lambdaP, psiX=psi3)
                        
            
            #self.createTaskSequence()
            #self.solveLP()
            #scorealns.append(self.getScoreALNS())
            #scorenormal.append(self.getGreedyTaskListScore())
            numIter+=1                
        
        self.taskList=bestTL
        self.createTaskSequence()
        self.solveLP()
        solution={'numIter':numIter,
                  'stationDestroyN':stationDestroyN,
                  'stationRepairN':stationRepairN,
                  'scores':scores,
                  'bestScores':bestScores,
                  'stationDestroyB':stationDestroyB,
                  'stationRepairB':stationRepairB,
                  'customerDestroyN':customerDestroyN,
                  'customerRepairN':customerRepairN,
                  'customerDestroyB':customerDestroyB,
                  'customerRepairB':customerRepairB,
                  'methodN':None,
                  }
        return solution
    
    def alns4(self, solTime):
        '''
        Adaptive Large Neighborhood Search
        TODO: See the effect of adding a Deep Neural Net to initialize initial weights of destroy and repair methods
        '''
        print('method 4 executed')
        psi1 = self.hyperparams.get('psi1',0.9) # if new solution is new global best
        psi2 = self.hyperparams.get('psi2',0.6) # if new solution is better than current solution but not the best
        psi3 = self.hyperparams.get('psi3',0.3) # if new solution is accepted
        lambdaP = self.hyperparams.get('lambdaP',0.5) # lambda parameter to cont
        
        
        bestTL = copy.deepcopy(self.taskList)
        bestScore = self.getScoreALNS()
        
        currentScore = bestScore
        #print(f'Best Score at the start:{bestScore}')
        '''
        Initialize set of destroy and repair methods, initialize weights of respective methods
        '''
        customerDestroy = [self.destroyShaw,self.destroyShawDistance,self.destroyShawTime,self.destroyShawCapability,\
                          self.destroyRandomTasks,self.destroyWorstTardinessCustomers]
        customerRepair = [self.repairKRegret,self.repairInsertGreedyTask,self.repairGreedyEDDInsert]
        
        customerRhoD=[1/len(customerDestroy) for i in range(len(customerDestroy))]
        customerDestroyN = [0 for i in range(len(customerDestroy))]
        customerDestroyB= [0 for i in range(len(customerDestroy))]
        customerRepairN = [0 for i in range(len(customerRepair))]
        customerRepairB = [0 for i in range(len(customerRepair))]
        
        customerRhoR=[1/len(customerRepair) for i in range(len(customerRepair))]
    
        stationDestroy = [self.destroyRandomCharge,self.destroyAllCharge, self.destroyWorstCharge] # destroy methods
        stationRepair = [self.repairInsertCCharge,self.repairInsertNCCharge,self.repairInsertAllCharge,\
                  self.repairInsertNCandCCharge] # repair methods
        stationRhoD=[1/len(stationDestroy) for i in range(len(stationDestroy))] # weight vector of destroy methods
        stationDestroyN=[0 for i in range(len(stationDestroy))] # number of times station destroy is chosen
        stationDestroyB=[0 for i in range(len(stationDestroy))] # number of times station destroy leads to best solution
        
        stationRhoR=[1/len(stationRepair) for i in range(len(stationRepair))] # weight vector of repair methods
        stationRepairN=[0 for i in range(len(stationRepair))] # number of times station reair is chosen
        stationRepairB=[0 for i in range(len(stationRepair))] # number of times it leads to best solution
        
        numIter=0
        scores=[]
        scores.append((numIter,bestScore)) # first observation
        bestScores=[]
        startTime = perf_counter()
        bestScores.append((numIter,bestScore,perf_counter()-startTime))
        s_n=int(self.hyperparams.get('b',5))
        #DEBUG
       
        
        while perf_counter()-startTime<=solTime: # solTime is passed as the time available to provide a solution
            self.setAGVInfo()# reset the information about AGVs
            selD = np.random.choice(customerDestroy,p=customerRhoD) #select destroy method
            selR = np.random.choice(customerRepair, p=customerRhoR) #select repair method
            selD() #destroy
            customerIndexD=customerDestroy.index(selD) #record index of destroy method
            customerDestroyN[customerIndexD]+=1 #inc. by 1
            selR() # repair
            customerIndexR=customerRepair.index(selR) #record repair method index
            customerRepairN[customerIndexR]+=1 # inc.
            self.createTaskSequence()
            self.solveLP()
            newScore = self.getScoreALNS()
            scores.append((numIter,newScore))
            
            if newScore<bestScore:
                bestTL=copy.deepcopy(self.taskList)#copy tasklist
                bestScore=newScore
                bestScores.append((numIter,bestScore,perf_counter()-startTime))
                self.updateWeightVectors(rhoD= customerRhoD, rhoR=customerRhoR, indexD=customerIndexD, indexR=customerIndexR, lambdaX=lambdaP, psiX=psi1)
                customerDestroyB[customerIndexD]+=1
                customerRepairB[customerIndexR]+=1
                bestFound = False
                s_i=0
                stationxtl = copy.deepcopy(self.taskList)
                while bestFound or s_i<=s_n:
                    self.setAGVInfo()# reset the information about AGVs
                    agv = np.random.choice(self.agvs) 
                    selD = np.random.choice(stationDestroy,p=stationRhoD)
                    stationIndexD = stationDestroy.index(selD) # index of destroy method
                    stationDestroyN[stationIndexD]+=1
                    selR = np.random.choice(stationRepair,p=stationRhoR) 
                    stationIndexR = stationRepair.index(selR) # index of repair method
                    stationRepairN[stationIndexR]+=1
                    selD(agv) # destroy agv sequence
                    selR(agv) # repair agv sequence
                    isStationMethod=True
                    self.createTaskSequence()
                    self.solveLP()
                    newScore = self.getScoreALNS()
                    scores.append((numIter,newScore))
                    if newScore < bestScore:
                        s_i=0
                        bestTL=copy.deepcopy(self.taskList)#copy tasklist
                        stationxtl = copy.deepcopy(self.taskList)
                        bestScore=newScore
                        stationDestroyB[stationIndexD]+=1
                        stationRepairB[stationIndexR]+=1
                        bestScores.append((numIter,bestScore,perf_counter()-startTime))
                        self.updateWeightVectors(rhoD= stationRhoD, rhoR=stationRhoR, indexD=stationIndexD, indexR=stationIndexR, lambdaX=lambdaP, psiX=psi1)
                        bestFound=True
                        
                    elif newScore<currentScore:
                        currentScore=newScore
                        bestFound=False
                        self.updateWeightVectors(rhoD= stationRhoD, rhoR=stationRhoR, indexD=stationIndexD, indexR=stationIndexR, lambdaX=lambdaP, psiX=psi2)
                        self.taskList = copy.deepcopy(stationxtl)
                        s_i+=1
                        
                    else:
                        bestFound=False
                        s_i+=1
                        self.taskList = copy.deepcopy(stationxtl)
                        self.updateWeightVectors(rhoD = stationRhoD, rhoR=stationRhoR, indexD=stationIndexD, indexR=stationIndexR, lambdaX=lambdaP, psiX=psi3)           
            elif newScore<currentScore:
                currentScore=newScore
                self.updateWeightVectors(rhoD=customerRhoD,rhoR=customerRhoR,indexD=customerIndexD, indexR=customerIndexR, lambdaX=lambdaP, psiX=psi2)
            else:
                currentScore=newScore
                self.updateWeightVectors(rhoD=customerRhoD,rhoR=customerRhoR,indexD=customerIndexD, indexR=customerIndexR, lambdaX=lambdaP, psiX=psi3)
                        
            
            
            numIter+=1                
        
        self.taskList=bestTL
        self.createTaskSequence()
        self.solveLP(requireConflictFree=False)
        solution={'numIter':numIter,
                  'stationDestroyN':stationDestroyN,
                  'stationRepairN':stationRepairN,
                  'scores':scores,
                  'bestScores':bestScores,
                  'stationDestroyB':stationDestroyB,
                  'stationRepairB':stationRepairB,
                  'customerDestroyN':customerDestroyN,
                  'customerRepairN':customerRepairN,
                  'customerDestroyB':customerDestroyB,
                  'customerRepairB':customerRepairB,
                  'methodN':None,
                  }
        return solution
    
    def updateWeightVectors(self,rhoD,rhoR,indexD, indexR, lambdaX, psiX, runtimeD=0.01, runtimeR=0.01):
        runtimeD = 0.6 if runtimeD<0.001 else 1.4 if runtimeD>0.01 else 1
        runtimeR = 0.6 if runtimeD<0.001 else 1.4 if runtimeD>0.01 else 1
        rhoD[indexD]= lambdaX*rhoD[indexD]+(1-lambdaX)*(psiX/runtimeD)
        rhoR[indexR]= lambdaX*rhoR[indexR]+(1-lambdaX)*(psiX/runtimeR)
        rhoD[:]=[val/sum(rhoD) for val in rhoD]
        rhoR[:]=[val/sum(rhoR) for val in rhoR]
        
    
#KPI and other functionalities

    def getGreedyTaskListScore(self, alpha=0.5):
        alpha = self.hyperparams.get('alpha',0.5)
        tardinessTimeCost=0
        totalTTCost=0
        totalTT=0
        for key in self.taskList.keys():
            a = self.getAgvById(key)
            chargeRate = a.getChargeRate()
            dischargeRate = a.getDischargeRate()
            agvSpeed = a.getSpeed()
            currentNode = a.startNode
            currentCharge = a.charge
            runTime=0
            taskList = self.taskList.get(key)
            for n,task in enumerate(taskList):
                if task.taskType=='C':
                    nearestChargeNode = self.getNearestChargeLocation(currentNode)      
                    unloadedTT=self.layout.getDistanceFromNode(currentNode,nearestChargeNode)*(1/agvSpeed)
                    loadedTT=0
                    currentCharge-=dischargeRate*unloadedTT
                    reqdCharge=0
                    currentNodeC=nearestChargeNode
                    for t in range(n+1, len(taskList)):
                        if taskList[t].taskType!='C':
                            unloadedTTC=self.layout.getDistanceFromNode(currentNodeC,taskList[t].source)*(1/agvSpeed)
                            loadedTTC=self.layout.getDistanceFromNode(taskList[t].source,taskList[t].dest)*(1/agvSpeed)
                            currentNodeC=taskList[t].dest
                            reqdCharge+=dischargeRate*(unloadedTTC+loadedTTC)
                        else: break;
                    runTime+=unloadedTT+min(reqdCharge,100-currentCharge)/chargeRate
                    currentNode=nearestChargeNode
                    currentCharge+=reqdCharge
                    currentCharge=min(currentCharge,100)
                    tardinessTimeCost=0
                else:
                    unloadedTT=self.layout.getDistanceFromNode(currentNode,task.source)*(1/agvSpeed)
                    loadedTT=self.layout.getDistanceFromNode(task.source,task.dest)*(1/agvSpeed)
                    currentCharge-=dischargeRate*(unloadedTT+loadedTT)
                    if runTime<task.ept:
                        currentCharge+=(task.ept-unloadedTT-runTime)/chargeRate if n>0 and taskList[n-1].taskType=='C' else currentCharge
                        currentCharge=min(currentCharge,100)
                        runTime=task.ept-unloadedTT
                    runTime+=unloadedTT+self.getMhtById(task.source)+loadedTT+self.getMhtById(task.dest)
                    currentNode=task.dest
                    tardinessTimeCost+=max(0,(runTime-task.ldt)*task.requestCost)
                totalTT=unloadedTT+loadedTT
                totalTTCost+=(totalTT*a.travelCost)
        #print(f'tardiness:{tardinessTime},unloadedTravelTime:{totalUnloadedTime}, score:{alpha*tardinessTime+(1-alpha)*totalUnloadedTime}')
        return alpha*tardinessTimeCost+(1-alpha)*totalTTCost

    def getTaskfromTaskScheduleByIndex(self, index, agv):
        '''
        Return a task from taskSchedule based on provided index
        '''
        try:
            return list(filter(lambda x:x.get('taskIndex')==index,self.taskSchedule.get(agv.agvId)))[0]
        except: #perhaps IndexError
            return None
    

    def getScheduleKPI(self, taskSchedule=None):
        '''
        Returns KPI of a given taskSchedule
        '''
        taskSchedule = self.taskSchedule if taskSchedule is None else taskSchedule
        tardiness=0
        tardinessCost = 0
        otd=0
        numTRs=0
        ult=0
        lt=0
        totalTravel=0
        totalTravelTime=0
        totalTravelTimeCost=0
        for a,agv in enumerate(self.agvs):
            t=0 #
            for r,req in enumerate(taskSchedule.get(a)):
                t += self.layout.getDistanceFromNode(req['source'],req['dest'])
                if req.get('taskType')=='TO':
                    numTRs+=1 #track otd
                    tardiness += max(0, req['D']-req['ldt'])
                    tardinessCost = tardiness * req['requestCost']
                    lt+=self.layout.getDistanceFromNode(req['source'], req['dest'])
                    if req['D'] <= req['ldt']:
                        otd+=1
                else:
                    ult+=self.layout.getDistanceFromNode(req['source'],req['dest'])
            totalTravel+=t
            totalTravelTime+=(t/agv.speed)
            totalTravelTimeCost+=((t*agv.travelCost)/agv.speed)
        sla = (otd/numTRs)
        
        return tardiness,tardinessCost,totalTravel,lt,ult,totalTravelTime,totalTravelTimeCost, sla
    
    def getScoreALNS(self,taskSchedule=None, alpha=0.5):
        '''
        returns score of a schedule, used in alns algorithm
        '''
        alpha = self.hyperparams.get('alpha',0.5)
        _,tardinessCost,_,_,_,_,totalTravelCost,_ = self.getScheduleKPI(taskSchedule)
        return (alpha*tardinessCost + (1-alpha)*(totalTravelCost))
        
    
    def writeScheduleToFile(self):
        '''
        Writes schedule in a text/excel file and visualizes the schedule
        
        '''
        scheduleDf = pd.DataFrame(columns=['Id', 'source', 'target', 'ept', 'ldt', 'capability', 'requestCost', 'S', 'D', 'B', 'tardiness', 'assignedAGV'])
        
        for agv, schedule in self.taskSchedule.items():
            for task in schedule:
                row={}
                row['Id'] = task['taskId']
                row['source'] = task['source']
                row['target'] = task['dest']
                row['S'] = task['S']
                row['D'] = task['D']
                row['B'] = task['B']
                row['assignedAGV']=agv
                if task['taskType']=='TO':
                    row['ept'] = task['ept']
                    row['ldt'] = task['ldt']
                    row['requestCost'] = task['requestCost']
                    row['tardiness'] = int(max(0,task['D']-task['ldt']))
                    
                scheduleDf=scheduleDf.append(row, ignore_index=True)
            
                
        scheduleDf.to_excel('schedule.xlsx', sheet_name='schedule')
        
    def getScheduleCostAnalysis(self):
        scheduleDf = pd.DataFrame(columns=['Id', 'source', 'target', 'ept', 'ldt', 'capability', 'requestCost', 'S', 'D', 'B', 'tardiness', 'assignedAGV'])
        
        for agv, schedule in self.taskSchedule.items():
            for task in schedule:
                row={}
                row['Id'] = task['taskId']
                row['source'] = task['source']
                row['target'] = task['dest']
                row['S'] = task['S']
                row['D'] = task['D']
                row['B'] = task['B']
                row['assignedAGV']=agv
                if task['taskType']=='TO':
                    row['ept'] = task['ept']
                    row['ldt'] = task['ldt']
                    row['requestCost'] = task['requestCost']
                    row['tardiness'] = int(max(0,task['D']-task['ldt']))
                    
                scheduleDf=scheduleDf.append(row, ignore_index=True)
            
        typesOfVehicles = scheduleDf['assignedAGV'].unique()
        typesOfRequests=set()
        agvStat = {}
        for agv,schedule in self.taskSchedule.items():
            agvTotalTravel = 0
            for task in schedule:
                if task['taskType']=='TO':
                    typesOfRequests.add(task['requestCost'])
                agvTotalTravel+=self.layout.getDistanceFromNode(task['source'], task['dest'])
            agvStat[agv] = agvTotalTravel
                    
        reqStat = {}
        
        for request in typesOfRequests:
            reqStat[request]=scheduleDf[scheduleDf['requestCost']==request]['tardiness'].sum()
        
        return reqStat, agvStat
        
        
    
    
    def solve(self, runtime):
        method = self.hyperparams.get('alnsMethod',4)
        #print(method)
        self.createGreedySequence()
        self.createTaskSequence()
        self.solveLP(printOutput=False)
        solution =self.alns(runtime) if method==0 else self.alns1(runtime) if method==1 else \
        self.alns2(runtime) if method==2 else self.alns3(runtime) if method==3 else self.alns4(runtime)
        
        return solution
    
    
        