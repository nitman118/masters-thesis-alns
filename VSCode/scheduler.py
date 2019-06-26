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
from time import *
#OBJECT IMPORTS
from agv import AGV
from layout import Layout
from station import Station
from task import Task,TransportOrder
from stationDandR import ALNSStationDestroyAndRepairMethods
from customerDandR import ALNSCustomerDestroyAndRepairMethods

class Scheduler():
    
    def __init__(self, layoutFile, agvFile, requestFile, stationFile):
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
        self.stationALNSMethods = ALNSStationDestroyAndRepairMethods() # instantiating ALNS repair method class
        self.setNearestChargeLocation() # keep a dictionary of from to relationship b/w source station to charging stations
        self.customerALNSMethods = ALNSCustomerDestroyAndRepairMethods(self)
        #DEBUG - REMOVE LATER
        self.tlLength = list()
        self.temp=list()
        
        
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
            agv.LOWER_THRESHOLD+=safety_charge # to ensure feasibility of LP

            
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
                                            dest= row['target'],ept= row['ept'], ldt=row['ldt'], cap=row['capability'])
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
        srcStation = list(filter(lambda x:x.nodeId==task.source,self.stations))[0] # src station
        dstStation = list(filter(lambda x:x.nodeId==task.dest,self.stations))[0] # destination station
        if self.getCharge(agv)<agv.LOWER_THRESHOLD:
            #unloaded travel Cost
            dists = [self.layout.getDistanceFromNode(self.getCurrentReleaseNode(agv),station.getNode())\
                     for station in self.chargingStations]
            optIndex = dists.index(min(dists))
            nearestChargeNode = self.chargingStations[optIndex].getNode()
            distScore = (self.layout.getDistanceFromNode(nearestChargeNode,task.source)+ \
                         self.layout.getDistanceFromNode(task.source,task.dest))*agv.travelCost
            
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
            
            tardScore = tardiness**2
            
            return tardScore + distScore
        
        else:
            #dist score
            distScore = (self.layout.getDistanceFromNode(self.getCurrentReleaseNode(agv),task.source)+self.layout.getDistanceFromNode(task.source,task.dest))*agv.travelCost
            #tardiness score
            tardScore=0
            dist = self.layout.getDistanceFromNode(self.getCurrentReleaseNode(agv),task.source)+\
                self.layout.getDistanceFromNode(task.source,task.dest)
            
            absRelTime = max(self.getCurrentReleaseTime(agv),task.ept)
            
            time = absRelTime+dist/agv.speed+srcStation.mhTime+dstStation.mhTime
            
            tardiness = max(0,time-task.ldt)
            
            tardScore = tardiness**2
            
            return tardScore + distScore
        
    
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
        self.taskSequence.get(agvId).append(taskDict)
#         print(self.taskList)
    
    def setNearestChargeLocation(self):
        '''
        creates a dictionary of nearest charge locations from nodes in a layout
        '''
        nonChargeStations = [s.getNode() for s in self.stations if s.getType()!='C']
        for s in nonChargeStations:
            dists = [self.layout.getDistanceFromNode(s, station.getNode()) for station in self.chargingStations]
            optIndex = dists.index(min(dists))
            nearestChargeNode = self.chargingStations[optIndex].getNode()
            self.nearestChargeFromStation[s] = nearestChargeNode
            
            
    def getNearestChargeLocation(self, stationNode):
        '''
        returns nodeId of nearest charge location
        '''
        return self.nearestChargeFromStation.get(stationNode)
        
    def createTaskSequence(self):
        '''
        function that converts scheduler's taskList into a scheduler's taskSequence (postprocessing - add ut tasks)
        and checks for multiple charge tasks and assigns source and destination of charge tasks
        ''' 
        #check tasklist and remove duplicate charge tasks and add charge task source and destination
        for agv,tl in self.taskList.items():
            taskListLen=len(tl)-1
            toRemove=[]
            tasklist= self.taskList.get(agv)
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
            self.tlLength.append(len(list(filter(lambda x:x.taskType=='TO',self.taskList.get(0))))+len(list(filter(lambda x:x.taskType=='TO',self.taskList.get(1)))))
            if self.tlLength[-1]>100:
                self.temp = copy.deepcopy(self.taskList)
#             print(f'task list after processing:{tasklist}')
        
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
            if len(self.taskList.get(agv.agvId))>0:
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
    
    
    def solveLP(self,printOutput=False):
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
        
        #keep track of tasks (UT, TO , C) in agvs
        for a,agv in enumerate(self.agvs):
            REQ.append(len(self.taskSequence.get(a)))
            sp.append(agv.getSpeed())
            cr.append(agv.getChargeRate())
            dcr.append(agv.getDischargeRate())
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
                MHT[a].append(self.taskSequence.get(a)[r]['mhTime'])
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
             for v in range(numAGVs)  for r in range(REQ[v])}

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
        obj_lateness = quicksum(Z[r,v]*Z[r,v] for v in range(numAGVs) for r in range(REQ[v]))

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
                i=SRC[v][r]
                j=DEST[v][r]
                m.addConstr(D[r,v]==S[r+1,v], name=f"Dest_{r}_{v}=Src{r+1}_{v}")
                
        #Constraint 3 - A job cannot be picked up before EPT
        for v in range(numAGVs):
            for r in range(REQ[v]):
                m.addConstr(EPT[v][r]<=S[r,v],name=f"S_{r}_{v}>=EPT_{r}_{v}")
                
        #Constraint 4 - To represent lateness Z = D - LDT 
        for v in range(numAGVs):
            for r in range(REQ[v]):
                if LDT[v][r]>0:
                    m.addConstr(Z[r,v]>=D[r,v]+MHT[v][r]- LDT[v][r], name = f"Z_{r}_{v}>=D_{r}_{v}-LDT_{r}_{v}") # where Z[r] represents slack, when Z{r} is -ve, it will be 0 due to lower bound
        
        #Constraint 5 - Battery initialization at starting nodes to 100%
        for v in range(numAGVs):
            m.addConstr(B[0,v]==100, name=f"B_0_init_{v}") #remove hardcoded 100 here

        #Constraint 6 - Battery discharging and charging
        for v in range(numAGVs):
            for r in range(REQ[v]-1):
                i=SRC[v][r]
                j=SRC[v][r+1]
                dist = self.layout.getDistanceFromNode(i,j)
                if i in C and r >=2:
                    b = m.addVar()
                    m.addConstr(b==B[r,v]+((S[r+1,v]-S[r,v]-(1/sp[v])*dist)*cr[v])-((1/sp[v])*dist*dcr[v]), name=f"b_{r}_{v}")
                    m.addGenConstrMin(B[r+1,v],[b,100], name=f"B_{r}_{v}") # charge cannot be greater than 100%
                else:
                    m.addConstr((B[r+1,v]==B[r,v]-((1/sp[v])*dist*dcr[v])), name=f"B_{r}_{v}")
                    
        #Constraint 7 - Check for conflicts
#         for v1 in range(numAGVs-1):
#             for v2 in range(v1+1,numAGVs): 
#                 for r1 in range(REQ[v1]-1):
#                     for r2 in range(REQ[v2]-1):
#                         if SRC[v1][r1]==SRC[v2][r2]:
#                             if LDT[v1][r1]<LDT[v2][r2]:
#                                 dist = self.layout.getDistanceFromNode(source=SRC[v1][r1],dest=SRC[v1][r1+1])
#                                 m.addConstr(S[r2,v2]>=S[r1+1,v1]-((1/sp[v1])*dist)+1,name=f"ConflictS_v{v1}{r1}_v{v2}{r2}")
#                             elif LDT[v1][r1]>LDT[v2][r2]:
#                                 dist = self.layout.getDistanceFromNode(source=SRC[v2][r2],dest=SRC[v2][r2+1])
#                                 m.addConstr(S[r1,v1]>=S[r2+1,v2]-((1/sp[v2])*dist)+1,name=f"Conflict_v{v1}{r1}_v{v2}{r2}")
#                         elif DEST[v1][r1]==DEST[v2][r2] and REQID[v1][r1]<998 and REQID[v2][r2]<998:
#                             if LDT[v1][r1]<LDT[v2][r2]:
#                                 dist = self.layout.getDistanceFromNode(source=DEST[v1][r1],dest=DEST[v1][r1+1])
#                                 m.addConstr(D[r2,v2]>=D[r1+1,v1]-((1/sp[v1])*dist)+1,name=f"ConflictD_v{v1}{r1}_v{v2}{r2}")
#                             elif LDT[v1][r1]>LDT[v2][r2]:
#                                 dist = self.layout.getDistanceFromNode(source=DEST[v2][r2],dest=DEST[v2][r2+1])
#                                 m.addConstr(D[r1,v1]>=D[r2+1,v2]-((1/sp[v2])*dist)+1,name=f"Conflict_v{v1}{r1}_v{v2}{r2}")
    
    
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
        self.stationALNSMethods.destroyRandomCharge(agv=agv, scheduler=self)           

    def destroyAllCharge(self,agv):
        '''
        destroy all the charging tasks of a random agv
        '''
        self.stationALNSMethods.destroyAllCharge(agv=agv, scheduler=self)
        
    def destroyWorstCharge(self,agv):
        '''
        Destroys the worst charge task from a set of charge tasks
        '''
        self.stationALNSMethods.destroyWorstCharge(agv=agv, scheduler=self)
                      
    
    def repairInsertNCCharge(self,agv):
        '''
        repair sequence by introducing Non-Critical charge after tasks in a random agv
        this function should assign tasks with a charge threshold of 60%, however, it is not a critical 
        '''
        self.stationALNSMethods.repairInsertNCCharge(agv=agv,taskList=self.taskList.get(agv.agvId),scheduler=self)
        
        pass
    
    def repairInsertNCandCCharge(self, agv):
        '''
        repair sequence by introducing ONE NC charge followed by greedily placing C charge
        '''
        self.stationALNSMethods.repairInsertNCandCCharge(agv=agv,taskList=self.taskList.get(agv.agvId),\
                                               scheduler=self)
        pass
    
    def repairInsertCCharge(self, agv):
        '''
        repair sequence by introducing C charge greedily
        '''
        self.stationALNSMethods.repairInsertCCharge(agv=agv,taskList=self.taskList.get(agv.agvId),\
                                               scheduler=self, threshold=agv.LOWER_THRESHOLD)
        
    
    def repairInsertAllCharge(self,agv):
        '''
        repair sequence by introducing charge task after every ask in an agv
        '''
        self.stationALNSMethods.repairInsertAllCharge(agv=agv,taskList=self.taskList.get(agv.agvId), scheduler=self)
    
    def destroyShawDistance(self):
        self.customerALNSMethods.destroyShawDistance()
        
    def destroyShawTime(self):
        self.customerALNSMethods.destroyShawTimeWindow()
        
    def destroyShawCapability(self):
        self.customerALNSMethods.destroyShawCapability()
        
    def destroyShaw(self):
        self.customerALNSMethods.shawRemoval()
        
    def destroyRandomTasks(self):
        self.customerALNSMethods.destroyRandomTasks()
        
    def repairInsertRandomTasks(self):
        self.customerALNSMethods.repairInsertRandomTasks()

    def alns(self, solTime):
        '''
        Adaptive Large Neighborhood Search
        TODO: See the effect of adding a Deep Neural Net to initialize initial weights of destroy and repair methods
        '''
        psi1 = 0.9 # if new solution is new global best
        psi2 = 0.6 # if new solution is better than current solution but not the best
        psi3 = 0.2 # if new solution is accepted

        lambdaP = 0.5 # lambda parameter to cont
        
        bestSol = copy.deepcopy(self.taskSchedule)
        bestTL = copy.deepcopy(self.taskList)
        bestTaskList = copy.deepcopy(self.taskList)
        bestScore = self.getScoreALNS(bestSol)
        
        currentScore = bestScore
        print(f'Best Score at the start:{bestScore}')
        '''
        Initialize set of destroy and repair methods, initialize weights of respective methods
        '''
        customerDestroy = [self.destroyShaw,self.destroyShawDistance,self.destroyShawTime,self.destroyShawCapability]
        customerRepair = [self.repairInsertRandomTasks]
        
        customerRhoD=[1/len(customerDestroy) for i in range(len(customerDestroy))]
        customerRhoR=[1/len(customerRepair) for i in range(len(customerRepair))]
    
        destroy = [self.destroyRandomCharge,self.destroyWorstCharge,self.destroyAllCharge] # destroy methods
        repair = [self.repairInsertCCharge,self.repairInsertNCCharge,self.repairInsertAllCharge,\
                  self.repairInsertNCandCCharge] # repair methods
        rhoD=[1/len(destroy) for i in range(len(destroy))] # weight vector of destroy methods
        destroyN=[0 for i in range(len(destroy))]
        destroyB=[0 for i in range(len(destroy))]
        destroyI=[0 for i in range(len(destroy))]
        
        rhoR=[1/len(repair) for i in range(len(repair))] # weight vector of repair methods
        repairN=[0 for i in range(len(repair))]
        repairB=[0 for i in range(len(repair))]
        repairI=[0 for i in range(len(repair))]
        
        numIter=0
        scores=[]
        bestScores=[]
        startTime = time()
        infeasibleCount=0
        while time()-startTime<=solTime: # solTime is passed as the time available to provide a solution
            isStationMethod=True
            if False:
                agv = np.random.choice(self.agvs) 
                selD = np.random.choice(destroy,p=rhoD)
                indexD = destroy.index(selD) # index of destroy method
                destroyN[indexD]+=1
                selR = np.random.choice(repair,p=rhoR) 
                indexR = repair.index(selR) # index of repair method
                repairN[indexR]+=1

                selD(agv) # destroy agv sequence
                selR(agv) # repair agv sequence
                isStationMethod=True
            else:
                selD = np.random.choice(customerDestroy,p=customerRhoD)
                selR = np.random.choice(customerRepair, p=customerRhoR)
                
                selD()
                customerIndexD=customerDestroy.index(selD)
                selR()
                customerIndexR=customerRepair.index(selR)
                isStationMethod=False
                
            self.createTaskSequence()
            isSolvable = self.solveLP()
            
            if isSolvable:
                xtl=self.taskList
                xt=self.taskSchedule
                newScore = self.getScoreALNS(xt)
                scores.append((numIter,newScore))
                
                if newScore<bestScore:
                    bestTL=copy.deepcopy(xtl)#copy tasklist
                    bestSol=copy.deepcopy(xt)#copy taskSchedule
                    bestScore=newScore
                    bestScores.append((numIter,bestScore))
                    
                    if isStationMethod:
                        rhoD[indexD]= lambdaP*rhoD[indexD]+(1-lambdaP)*psi1
                        rhoR[indexR]= lambdaP*rhoR[indexR]+(1-lambdaP)*psi1
                        destroyB[indexD]+=1
                        repairB[indexR]+=1
                        self.updateWeightVectors(rhoD, rhoR)
                    else:
                        customerRhoD[customerIndexD]=lambdaP*customerRhoD[customerIndexD]+(1-lambdaP)*psi1
                        customerRhoR[customerIndexR]=lambdaP*customerRhoR[customerIndexR]+(1-lambdaP)*psi1
                        self.updateWeightVectors(customerRhoD,customerRhoR)
                        
                    
                    
                elif newScore < currentScore:
                    currentScore=newScore
#                     scores.append((numIter,currentScore))
                    if isStationMethod:
                        rhoD[indexD]= lambdaP*rhoD[indexD]+(1-lambdaP)*psi2
                        rhoR[indexR]= lambdaP*rhoR[indexR]+(1-lambdaP)*psi2
                        destroyI[indexD]+=1
                        repairI[indexR]+=1
                        self.updateWeightVectors(rhoD, rhoR)
                    else:
                        customerRhoD[customerIndexD]=lambdaP*customerRhoD[customerIndexD]+(1-lambdaP)*psi2
                        customerRhoR[customerIndexR]=lambdaP*customerRhoR[customerIndexR]+(1-lambdaP)*psi2
                        self.updateWeightVectors(customerRhoD,customerRhoR)
                    
                else:
                    if isStationMethod:
                        
                        rhoD[indexD]= lambdaP*rhoD[indexD]+(1-lambdaP)*psi3
                        rhoR[indexR]= lambdaP*rhoR[indexR]+(1-lambdaP)*psi3
                        self.updateWeightVectors(rhoD, rhoR)
                    else:
                        customerRhoD[customerIndexD]=lambdaP*customerRhoD[customerIndexD]+(1-lambdaP)*psi3
                        customerRhoR[customerIndexR]=lambdaP*customerRhoR[customerIndexR]+(1-lambdaP)*psi3
                        self.updateWeightVectors(customerRhoD,customerRhoR)
                        
                    
            else:
                infeasibleCount+=1
                
            numIter+=1
                
        
        print('rhoD:',rhoD)
        print('rhoR:',rhoR)
        print(f'customerRhoD:{customerRhoD}')
        print(f'customerRhoR:{customerRhoR}')
        print('Infeasibility Count:', infeasibleCount)
        self.taskSchedule=bestSol
        self.taskList=bestTL
        return bestSol,bestScore,numIter, destroyN,repairN,scores,bestScores,destroyB,repairB,destroyI,repairI
    
    def updateWeightVectors(self,rhoD,rhoR):
        rhoD[:]=[val/sum(rhoD) for val in rhoD]
        rhoR[:]=[val/sum(rhoR) for val in rhoR]
    
#KPI and other functionalities

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
        unloadedTravel=0
        unloadedTravelTime=0
        unloadedTravelTimeCost=0
        for a,agv in enumerate(self.agvs):
            for r,req in enumerate(taskSchedule.get(a)):
                if req.get('taskType')=='C' or req.get('taskType')=='UT':
                    unloadedTravel += self.layout.getDistanceFromNode(req['source'],req['dest'])
                elif req.get('taskType')=='TO':
                    tardiness += max(0, req['D']-req['ldt'])
            unloadedTravelTime+=unloadedTravel/agv.speed
            unloadedTravelTimeCost=unloadedTravelTime*agv.travelCost
        
        return tardiness,unloadedTravel,unloadedTravelTime,unloadedTravelTimeCost
    
    def getScoreALNS(self,taskSchedule, alpha=0.5):
        '''
        returns score of a schedule, used in alns algorithm
        '''
        tardiness,_,_,unloadedTravelCost = self.getScheduleKPI(taskSchedule)
        return (alpha*tardiness + (1-alpha)*(unloadedTravelCost))
        
    
    def writeScheduleToFile(self):
        '''
        Writes schedule in a text/excel file and visualizes the schedule
        '''
        
        pass


if __name__== '__main__':
    start=time()
    scheduler = Scheduler(layoutFile='outputDM.csv', agvFile='agvs.xlsx', requestFile='transportOrders2.xlsx', \
                        stationFile='stations.xlsx')
    scheduler.createGreedySequence()
    scheduler.createTaskSequence()
    scheduler.solveLP(printOutput=False)

    end = time()
    print(f'time:{end-start}')
    scheduler.getScheduleKPI()
    bestSol,bestScore,numIter, destroyN,repairN,scores,bestScores,destroyB,repairB,destroyI,repairI = scheduler.alns(5)
    