'''
Class representing a task
Charge Task
Non-Critical Charge task
MH Task
Unloaded Travel Task
'''
class Task():
    
    def __init__(self, taskId, taskType, source, dest):
        self.taskId=taskId #this is the original id of the request from the file
        self.taskIndex=888 #this is changed based on order of tasks, use this to locate charge tasks
        self.taskType = taskType
        self.source = source
        self.dest = dest
        
    def __repr__(self):
        return f"{self.taskId}-{self.taskType}-{self.source}-{self.dest}"
    
    def getTaskId(self):
        '''
        returns the unique id of the task
        '''
        return self.taskId

'''
This class represents an inherited object(Transport Order) of a TASK'''
class TransportOrder(Task):
    def __init__(self,taskId, taskType, source, dest, ept, ldt, cap):
        super().__init__(taskId, taskType, source, dest) #call 'Task' class constructor
        self.ept = ept
        self.ldt = ldt
        self.cap = cap
        
    def __repr__(self):
        return f"{self.taskId}-{self.taskType}-{self.source}-{self.dest}-{self.ept}-{self.ldt}-{self.cap}"
    
    def getEPT(self):
        return self.ept
    
    def getLDT(self):
        return self.ldt