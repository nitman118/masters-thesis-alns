'''
Station Object
- Pickup / Delivery Nodes
- Charging Nodes
'''
class Station():
    
    def __init__(self,nodeId,name, stType, mhTime):
        self.nodeId = int(nodeId) # an integer representing the unique id of a node
        self.name = name #name of the node
        self.stType = stType #station Type (P/D/C/PD)
        self.mhTime = mhTime # material handling time associated with the station 
    
    def __repr__(self):
        return f"Station#{self.nodeId} - {self.name} - {self.stType}"
    
    def getType(self):
        '''
        Returns type of node
        '''
        return self.stType
    
    def getNode(self):
        '''
        returns integer id of the node
        '''
        return self.nodeId
    
    def getMHT(self):
        '''
        returns material handling time at the station
        '''
        return self.mhTime