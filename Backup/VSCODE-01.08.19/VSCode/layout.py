
from csv import reader
import copy # to deep copy from taskSequence to taskSchedule
from sklearn.preprocessing import normalize # for normalizing distance matrix
#OBJECT IMPORTS



class Layout():
    
    def __init__(self, fileName):
        self.distMat=[]
        self.normalizedDistanceMatrix=[]
        self.readDistanceMatrix(fileName)
        self.setNormalizedDistanceMatrix()
    
    def readDistanceMatrix(self,fileName):
        dm = reader(open(fileName))
        self.distMat = list(dm)
        self.distMat[0][0]=0
        
        for f,fr in enumerate(self.distMat):
            for t,to in enumerate(fr):
                self.distMat[f][t] = float(self.distMat[f][t])
                
    
    def getDistanceFromNode(self, source, dest):
        '''
        returns the distance (in m) between 2 nodes
        '''
        return float(self.distMat[source][dest])
    
    def getMaxDistance(self):
        '''
        returns the maximum distance value
        '''
        maximum_distance=0
        for i in self.distMat:
            for j in i:
                if j>maximum_distance:
                    maximum_distance=j
        return maximum_distance
    
    def setNormalizedDistanceMatrix(self):
        '''
        keeps a normalized version of distance matrix
        '''
        self.normalizedDistanceMatrix = copy.deepcopy(self.distMat)
        self.normalizedDistanceMatrix = normalize(self.normalizedDistanceMatrix)
    
    def getNormalizedDistanceFromNode(self, source, dest):
        '''
        returns normalized distance from distance matrix
        '''
        return float(self.normalizedDistanceMatrix[source][dest])
    
    def getDistanceMatrix(self):
        '''
        returns a copy of the distance matrix
        '''
        return copy.deepcopy(self.distMat) 