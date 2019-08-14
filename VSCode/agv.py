class AGV():
    def __init__(self, agvId, startNode, caps, speed, charge, dischargeRate, chargeRate, taskList, travelCost, low=30, up=60):
        
        self.agvId = agvId #agv id
        self.startNode = startNode #initial node
        self.caps = caps #capability
        self.speed = speed # speed in m/s
        self.charge = charge # charge %
        self.dischargeRate = dischargeRate # % per second
        self.chargeRate = chargeRate # % per second
        self.travelCost = travelCost # weighted travel cost
        self.LOWER_THRESHOLD = low # lower threshold of charging
        self.UPPER_THRESHOLD =up # upper threshold of charging
        self.state = 'N'
    
    def __repr__(self):
        return f"{self.agvId}-{self.speed}m/s-{self.charge}%-{self.travelCost}-{self.UPPER_THRESHOLD}%"
    
    def getSpeed(self):
        '''
        Returns speed of agv in m/s
        '''
        return self.speed
    
    def getChargeRate(self):
        '''
        Returns charging rate of agv in %/sec
        '''
        return self.chargeRate
    
    def getDischargeRate(self):
        '''
        Returns discharging rate of agv in %/sec
        '''
        return self.dischargeRate
    
    def getInitialCharge(self):
        ''' returns agv initial charge'''
        return self.charge
    