
import os
import sys
import numpy as np

from navigation.dqn.core import agent
from navigation.dqn.utils import config 

class DqnGridworldAgent( agent.IDqnAgent ) :

    def __init__( self, agentConfig, modelConfig, modelBuilder, backendInitializer ) :
        super( DqnGridworldAgent, self ).__init__( agentConfig, modelConfig, modelBuilder, backendInitializer )

    def _preprocess( self, rawState ) :
        # rawState is an index, so convert it to a one-hot representation
        _stateOneHot = np.zeros( self._stateDim )
        _stateOneHot[rawState] = 1.0

        return _stateOneHot

AGENT_CONFIG = config.DqnAgentConfig()
AGENT_CONFIG.stateDim                   = -1 # filled by trainer
AGENT_CONFIG.nActions                   = -1 # filled by trainer
AGENT_CONFIG.epsilonSchedule            = 'geometric'
AGENT_CONFIG.epsilonStart               = 1.0
AGENT_CONFIG.epsilonEnd                 = 0.1
AGENT_CONFIG.epsilonDecay               = 0.995
AGENT_CONFIG.lr                         = 0.001
AGENT_CONFIG.minibatchSize              = 32
AGENT_CONFIG.learningStartsAt           = 0
AGENT_CONFIG.learningUpdateFreq         = 4
AGENT_CONFIG.learningUpdateTargetFreq   = 4
AGENT_CONFIG.learningMaxSteps           = 1000
AGENT_CONFIG.replayBufferSize           = 10000
AGENT_CONFIG.discount                   = 0.999
AGENT_CONFIG.tau                        = 0.001
AGENT_CONFIG.seed                       = 0

MODEL_CONFIG = config.DqnModelConfig()
MODEL_CONFIG.inputShape     = (-1,) # filled by trainer
MODEL_CONFIG.outputShape    = (-1,) # filled by trainer
MODEL_CONFIG.layers = [ { 'name': 'fc1', 'type' : 'fc', 'units' : 128, 'activation' : 'relu' },
                        { 'name': 'fc2', 'type' : 'fc', 'units' : 64, 'activation' : 'relu' },
                        { 'name': 'fc3', 'type' : 'fc', 'units' : 16, 'activation' : 'relu' },
                        { 'name': 'fc4', 'type' : 'fc', 'units' : -1, 'activation' : 'linear' } ]

def CreateAgent( agentConfig, modelConfig, modelBuilder, backendInitializer ) :
    return DqnGridworldAgent( agentConfig, modelConfig, modelBuilder, backendInitializer )
