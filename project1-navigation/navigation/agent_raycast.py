
import os
import sys
import numpy as np

from navigation.dqn.core import agent
from navigation.dqn.utils import config 

class DqnBananaRaycastAgent( agent.IDqnAgent ) :

    def __init__( self, agentConfig, modelConfig, modelBuilder, backendInitializer ) :
        super( DqnBananaRaycastAgent, self ).__init__( agentConfig, modelConfig, modelBuilder, backendInitializer )

    def _preprocess( self, rawState ) :
        """Default preprocessing by just copying the data

        Args:
            rawState (np.ndarray) : raw state from lunar lander environment

        Returns:
            np.ndarray : copy of the gym-env. observation for the model

        """
        return rawState.copy()

_AGENT_CONFIG = config.DqnAgentConfig()
_AGENT_CONFIG.stateDim                   = 37
_AGENT_CONFIG.nActions                   = 4
_AGENT_CONFIG.epsilonSchedule            = 'geometric'
_AGENT_CONFIG.epsilonStart               = 1.0
_AGENT_CONFIG.epsilonEnd                 = 0.1
_AGENT_CONFIG.epsilonDecay               = 0.995
_AGENT_CONFIG.lr                         = 0.0005
_AGENT_CONFIG.minibatchSize              = 64
_AGENT_CONFIG.learningStartsAt           = 0
_AGENT_CONFIG.learningUpdateFreq         = 4
_AGENT_CONFIG.learningUpdateTargetFreq   = 4
_AGENT_CONFIG.learningMaxSteps           = 2000
_AGENT_CONFIG.replayBufferSize           = 10000
_AGENT_CONFIG.discount                   = 0.99
_AGENT_CONFIG.tau                        = 0.001
_AGENT_CONFIG.seed                       = 0

_MODEL_CONFIG = config.DqnModelConfig()
_MODEL_CONFIG.inputShape = (37,)
_MODEL_CONFIG.outputShape = (4,)
_MODEL_CONFIG.layers = [ { 'name': 'fc1', 'type' : 'fc', 'units' : 128, 'activation' : 'relu' },
                         { 'name': 'fc2', 'type' : 'fc', 'units' : 64, 'activation' : 'relu' },
                         { 'name': 'fc3', 'type' : 'fc', 'units' : 16, 'activation' : 'relu' },
                         { 'name': 'fc4', 'type' : 'fc', 'units' : -1, 'activation' : 'linear' } ]

def CreateAgent( modelBuilder, backendInitializer ) :
    return DqnBananaRaycastAgent( _AGENT_CONFIG, _MODEL_CONFIG, modelBuilder, backendInitializer )
