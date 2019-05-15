
import os
import sys
import numpy as np

from navigation.dqn.core import agent
from navigation.dqn.utils import config 

class DqnGymControlAgent( agent.IDqnAgent ) :

    def __init__( self, agentConfig, modelConfig, modelBuilder, backendInitializer ) :
        super( DqnGymControlAgent, self ).__init__( agentConfig, modelConfig, modelBuilder, backendInitializer )

    def _preprocess( self, rawState ) :
        """Default preprocessing by just copying the data

        Args:
            rawState (np.ndarray) : raw state from lunar lander environment

        Returns:
            np.ndarray : copy of the gym-env. observation for the model

        """
        return rawState.copy()

AGENT_CONFIG = config.DqnAgentConfig()
AGENT_CONFIG.stateDim                   = -1 # written by trainer
AGENT_CONFIG.nActions                   = -1 # written by trainer
AGENT_CONFIG.epsilonSchedule            = 'geometric'
AGENT_CONFIG.epsilonStart               = 1.0
AGENT_CONFIG.epsilonEnd                 = 0.01
AGENT_CONFIG.epsilonDecay               = 0.9975
AGENT_CONFIG.lr                         = 0.0005
AGENT_CONFIG.minibatchSize              = 64
AGENT_CONFIG.learningStartsAt           = 0
AGENT_CONFIG.learningUpdateFreq         = 4
AGENT_CONFIG.learningUpdateTargetFreq   = 4
AGENT_CONFIG.learningMaxSteps           = 2000
AGENT_CONFIG.replayBufferSize           = int( 2 ** 17 ) # 131072 ~ 1e5 -> power of 2 for exp. replay
AGENT_CONFIG.discount                   = 0.999
AGENT_CONFIG.tau                        = 0.001
AGENT_CONFIG.seed                       = 0
AGENT_CONFIG.useConvolutionalBasedModel = False

MODEL_CONFIG = config.DqnModelConfig()
MODEL_CONFIG.inputShape = ( -1, ) # written by trainer
MODEL_CONFIG.outputShape = ( -1, ) # written by trainer
MODEL_CONFIG.saveGradients = False # no gradients for now (as tf filesize is huge, possibly a bug in my tf implementation)
MODEL_CONFIG.saveBellmanErrors = False
MODEL_CONFIG.layers = [ { 'name': 'fc1', 'type' : 'fc', 'units' : 128, 'activation' : 'relu' },
                        { 'name': 'fc2', 'type' : 'fc', 'units' : 64, 'activation' : 'relu' },
                        { 'name': 'fc3', 'type' : 'fc', 'units' : 16, 'activation' : 'relu' },
                        { 'name': 'fc4', 'type' : 'fc', 'units' : -1, 'activation' : 'linear' } ]

def CreateAgent( agentConfig, modelConfig, modelBuilder, backendInitializer ) :
    return DqnGymControlAgent( agentConfig, modelConfig, modelBuilder, backendInitializer )
