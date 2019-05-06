
import os
import sys
import numpy as np
from collections import deque

from navigation.dqn.core import agent
from navigation.dqn.utils import config 

from IPython.core.debugger import set_trace

USE_GRAYSCALE = True

class DqnBananaVisualAgent( agent.IDqnAgent ) :

    def __init__( self, agentConfig, modelConfig, modelBuilder, backendInitializer ) :
        super( DqnBananaVisualAgent, self ).__init__( agentConfig, modelConfig, modelBuilder, backendInitializer )

        self._frames = deque( maxlen = 4 )

    def _preprocess( self, rawState ) :
        # if queue is empty, just repeat this rawState -------------------------
        if len( self._frames ) < 1 :
            for _ in range( 4 ) :
                self._frames.append( rawState )
        # ----------------------------------------------------------------------

        # send this rawState to the queue
        self._frames.append( rawState )

        # grab the states to be preprocessed
        _frames = list( self._frames )

        if USE_GRAYSCALE :
            # convert each frame into grayscale
            _frames = [ 0.299 * rgb[0,...] + 0.587 * rgb[1,...] + 0.114 * rgb[2,...] \
                        for rgb in _frames ]
            _frames = np.stack( _frames )

        else :
            _frames = np.concatenate( _frames )

        ## set_trace()

        return _frames

AGENT_CONFIG = config.DqnAgentConfig()
AGENT_CONFIG.stateDim                   = (3, 84, 84)
AGENT_CONFIG.nActions                   = 4
AGENT_CONFIG.epsilonSchedule            = 'geometric'
AGENT_CONFIG.epsilonStart               = 1.0
AGENT_CONFIG.epsilonEnd                 = 0.01
AGENT_CONFIG.epsilonDecay               = 0.99925
AGENT_CONFIG.lr                         = 0.0005
AGENT_CONFIG.minibatchSize              = 64
AGENT_CONFIG.learningStartsAt           = 0
AGENT_CONFIG.learningUpdateFreq         = 4
AGENT_CONFIG.learningUpdateTargetFreq   = 4
AGENT_CONFIG.learningMaxSteps           = 4000
AGENT_CONFIG.replayBufferSize           = int( 2 ** 17 ) # 1048576 ~ 1e6 -> power of 2 for exp. replay
AGENT_CONFIG.discount                   = 0.999
AGENT_CONFIG.tau                        = 0.00025
AGENT_CONFIG.seed                       = 0
AGENT_CONFIG.useConvolutionalBasedModel = True

MODEL_CONFIG = config.DqnModelConfig()
MODEL_CONFIG.inputShape = (4 if USE_GRAYSCALE else 12, 84, 84) # stack 4 grayscales, or 4 rgbs
MODEL_CONFIG.outputShape = (4,)
MODEL_CONFIG.saveGradients = False # no gradients for now (for tf filesize is huge)
MODEL_CONFIG.saveBellmanErrors = False
MODEL_CONFIG.layers = [ { 'name': 'fc1', 'type' : 'fc', 'units' : 128, 'activation' : 'relu' },
                        { 'name': 'fc2', 'type' : 'fc', 'units' : 64, 'activation' : 'relu' },
                        { 'name': 'fc3', 'type' : 'fc', 'units' : 16, 'activation' : 'relu' },
                        { 'name': 'fc4', 'type' : 'fc', 'units' : -1, 'activation' : 'linear' } ]

def CreateAgent( agentConfig, modelConfig, modelBuilder, backendInitializer ) :
    return DqnBananaVisualAgent( agentConfig, modelConfig, modelBuilder, backendInitializer )
