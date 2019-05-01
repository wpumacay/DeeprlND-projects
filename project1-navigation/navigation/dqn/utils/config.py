
import random
import numpy as np

from IPython.core.debugger import set_trace

class DqnAgentConfig( object ) :

    def __init__( self ) :
        super( DqnAgentConfig, self ).__init__()

        # environment state and action info
        self.stateDim = 7056
        self.nActions = 18

        # parameters for linear schedule of eps
        self.epsilonStart       = 1.0
        self.epsilonEnd         = 0.1
        self.epsilonSteps       = 100000
        self.epsilonDecay       = 0.995
        self.epsilonSchedule    = 'linear'

        # learning rate and related parameters
        self.lr                         = 0.00025
        self.minibatchSize              = 32
        self.learningStartsAt           = 50000
        self.learningUpdateFreq         = 4
        self.learningUpdateTargetFreq   = 10000
        self.learningMaxSteps           = 50000000

        # size of replay buffer
        self.replayBufferSize = 1000000

        # discount factor
        self.discount = 0.99

        # tau factor to control interpolation in target-network params
        self.tau = 1.0 # 1.0 means just copy as is from actor to target network

        # random seed
        self.seed = 1

    @classmethod
    def load( filename ) :
        return None

class DqnModelConfig( object ) :

    def __init__( self ) :
        super( DqnModelConfig, self ).__init__()

        # shape of the input tensor for the model
        self.inputShape = ( 4, 84, 84 )
        self.outputShape = ( 18, )
        self.saveGradients = False
        self.saveBellmanErrors = False
        self.layers = [ { 'name' : 'conv1' , 'type' : 'conv2d', 'ksize' : 8, 'kstride' : 4, 'nfilters' : 32, 'activation' : 'relu' },
                        { 'name' : 'conv2' , 'type' : 'conv2d', 'ksize' : 4, 'kstride' : 2, 'nfilters' : 64, 'activation' : 'relu' },
                        { 'name' : 'conv3' , 'type' : 'conv2d', 'ksize' : 3, 'kstride' : 1, 'nfilters' : 64, 'activation' : 'relu' },
                        { 'name' : 'flatten' , 'type' : 'flatten' },
                        { 'name' : 'fc1' , 'type' : 'fc', 'units' : 512, 'activation' : 'relu' },
                        { 'name' : 'fc2' , 'type' : 'fc', 'units' : 18, 'activation' : 'relu' } ]

        # parameters copied from the agent configuration
        self._lr = 0.00025

    @classmethod
    def load( filename ) :
        return None
