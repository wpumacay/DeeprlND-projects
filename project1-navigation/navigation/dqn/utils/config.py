
import random
import json
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

    def serialize( self ) :
        _dataDict = {}

        _dataDict['stateDim']                   = self.stateDim
        _dataDict['nActions']                   = self.nActions
        _dataDict['epsilonStart']               = self.epsilonStart
        _dataDict['epsilonEnd']                 = self.epsilonEnd
        _dataDict['epsilonSteps']               = self.epsilonSteps
        _dataDict['epsilonDecay']               = self.epsilonDecay
        _dataDict['epsilonSchedule']            = self.epsilonSchedule
        _dataDict['lr']                         = self.lr
        _dataDict['minibatchSize']              = self.minibatchSize
        _dataDict['learningStartsAt']           = self.learningStartsAt
        _dataDict['learningUpdateFreq']         = self.learningUpdateFreq
        _dataDict['learningUpdateTargetFreq']   = self.learningUpdateTargetFreq
        _dataDict['learningMaxSteps']           = self.learningMaxSteps
        _dataDict['replayBufferSize']           = self.replayBufferSize
        _dataDict['discount']                   = self.discount
        _dataDict['seed']                       = self.seed

        return _dataDict

    @classmethod
    def save( cls, config, filename ) :
        with open( filename, 'w' ) as fhandle :
            json.dump( config.serialize(), fhandle, indent = 4 )

    @classmethod
    def load( cls, filename ) :
        with open( filename, 'r' ) as fhandle :
            _dataDict = json.load( fhandle )

        _config = DqnAgentConfig()

        # @TODO: Change to a simpler way (using **dataDict as kwargs to constructor)
        _config.stateDim                   = _dataDict['stateDim']
        _config.nActions                   = _dataDict['nActions']
        _config.epsilonStart               = _dataDict['epsilonStart']
        _config.epsilonEnd                 = _dataDict['epsilonEnd']
        _config.epsilonSteps               = _dataDict['epsilonSteps']
        _config.epsilonDecay               = _dataDict['epsilonDecay']
        _config.epsilonSchedule            = _dataDict['epsilonSchedule']
        _config.lr                         = _dataDict['lr']
        _config.minibatchSize              = _dataDict['minibatchSize']
        _config.learningStartsAt           = _dataDict['learningStartsAt']
        _config.learningUpdateFreq         = _dataDict['learningUpdateFreq']
        _config.learningUpdateTargetFreq   = _dataDict['learningUpdateTargetFreq']
        _config.learningMaxSteps           = _dataDict['learningMaxSteps']
        _config.replayBufferSize           = _dataDict['replayBufferSize']
        _config.discount                   = _dataDict['discount']
        _config.seed                       = _dataDict['seed']

        return _config

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

    def serialize( self ) :
        _dataDict = {}

        _dataDict['inputShape']         = self.inputShape
        _dataDict['outputShape']        = self.outputShape
        _dataDict['saveGradients']      = self.saveGradients
        _dataDict['saveBellmanErrors']  = self.saveBellmanErrors
        _dataDict['layers']             = self.layers

        return _dataDict

    @classmethod
    def save( cls, config, filename ) :
        with open( filename, 'w' ) as fhandle :
            json.dump( config.serialize(), fhandle, indent = 4 )

    @classmethod
    def load( cls, filename ) :
        with open( filename, 'r' ) as fhandle :
            _dataDict = json.load( fhandle )

        _config = DqnModelConfig()

        _config.inputShape         = _dataDict['inputShape']
        _config.outputShape        = _dataDict['outputShape']
        _config.saveGradients      = _dataDict['saveGradients']
        _config.saveBellmanErrors  = _dataDict['saveBellmanErrors']
        _config.layers             = _dataDict['layers']

        return _config