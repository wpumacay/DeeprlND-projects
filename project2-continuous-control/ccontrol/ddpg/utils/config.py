
import gin
import copy
import numpy as np

@gin.configurable
class DDPGTrainerConfig( object ) :
    r"""Configuration options for the DDPG trainer

    Args:
        numTrainingEpisodes (int)   : number of episodes used for training
        maxStepsInEpisode (int)     : maximum number of steps per episode
        logWindowSize (int)         : size of the logging averaging window (in episodes)
        seed (int)                  : seed for the random-number generators
        sessionID (str)             : name of the session used for training, used as 
                                      savedir or loaddir during training or testing respectively

    """
    def __init__( self,
                  numTrainingEpisodes = 2000,
                  maxStepsInEpisode = 3000,
                  logWindowSize = 100,
                  seed = 0,
                  sessionID = 'session_default' ) :
        super( DDPGTrainerConfig, self ).__init__()

        self.numTrainingEpisodes    = numTrainingEpisodes
        self.maxStepsInEpisode      = maxStepsInEpisode
        self.logWindowSize          = logWindowSize
        self.seed                   = seed
        self.sessionID              = sessionID


@gin.configurable
class DDPGAgentConfig( object ) :
    r"""Configuration options for DDPG based agents

    Args:
        observationsShape (tuple)   : shape of the observations provided to the agent
        actionsShape (tuple)        : shape of the actions that the agent can take
        seed (int)                  : random seed use to initialize the random number generators
        gamma (float)               : discount factor
        tau (float)                 : polyak averaging factor used for soft-updates
        replayBufferSize (int)      : size of the replay buffer
        lrActor (float)             : learning rate to be used for the actor
        lrCritic (float)            : learning rate to be used for the critic
        batchSize (int)             : size of the batch taken from the replay buffer at each learning step
        trainFrequencySteps (int)   : frequency (in steps) at which to take learning steps
        trainNumLearningSteps (int) : number of learning steps to take when learning is required
        noiseType (str)             : type of noise to be used, either (ounoise|normal)
        noiseOUMu (float)           : mu factor for the Ornstein-Uhlenbeck noise process
        noiseOUTheta (float)        : theta factor for the Ornstein-Uhlenbeck noise process
        noiseOUSigma (float)        : sigma factor for the Ornstein-Uhlenbeck noise process
        noiseNormalStddev (float)   : standard deviation of the zero-mean gaussian noise
        epsilonSchedule (str)       : type of schedule to be used to decay epsilon (noise), either 'linear' or 'geometric'
        epsilonFactorGeom (float)   : decay factor (multiplicative) used for the geometric schedule
        epsilonFactorLinear (float) : decay factor (decrement) used for the linear schedule
        trainingStartingStep (int)  : step number at which training actually starts

    """
    def __init__( self,
                  observationsShape = (2,),
                  actionsShape = (2,),
                  seed = 0,
                  gamma = 0.99,
                  tau = 0.001,
                  replayBufferSize = 1000000,
                  lrActor = 0.001,
                  lrCritic = 0.001,
                  batchSize = 256,
                  trainFrequencySteps = 20,
                  trainNumLearningSteps = 10,
                  noiseType = 'ounoise',
                  noiseOUMu = 0.0,
                  noiseOUTheta = 0.15,
                  noiseOUSigma = 0.2,
                  noiseNormalStddev = 0.25,
                  epsilonSchedule = 'linear',
                  epsilonFactorGeom = 0.999,
                  epsilonFactorLinear = 1e-5,
                  trainingStartingStep = 0 ) :
        super( DDPGAgentConfig, self ).__init__()

        self.observationsShape = observationsShape
        self.actionsShape = actionsShape
        self.seed = 0
        self.gamma = gamma
        self.tau = tau
        self.replayBufferSize = replayBufferSize
        self.lrActor = lrActor
        self.lrCritic = lrCritic
        self.batchSize = batchSize
        self.trainFrequencySteps = trainFrequencySteps
        self.trainNumLearningSteps = trainNumLearningSteps
        self.noiseType = noiseType
        self.noiseOUMu = noiseOUMu
        self.noiseOUTheta = noiseOUTheta
        self.noiseOUSigma = noiseOUSigma
        self.noiseNormalStddev = noiseNormalStddev
        self.epsilonSchedule = epsilonSchedule
        self.epsilonFactorGeom = epsilonFactorGeom
        self.epsilonFactorLinear = epsilonFactorLinear
        self.trainingStartingStep = trainingStartingStep


@gin.configurable
class DDPGModelBackboneConfig( object ) :
    r"""Configuration options of the backbone of models used with DDPG based agents

    Args:
        observationsShape (tuple)   : shape of the observation space
        actionsShape (tuple)        : shape of the action space
        inputShape (tuple)          : shape of the input to the model
        outputShape (tuple)         : shape of the output of the model
        layersDefs (list)           : a list of dictionaries each describing a layer of the model
        useBatchnorm (boolean)      : whether or not to use batchnorm in the backbone of the model
        clipGradients (boolean)     : whether or not to clip the norm of the gradients in the layers
        gradientsClipNorm (float)   : norm to which to clip the gradients (if applicable)
        seed (int)                  : seed for random number generators to use

    """
    def __init__( self,
                  observationsShape = (2,),
                  actionsShape = (2,),
                  inputShape = (2,),
                  outputShape = (2,),
                  layersDefs = [],
                  useBatchnorm = True,
                  clipGradients = False,
                  gradientsClipNorm = 1.,
                  seed = 0 ) :
        super( DDPGModelBackboneConfig, self ).__init__()

        self.observationsShape = copy.copy( observationsShape )
        self.actionsShape = copy.copy( actionsShape )
        self.inputShape = copy.copy( inputShape )
        self.outputShape = copy.copy( outputShape )
        self.layersDefs = copy.copy( layersDefs )
        self.useBatchnorm = useBatchnorm
        self.clipGradients = clipGradients
        self.gradientsClipNorm = gradientsClipNorm
        self.seed = seed