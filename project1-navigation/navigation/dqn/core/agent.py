
import numpy as np

# our helpers
from navigation.dqn.utils import replaybuffer

# debugging helpers
from IPython.core.debugger import set_trace

class IDqnAgent( object ) :

    def __init__( self, agentConfig, modelConfig, modelBuilder, backendInitializer ) :
        """Constructs a generic Dqn agent, given configuration information

        Args:
            agentConfig (DqnAgentConfig) : config object with agent parameters
            modelConfig (DqnModelConfig) : config object with model parameters
            modelBuilder (IDqnModel) : factory function to instantiate the model

        """

        super( IDqnAgent, self ).__init__()

        # environment state and action spaces info
        self._stateDim = agentConfig.stateDim
        self._nActions = agentConfig.nActions

        # random seed
        self._seed = agentConfig.seed
        np.random.seed( self._seed )

        # parameters for linear schedule of eps
        self._epsStart      = agentConfig.epsilonStart
        self._epsEnd        = agentConfig.epsilonEnd
        self._epsSteps      = agentConfig.epsilonSteps
        self._epsDecay      = agentConfig.epsilonDecay
        self._epsSchedule   = agentConfig.epsilonSchedule
        self._epsilon       = self._epsStart

        # learning rate and related parameters
        self._lr                        = agentConfig.lr
        self._minibatchSize             = agentConfig.minibatchSize
        self._learningStartsAt          = agentConfig.learningStartsAt
        self._learningUpdateFreq        = agentConfig.learningUpdateFreq
        self._learningUpdateTargetFreq  = agentConfig.learningUpdateTargetFreq
        self._learningMaxSteps          = agentConfig.learningMaxSteps

        # size of replay buffer
        self._replayBufferSize = agentConfig.replayBufferSize

        # discount factor gamma
        self._gamma = agentConfig.discount

        # tau factor for soft-updates
        self._tau = agentConfig.tau

        # some counters used by the agent's logic
        self._istep = 0
        self._iepisode = 0

        # copy some parameters from the agent config into the model config
        modelConfig._lr = self._lr

        # create the model accordingly
        self._qmodel_actor = modelBuilder( 'actor_model', modelConfig, True )
        self._qmodel_target = modelBuilder( 'target_model', modelConfig, False )

        # initialize backend-specific functionality
        _initInfo = backendInitializer()
        self._qmodel_actor.initialize( _initInfo )
        self._qmodel_target.initialize( _initInfo )

        self._qmodel_target.clone( self._qmodel_actor, tau = 1.0 )

        # replay buffer
        self._rbuffer = replaybuffer.DqnReplayBuffer( self._replayBufferSize,
                                                      self._seed )

        # states (current and next) for the model representation
        self._currState = None
        self._nextState = None

        self._printConfig();

    def save( self, filename ) :
        if self._qmodel_actor :
            self._qmodel_actor.save( filename )

    def load( self, filename ) :
        if self._qmodel_actor :
            self._qmodel_actor.load( filename )
            self._qmodel_target.clone( self._qmodel_actor, tau = 1.0 )

    def act( self, state, inference = False ) :
        if inference or np.random.rand() > self._epsilon :
            return np.argmax( self._qmodel_actor.eval( self._preprocess( state ) ) )
        else :
            return np.random.choice( self._nActions )

    def step( self, transition ) :
        # grab information from this transition
        _s, _a, _snext, _r, _done = transition
        # preprocess the raw state
        self._nextState = self._preprocess( _snext )
        if self._currState is None :
            self._currState = self._preprocess( _s ) # for first step
        # store in replay buffer
        self._rbuffer.add( self._currState, _a, self._nextState, _r, _done )

        # check if can do a training step
        if self._istep > self._learningStartsAt and \
           self._istep % self._learningUpdateFreq == 0 and \
           len( self._rbuffer ) >= self._minibatchSize :
            self._learn()

        # update the parameters of the target model (every update_target steps)
        if self._istep > self._learningStartsAt and \
           self._istep % self._learningUpdateTargetFreq == 0 :
           self._qmodel_target.clone( self._qmodel_actor, tau = self._tau )

        # save next state (where we currently are in the environment) as current
        self._currState = self._nextState

        # update the agent's step counter
        self._istep += 1
        # and the episode counter if we finished an episode, and ...
        # the states as well (I had a bug here, becasue I didn't ...
        # reset the states).
        if _done :
            self._iepisode += 1
            self._currState = None
            self._nextState = None

        # check epsilon update schedule and update accordingly
        if self._epsSchedule == 'linear' :
            # update epsilon using linear schedule
            _epsFactor = 1. - ( max( 0, self._istep - self._learningStartsAt ) / self._epsSteps )
            _epsDelta = max( 0, ( self._epsStart - self._epsEnd ) * _epsFactor )
            self._epsilon = self._epsEnd + _epsDelta

        elif self._epsSchedule == 'geometric' :
            if _done :
                # update epsilon with a geometric decay given by a decay factor
                _epsFactor = self._epsDecay if self._istep >= self._learningStartsAt else 1.0
                self._epsilon = max( self._epsEnd, self._epsilon * _epsFactor )

    def _preprocess( self, rawState ) :
        """Preprocess a raw state into an appropriate state representation
    
        Args:
            rawState (np.ndarray) : raw state to be transformed

        Returns:
            np.ndarray : preprocess state into the approrpiate representation
        """

        """ OVERRIDE this method with your specific preprocessing """

        raise NotImplementedError( 'IDqnAgent::_preprocess> virtual method' )
        
    def _learn( self ) :
        """Makes a learning step using the DQN algorithm from Mnih et. al.
           https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

        """

        # get a minibatch from the replay buffer
        _minibatch = self._rbuffer.sample( self._minibatchSize )
        _states, _actions, _nextStates, _rewards, _dones = _minibatch

        # compute targets (in a vectorized way). Recall:
        #               
        #            |->   _reward                            if s' is a terminal
        # q-target = |     
        #            |->   _reward + gamma * max( Q(s',a') )  otherwise
        #                                     a'
        # Or in vectorized form ( recall Q(s') computes all qvalues ) :
        #
        # qtargets = _rewards + (1 - terminals) * gamma * max(Q(nextStates), batchAxis)
        #
        # Notes (for nnetwork models): 
        #       * Just to clarify, we are assuming that in this call to Q
        #         the targets generated are not dependent of the weights
        #         of the network (should not take into consideration gradients 
        #         here, nor take them as part of the computation graph).
        #         Basically the targets are like training data from a 'dataset'.

        _qtargets = _rewards + ( 1 - _dones ) * self._gamma * \
                    np.max( self._qmodel_target.eval( _nextStates ), 1 )
        _qtargets = _qtargets.astype( np.float32 )

        # make the learning call to the model (kind of like supervised setting)
        self._qmodel_actor.train( _states, _actions, _qtargets )

    @property
    def epsilon( self ) :
        return self._epsilon

    @property
    def seed( self ) :
        return self._seed
        
    @property
    def learningMaxSteps( self ) :
        return self._learningMaxSteps
    

    def _printConfig( self ) :
        print( '#############################################################' )
        print( '#                                                           #' )
        print( '#                 Agent configuration                       #' )
        print( '#                                                           #' )
        print( '#############################################################' )

        print( 'state space dimension                           : ', self._stateDim )
        print( 'number of actions                               : ', self._nActions )
        print( 'seed                                            : ', self._seed )
        print( 'epsilon start value                             : ', self._epsStart )
        print( 'epsilon end value                               : ', self._epsEnd )
        print( 'epsilon schedule type                           : ', self._epsSchedule )
        print( 'epsilon linear decay steps                      : ', self._epsSteps )
        print( 'epsilon geom. decay factor                      : ', self._epsDecay )
        print( 'learning rate                                   : ', self._lr )
        print( 'minibatch size                                  : ', self._minibatchSize )
        print( 'learning starting step for training             : ', self._learningStartsAt )
        print( 'learning updateFreq (training actor-model)      : ', self._learningUpdateFreq )
        print( 'learning updateTargetFreq (target-model)        : ', self._learningUpdateTargetFreq )
        print( 'learning max steps                              : ', self._learningMaxSteps )
        print( 'replay buffer size                              : ', self._replayBufferSize )
        print( 'gamma (discount factor)                         : ', self._gamma )
        print( 'tau (target model soft-updates)                 : ', self._tau )

        print( '#############################################################' )
