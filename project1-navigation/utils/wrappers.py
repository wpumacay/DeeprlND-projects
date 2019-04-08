
from unityagents import UnityEnvironment


class IUnityEnvWrapper( object ) :

    def __init__( self, unityEnv ) :

        self._unityEnv = unityEnv

    def reset( self, mode ) :
        raise NotImplementedError( 'IUnityEnvWrapper::reset> pure virtual' )

    def step( self ) :
        raise NotImplementedError( 'IUnityEnvWrapper::step> pure virtual' )


class SingleAgentEnv( IUnityEnvWrapper ) :

    def __init__( self, unityEnv, brainName ) :
        super( SingleAgentEnv, self ).__init__( unityEnv )

        # name of the brain for the single agent in the environment
        self._uBrainName = brainName
        # brain releated for this agent
        self._uBrain = self._unityEnv.brains[ self._uBrainName ]
        # observation space shape (initialize at reset)
        self._uObservationsShape = None
        # action space shape (initialize at reset)
        self._uActionsShape = None



    def reset( self, mode ) :
        _info = self._unityEnv.reset( train_mode = mode )
        # grab the observations from the info object
        _observations = _info.vector_observations[0]
        # set the action and observation shapes
        self._uObservationsShape = _observations.shape
        self._uActionsShape = self._uBrain.vector_action_space_size

        return _observations

    def step( self, action ) :
        # apply actions in the environment
        _stepInfo = self._unityEnv.step( action )[ self._uBrainName ]
        # grab the required information fron the step-info object
        _observations = _stepInfo.vector_observations[0]
        _reward = _stepInfo.rewards[0]
        _done = _stepInfo.local_done[0]

        return _observations, _reward, _done