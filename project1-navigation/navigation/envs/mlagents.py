
import os
import sys
import numpy as np

from unityagents import UnityEnvironment

from IPython.core.debugger import set_trace

class IUnityEnvWrapper( object ) :

    def __init__( self, unityEnv, execName ) :
        # unity environment to wrap
        self._unityEnv = unityEnv
        # executable name
        self._execName = execName

    def reset( self, mode ) :
        raise NotImplementedError( 'IUnityEnvWrapper::reset> pure virtual' )

    def step( self ) :
        raise NotImplementedError( 'IUnityEnvWrapper::step> pure virtual' )


class SingleAgentDiscreteActionsEnv( IUnityEnvWrapper ) :

    def __init__( self, unityEnv, execName ) :
        super( SingleAgentDiscreteActionsEnv, self ).__init__( unityEnv, execName )

        # sanity check: at least one brain
        assert len( self._unityEnv.brain_names ) > 0, 'ERROR> no brains in unity-env'

        # this case handles single-brain environments, so just warn the user (use indx 0)
        if len( self._unityEnv.brain_names ) > 1 :
            print( 'WARNING> wrapping a multi-brain unity environment. Using 0th brain' )

        # name of the brain for the single agent in the environment
        self._uBrainName = self._unityEnv.brain_names[0]
        # brain releated for this agent
        self._uBrain = self._unityEnv.brains[ self._uBrainName ]
        # flag active only if visual-observations are available
        self._uHasVisualObservations = hasattr( self._uBrain, 'number_visual_observations' ) and \
                                       hasattr( self._uBrain, 'camera_resolutions' )

        if hasattr( self._uBrain, 'number_visual_observations' ) :
            self._uHasVisualObservations = self._uBrain.number_visual_observations != 0

        # number of visual observations
        self._uNumVisualObservations = 0 if not self._uHasVisualObservations else \
                                       self._uBrain.number_visual_observations

        # sanity check: action space must be discrete
        assert self._uBrain.vector_action_space_type == 'discrete', \
               'ERROR> environment %s must have a discrete action space' % ( self._execName )

        # sanity check: observation space must be continuous
        assert self._uBrain.vector_observation_space_type == 'continuous', \
               'ERROR> environment %s must have a continuous observation space' % ( self._execName )

        # @TODO: For now, the wrapper only accepts either pure vector observations, ...
        #        or pure virtual observations. If the later are present, we use them ...
        #        as the only observations, and use their shape of the observation space.

        # grab observation and action space sizes
        if self._uHasVisualObservations :
            # grab camera resolutions
            _cameraResolution   = self._uBrain.camera_resolutions[0]
            _cameraWidth        = _cameraResolution['width']
            _cameraHeight       = _cameraResolution['height']
            _cameraNumChannels  = 1 if _cameraResolution['blackAndWhite'] else 3

            self._uObservationsShape = (_cameraNumChannels, _cameraHeight, _cameraWidth )
        else :
            self._uObservationsShape = (self._uBrain.vector_observation_space_size,)

        self._uNumActions = self._uBrain.vector_action_space_size

    def reset( self, training = False ) :
        # request a reset of the environment
        _info = self._unityEnv.reset( train_mode = training )[ self._uBrainName ]

        # grab the observations from the info object
        if self._uHasVisualObservations :
            # remove batch shape, and transpose it to (depth, height, width) format
            _observations = np.transpose( np.squeeze( _info.visual_observations[0], 
                                                      axis = 0 ), 
                                          (2, 0, 1) )
        else :
            _observations = _info.vector_observations[0]

        return _observations

    def step( self, action ) :
        # apply requested action and retrieve info for this single brain
        _stepInfo = self._unityEnv.step( action )[ self._uBrainName ]

        # grab the required information fron the step-info object
        if self._uHasVisualObservations :
            _observations   = np.transpose( np.squeeze( _stepInfo.visual_observations[0], 
                                                        axis = 0 ), 
                                            (2, 0, 1) )
        else :
            _observations   = _stepInfo.vector_observations[0]

        _reward         = _stepInfo.rewards[0]
        _done           = _stepInfo.local_done[0]

        return _observations, _reward, _done, {}

    def close( self ) :
        self._unityEnv.close()

    @property
    def obsShape( self ) :
        return self._uObservationsShape

    @property
    def numActions( self ) :
        return self._uNumActions

def createDiscreteActionsEnv( executableFullPath, envType = 'single', seed = 0 ) :
    _unityEnv = UnityEnvironment( executableFullPath, seed = seed )

    if envType == 'single' :
        return SingleAgentDiscreteActionsEnv( _unityEnv, executableFullPath )
    else :
        print( 'ERROR> multi-simulations with MPI not supported yet' )
        sys.exit( 1 )