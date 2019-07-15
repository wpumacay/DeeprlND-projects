
import os
import sys
import gym
import numpy as np

from unityagents import UnityEnvironment

from IPython.core.debugger import set_trace

class IUnityEnvWrapper( gym.Env ) :


    def __init__( self, executableFullPath, numAgents, mode, workerID, seed ) :
        # unity environment to wrap
        self._unityEnv = UnityEnvironment( executableFullPath, seed = seed, worker_id = workerID )
        # executable name
        self._executableFullPath = executableFullPath
        # mode in which the env. will be used
        self._mode = mode
        # observation space
        self.observation_space = None
        # action space
        self.action_space = None

    def seed( self, seed ) :
        pass

    def reset( self ) :
        raise NotImplementedError( 'IUnityEnvWrapper::reset> pure virtual' )


    def step( self ) :
        raise NotImplementedError( 'IUnityEnvWrapper::step> pure virtual' )

    def render( self, mode = 'human', close = False ) :
        # do nothing for now, as the env. rendering is handled separately
        pass


class UnityEnvWrapper( IUnityEnvWrapper ) :


    def __init__( self, executableFullPath, numAgents, mode = 'training', workerID = 0, seed = 0 ) :
        super( UnityEnvWrapper, self ).__init__( executableFullPath, numAgents, mode, workerID, seed )

        # sanity check: at least one brain
        assert len( self._unityEnv.brain_names ) > 0, 'ERROR> no brains in unity-env'

        # number of agents in the scene
        self._numAgents = numAgents

        # this case handles single-brain environments, so just warn the user (use indx 0)
        if len( self._unityEnv.brain_names ) > 1 :
            print( 'WARNING> wrapping a multi-brain unity environment. Using 0th brain' )

        # name of the brain for the single agent in the environment
        self._uBrainName = self._unityEnv.brain_names[0]
        # brain releated for this agent
        self._uBrain = self._unityEnv.brains[ self._uBrainName ]

        # grab observation and action space sizes
        self._uObservationSpaceShape = (self._uBrain.num_stacked_vector_observations * 
                                        self._uBrain.vector_observation_space_size,)
        self._uActionSpaceShape = (self._uBrain.vector_action_space_size,)

        # define the gym-like observation and action spaces (not sure of limits, so min-max of float32)
        self.observation_space = gym.spaces.Box( low = np.finfo(np.float32).min,
                                                 high = np.finfo(np.float32).max,
                                                 shape = self._uObservationSpaceShape )

        if self._uBrain.vector_action_space_type == 'discrete' :
            self.action_space = gym.spaces.Discrete( self._uActionSpaceShape[0] )
        else :
            self.action_space = gym.spaces.Box( low = -1.,
                                                high = 1.,
                                                shape = self._uActionSpaceShape )

    def reset( self ) :
        # request a reset of the environment
        _info = self._unityEnv.reset( train_mode = (self._mode == 'training') )[ self._uBrainName ]

        # sanity check: number of agents in environment should be correct
        assert self._numAgents == len( _info.agents )

        # sanity check: observation should have batch size according to number of agents
        assert len( _info.vector_observations ) == self._numAgents, \
                'ERROR> environment should return a batch of %d observations, but \
                got a batch of %d instead' % ( self._numAgents, len( _info.vector_observations ) )

        # return the full observations for all agents
        return np.array( _info.vector_observations )


    def step( self, actions ) :
        # sanity check: actions given should have batch size according to number of agents
        assert len( actions ) == self._numAgents, \
                'ERROR> actions provided should have batch size of %d, but \
                got a batch of %d instead' % ( self._numAgents, len( actions ) )

        # apply requested action and retrieve info for this single brain
        _stepInfo = self._unityEnv.step( actions )[ self._uBrainName ]

        # grab the required information fron the step-info object
        _observations   = _stepInfo.vector_observations

        _rewards         = np.array( _stepInfo.rewards )
        _dones           = np.array( _stepInfo.local_done )

        return np.array( _observations ), \
               np.array( _rewards ), \
               np.array( _dones ), {}


    def close( self ) :
        self._unityEnv.close()


    @property
    def obsSpaceShape( self ) :
        return self._uObservationSpaceShape


    @property
    def actSpaceShape( self ) :
        return self._uActionSpaceShape


if __name__ == '__main__' :

    executableFullPath = '../../executables/Tennis_Linux/Tennis.x86_64'
    numberOfAgents = 2

    _env = UnityEnvWrapper( executableFullPath,
                            numAgents = numberOfAgents, mode = 'test', workerID = 0, seed = 0 )

    _ss = _env.reset()
    for _ in range( 1000 ) :
        _aa = np.random.randn( *( (numberOfAgents,) + _env.action_space.shape ) )
        ## _aa = np.zeros( ( (numberOfAgents,) + _env.action_space.shape ) )
        _ssnext, _rr, _dd, _ = _env.step( _aa )

        if _dd.any() :
            break

        _ssdelta = _ssnext - _ss
        _ss = _ssnext
        set_trace()