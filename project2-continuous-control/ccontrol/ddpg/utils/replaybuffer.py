
import random
import numpy as np

from collections import deque

from IPython.core.debugger import set_trace

class DDPGReplayBuffer( object ) :

    def __init__( self, bufferSize ) :
        super( DDPGReplayBuffer, self ).__init__()

        self._memory = deque( maxlen = bufferSize )


    def store( self, transition ) :
        self._memory.append( transition )


    def sample( self, batchSize ) :
        _batch = random.sample( self._memory, batchSize )

        _states     = np.vstack( [ _transition[0] for _transition in _batch ] ).astype( np.float32 )
        _actions    = np.vstack( [ _transition[1] for _transition in _batch ] ).astype( np.float32 )
        _rewards    = np.vstack( [ _transition[2] for _transition in _batch ] ).astype( np.float32 )
        _statesNext = np.vstack( [ _transition[3] for _transition in _batch ] ).astype( np.float32 )
        _dones      = np.vstack( [ _transition[4] for _transition in _batch ] ).astype( np.float32 )

        return _states, _actions, _rewards, _statesNext, _dones


    def __len__( self ) :
        return len( self._memory )