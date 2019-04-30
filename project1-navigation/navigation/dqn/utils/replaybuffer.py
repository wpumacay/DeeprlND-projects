
import random
import numpy as np

from collections import namedtuple
from collections import deque

from IPython.core.debugger import set_trace

class DqnReplayBuffer( object ) :

    def __init__( self, bufferSize, randomSeed ) :
        super( DqnReplayBuffer, self ).__init__()

        self._bufferSize = bufferSize
        self._experience = namedtuple( 'Step', 
                                       field_names = [ 'state', 
                                                       'action',
                                                       'reward',
                                                       'nextState',
                                                       'endFlag' ] )

        self._memory = deque( maxlen = bufferSize )

        # seed random generator (@TODO: What is the behav. with multi-agents?)
        random.seed( randomSeed )

    def add( self, state, action, nextState, reward, endFlag ) :
        # create a experience object from the arguments
        _expObj = self._experience( state, action, reward, nextState, endFlag )
        # and add it to the deque memory
        self._memory.append( _expObj )

    def sample( self, batchSize ) :
        # grab a batch from the deque memory
        _expBatch = random.sample( self._memory, batchSize )

        # stack each experience component along batch axis
        _states = np.stack( [ _exp.state for _exp in _expBatch if _exp is not None ] )
        _actions = np.stack( [ _exp.action for _exp in _expBatch if _exp is not None ] )
        _rewards = np.stack( [ _exp.reward for _exp in _expBatch if _exp is not None ] )
        _nextStates = np.stack( [ _exp.nextState for _exp in _expBatch if _exp is not None ] )
        _endFlags = np.stack( [ _exp.endFlag for _exp in _expBatch if _exp is not None ] ).astype( np.uint8 )

        return _states, _actions, _nextStates, _rewards, _endFlags

    def __len__( self ) :
        return len( self._memory )