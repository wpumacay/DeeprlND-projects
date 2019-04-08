
import random
import numpy as np
from collections import deque, namedtuple


class ReplayBuffer( object ) :

    def __init__( self, bufferSize, batchSize, randomSeed ) :
        super( ReplayBuffer, self ).__init__()

        self._bufferSize = bufferSize
        self._batchSize = batchSize
        self._experience = namedtuple( 'Step', 
                                       field_names = [ 'state', 
                                                       'action',
                                                       'reward',
                                                       'nextState',
                                                       'finished' ] )

        self._memory = deque( maxlen = bufferSize )
        self._randomState = random.seed( randomSeed )


    def add( self, state, action, reward, nextState, finished ) :
        # create a experience object from the arguments
        _expObj = self._experience( state, action, reward, nextState, finished )

        # and add it to the deque memory
        self._memory.append( _expObj )

    def sample( self ) :
        # grab a batch from the deque memory
        _expBatch = random.sample( self._memory, self._batchSize )

        # transform to torch friendly data
        # @WIP

        return _expBatch

    def __len__( self ) :
        return len( self._memory )