

class IBuffer( object ) :

    def __init__( self, bufferSize, randomSeed ) :
        super( IBuffer, self ).__init__()

        # capacity of the buffer
        self._bufferSize = bufferSize

        # seed for random number generator (either numpy's or python's)
        self._randomSeed = randomSeed

    def add( self, state, action, nextState, reward, endFlag ) :
        """Adds a transition tuple into memory
        
        Args:
            state       (object)    : state at timestep t
            action      (int)       : action taken at timestep t
            nextState   (object)    : state from timestep t+1
            reward      (float)     : reward obtained from (state,action)
            endFlag     (bool)      : whether or not nextState is terminal

        """
        raise NotImplementedError( 'IBuffer::add> virtual method' )

    def sample( self, batchSize ) :
        """Adds a transition tuple into memory
        
        Args:
            batchSize (int) : number of experience tuples to grab from memory

        Returns:
            list : a list of experience tuples

        """
        raise NotImplementedError( 'IBuffer::sample> virtual method' )

    def __len__( self ) :
        raise NotImplementedError( 'IBuffer::__len__> virtual method' )