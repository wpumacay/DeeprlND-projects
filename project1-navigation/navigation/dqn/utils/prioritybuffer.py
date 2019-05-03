
import random
import numpy as np

from collections import namedtuple
from navigation.dqn.utils import segmentree

from IPython.core.debugger import set_trace

class PriorityBuffer( object ) :

    def __init__( self, 
                  bufferSize, 
                  randomSeed, 
                  eps = 0.01, 
                  alpha = 0.6, 
                  beta = 0.4,
                  dbeta = 0.001 ) :

        super( PriorityBuffer, self ).__init__()

        self._bufferSize = bufferSize

        # hyperparameters of Prioritized experience replay
        self._eps   = eps    # extra ammount added to the abs(tderror) to avoid zero probs.
        self._alpha = alpha  # regulates how much the priority affects the probs of sampling
        self._beta  = beta   # regulates how much away from true importance sampling we go (up annealed to 1)
        self._dbeta = dbeta  # regulates how much we anneal up the previous regulator of importance sampling

        # a handy experience tuple constructor
        self._experience = namedtuple( 'Step', 
                                       field_names = [ 'state', 
                                                       'action',
                                                       'reward',
                                                       'nextState',
                                                       'endFlag' ] )
        # sumtree for taking the appropriate samples
        self._sumtree = segmentree.SumTree( self._bufferSize )
        # mintree for taking the actual min as we go
        self._mintree = segmentree.MinTree( self._bufferSize )

        # a variable to store the running max priority
        self._maxpriority = 1
        # a variable to store the running min priority
        self._minpriority = eps

        ## # seed random generator (@TODO: What is the behav. with multi-agents?)
        ## random.seed( randomSeed ) # no need to seed this generator. Using seeded np generator

    def add( self, state, action, nextState, reward, endFlag ) :
        """Adds an experience tuple to memory

        Args:
            state (np.ndarray)      : state of the environment at time (t)
            action (int)            : action taken at time (t)
            nextState (np.ndarray)  : state of the environment at time (t+1) after taking action
            reward (float)          : reward at time (t+1) for this transition
            endFlag (bool)          : flag that indicates if this state (t+1) is terminal

        """

        # create a experience object from the arguments
        _expObj = self._experience( state, action, reward, nextState, endFlag )

        # store the data into a node in the smtree, with nodevalue equal its priority
        # maxpriority is used here, to ensure these tuples can be sampled later
        self._sumtree.add( _expObj, self._maxpriority ** self._alpha )
        self._mintree.add( _expObj, self._maxpriority ** self._alpha )

    def sample( self, batchSize ) :
        """Samples a batch of data using consecutive sampling intervals over the sumtree ranges

        Args:
            batchSize (int) : number of experience tuples to grab from memory

        Returns:
            (indices, experiences) : a tuple of indices (for later updates) and 
                                     experiences from memory

        Example:

                    29
                   /  \
                  /    \
                 13     16        
                |  |   |  |
               3  10  12  4       |---| |----------| |------------| |----|
                                    3        10            12          4
                                  ^______^______^_______^_______^_______^
                                        *      *      *   *    *     *

            5 samples using intervals, and got 10, 10, 12, 12, 4
        """
        # experiences sampled, indices and importance sampling weights
        _expBatch = []
        _indicesBatch = []
        _impSampWeightsBatch = []

        # compute intervals sizes for sampling
        _prioritySegSize = self._sumtree.sum() / batchSize

        # min node-value (priority) in sumtree
        _minPriority = self._mintree.min()
        # min probability that a node can have
        _minProb = _minPriority / self._sumtree.sum()

        # take sampls using segments over the total range of the sumtree
        for i in range( batchSize ) :
            # left and right ticks of the segments
            _a = _prioritySegSize * i
            _b = _prioritySegSize * ( i + 1 )

            # throw the dice over this segment
            _v = np.random.uniform( _a, _b )
            _indx, _priority, _exp = self._sumtree.getNode( _v )

            # Recall how importance sampling weight is computed (from paper)
            #
            # E   { r } = E    { p * r  } = E     { w * r }  -> w : importance 
            #  r~p         r~p'  -           r~p'                   sampling
            #                    p'                                 weight
            #
            # in our case:
            #  p  -> uniform distribution
            #  p' -> distribution induced by the priorities
            #
            #      1 / N
            #  w = ____   
            #
            #       P(i) -> given by sumtree (priority / total)
            #
            # for stability, the authors scale the weight by the max-weight ...
            # possible, which is (because maximizing a fraction minimizes the ...
            # denominator if the numrerator is constant=1/N) the weight of the ...
            # node with minimum probability. After some operations :
            # 
            #                          b                     b
            # w / wmax = ((1/N) / P(i))   / ((1/N) / minP(j))   
            #                                          j
            #                               b                      -b
            # w / wmax = ( min P(j) / P(i) )   = ( P(i) / min P(j) )
            #               j                              j

            # compute importance sampling weights
            _prob = _priority / self._sumtree.sum()
            _impSampWeight = ( _prob / _minProb ) ** -self._beta

            # accumulate in batch
            _expBatch.append( _exp )
            _indicesBatch.append( _indx )
            _impSampWeightsBatch.append( _impSampWeight )

        # stack each experience component along batch axis
        _states = np.stack( [ _exp.state for _exp in _expBatch if _exp is not None ] )
        _actions = np.stack( [ _exp.action for _exp in _expBatch if _exp is not None ] )
        _rewards = np.stack( [ _exp.reward for _exp in _expBatch if _exp is not None ] )
        _nextStates = np.stack( [ _exp.nextState for _exp in _expBatch if _exp is not None ] )
        _endFlags = np.stack( [ _exp.endFlag for _exp in _expBatch if _exp is not None ] ).astype( np.uint8 )

        # convert indices and importance sampling weights to numpy-friendly data
        _indicesBatch = np.array( _indicesBatch ).astype( np.int64 )
        _impSampWeightsBatch = np.array( _impSampWeightsBatch ).astype( np.float32 )

        return _states, _actions, _nextStates, _rewards, _endFlags, _indicesBatch, _impSampWeightsBatch

    def updatePriorities( self, indices, bellmanErrors ) :
        """Updates the priorities (node-values) of the sumtree with new bellman-errors

        Args:
            indices (np.ndarray)        : indices in the sumtree that have to be updated
            bellmanErrors (np.ndarray)  : bellman errors to be used for new priorities

        """
        # sanity-check: indices bath and bellmanErrors batch should be same length
        assert ( len( indices ) == len( bellmanErrors ) ), \
               'ERROR> indices and bellman-errors batch must have same size'

        # add the 'e' term to avoid 0s
        bellmanErrors += self._eps
        bellmanErrors = np.power( bellmanErrors, self._alpha )

        for i in range( len( indices ) ) : 
            # update each node
            self._sumtree.update( indices[i], bellmanErrors[i] )
            # update the max priority
            self._maxpriority = max( bellmanErrors[i], self._maxpriority )

    def __len__( self ) :
        return len( self._sumtree._data )

    @property
    def alpha( self ) :
        return self._alpha

    @property
    def beta( self ) :
        return self._beta    

    @property
    def eps( self ) :
        return self._eps