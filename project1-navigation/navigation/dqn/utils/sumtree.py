
# Source 1> Adapted (just added some comments and checks) from :
# https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
#
# Source 2> Adapted also from the code by Morvan Zhou from :
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
# which is a cleaner version of AI-blog's implenentation
#
# Give it a read to his blog on the PER implementation:
# https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
#
# This implementation follows the SumTree variant discussed in appendix B.2.1
# from the Priritized Experience Replay paper (https://arxiv.org/pdf/1511.05952.pdf)
#
# The proportianl prioritization variant uses a SumTree (the code below) to
# do efficient inserts, updates and samples

import numpy as np

class SumTree( object ) :

    def __init__( self, bufferSize ) :
        super( SumTree, self ).__init__()

        # capacity of the tree
        self._bufferSize = bufferSize

        # data used for the tree representation in ...
        # array-like form, kind of similar to a heap
        self._tree = np.zeros( 2 * bufferSize - 1 )

        # actual buffer of the data (why not list?, perhaps it is just an np array of pointers)
        self._data = np.zeros( bufferSize, dtype = object )

        # position pointer in the buffer
        self._pos = 0

    def add( self, data, nodeval ) :
        """Adds a node with internal data 'data' and node value 'nodeval'
    
        Args:
            data (object)   : the actual data stored in the node
            nodeval         : the value of this node (e.g. priority)
        """

        # index in the tree buffer
        _indx = self._pos + self._bufferSize - 1

        # store the data in the appropriate position in the data buffer
        self._data[self._pos] = data

        # update the node value in the tree buffer
        self.update( _indx, nodeval )

        # move the pointer to the next appropriate position
        self._pos = ( self._pos + 1 ) % self._bufferSize


    def update( self, index, nodeval ) :
        """Updates the tree, starting at the given index and then upwards

        Args:
            index (int)     : index of the node in tree to start the update
            nodeval (float) : value of the node in tree to start the update

        """

        # cache the change for later propagation
        _change = nodeval - self._tree[index]

        # do the actual update of the value of this node in the tree
        self._tree[index] = nodeval

        # Option 1: recursively propagate the change down the tree (like in Source 1)
        self._propagate( index, _change )

    def getNode( self, value ) :
        """Gets a node given a certain value that it should have (close to it)
        
        Args:
            value (float) : a value used to locate a node close to it

        """
        
        # search recursively in the tree
        _indx = self._retrieve( 0, value )

        # for this index, grab the corresponding index in the data buffer
        _dataIndx = _indx - self._bufferSize + 1

        # return the index in the tree, the node-value and the actual data
        return ( _indx, self._tree[_indx], self._data[_dataIndx] )

    def total( self ) :
        """Returns the total cumsum of node-values (stored at the root)

        """
        return self._tree[0]

    def maxNodeValue( self ) :
        """Returns the max node-value

        """
        return np.max( self._tree[-self._bufferSize:] )

    def minNodeValue( self ) :
        """Returns the min node-value

        """
        return np.min( self._tree[-self._bufferSize:] )

    def _propagate( self, index, change ) :
        """Recursively update changes in a node of the sumtree (upwards)

        Args:
            index   (int)   : index of the node in the tree to change its value
            change (float)  : change in the value of the node in the tree

        """
        _parentIndx = ( index - 1 ) // 2

        self._tree[_parentIndx] += change

        if _parentIndx != 0 :
            self._propagate( _parentIndx, change )

    def _retrieve( self, index, value ) :
        """Searchs recursively (starting from 'index') for a node with value close to 'value'

        Args:
            index (int)     : index in the tree where to start the search
            value (float)   : value (or close to it) of the resulting node we want

        """

        _leftIndx = 2 * index + 1
        _rightIndx = _leftIndx + 1

        if _leftIndx >= len( self._tree ) :
            return index

        if value <= self._tree[_leftIndx] :
            return self._retrieve( _leftIndx, value )
        else :
            return self._retrieve( _rightIndx, value - self._tree[_leftIndx] )