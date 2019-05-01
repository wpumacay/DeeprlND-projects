
from navigation.dqn.core import model

import numpy as np

from collections import deque

import tensorflow as tf
from tensorflow import keras

from IPython.core.debugger import set_trace

def createNetworkCustom( inputShape, outputShape, layersDefs ) :
    # vector as an observation (rank-1 tensor)
    assert len( inputShape ) == 1, 'ERROR> input should be a rank-1 tensor'
    # and also discrete actions , with a 4-vector for its qvalues
    assert len( outputShape ) == 1, 'ERROR> output should be rank-1 tensor'

    # keep things simple (use keras for core model definition)
    _networkOps = keras.Sequential()

    # define initializers
    _kernelInitializer = keras.initializers.glorot_normal( seed = 0 )
    _biasInitializer = keras.initializers.Zeros()

    # add the layers for our test-case
    _networkOps.add( keras.layers.Dense( 128, activation = 'relu', input_shape = inputShape, kernel_initializer = _kernelInitializer, bias_initializer = _biasInitializer ) )
    _networkOps.add( keras.layers.Dense( 64, activation = 'relu', kernel_initializer = _kernelInitializer, bias_initializer = _biasInitializer ) )
    _networkOps.add( keras.layers.Dense( 16, activation = 'relu', kernel_initializer = _kernelInitializer, bias_initializer = _biasInitializer ) )
    _networkOps.add( keras.layers.Dense( outputShape[0], kernel_initializer = _kernelInitializer, bias_initializer = _biasInitializer ) )

    ## _networkOps.summary()

    return _networkOps

def createNetworkGeneric( inputShape, outputShape, layersDefs ) :
    pass

class DqnModelTensorflow( model.IDqnModel ) :

    def __init__( self, name, modelConfig, trainable ) :
        super( DqnModelTensorflow, self ).__init__( name, modelConfig, trainable )

    def build( self ) :
        # placeholder for state inputs
        self._tfStates = tf.placeholder( tf.float32, (None,) + self._inputShape )

        # create the nnetwork model architecture
        self._nnetwork = createNetworkCustom( self._inputShape,
                                              self._outputShape,
                                              self._layersDefs )
        
        # create the ops for evaluating the output of the model (Q(s,:))
        self._opQhat_s = self._nnetwork( self._tfStates )

        # if trainable (action network), create the full resources
        if self._trainable :
            # placeholders: actions, act-indices (gather), and computed q-targets
            self._tfActions             = tf.placeholder( tf.int32, (None,) )
            self._tfActionsIndices      = tf.placeholder( tf.int32, (None,) )
            self._tfQTargets            = tf.placeholder( tf.float32, (None,) )

            # @TODO|CHECK: Change the gather call by multiply + one-hot
            # create the ops for getting the Q(s,a) for each batch of (states) + (actions)
            # using tf.gather_nd, and expanding action indices with batch indices
            self._opActionsWithIndices = tf.stack( [self._tfActionsIndices, self._tfActions], axis = 1 )
            self._opQhat_sa = tf.gather_nd( self._opQhat_s, self._opActionsWithIndices )
    
            # create ops for the loss function
            self._opLoss = tf.losses.mean_squared_error( self._tfQTargets, self._opQhat_sa )
    
            # create ops for the loss and optimizer
            self._optimizer = tf.train.AdamOptimizer( learning_rate = self._lr )
            self._opOptim = self._optimizer.minimize( self._opLoss, var_list = self._nnetwork.trainable_weights )

            # op for gradients computation
            self._opComputeGradients = self._optimizer.compute_gradients( self._opLoss )

            # op for bellman errors computation
            self._opComputeBellmanErrors = tf.abs( self._tfQTargets - self._opQhat_sa )

        # tf.Session, passed by the backend-initializer
        self._sess = None

    def initialize( self, args ) :
        # grab session and initialize
        self._sess = args['session']

    def eval( self, state, inference = False ) :
        # unsqueeze if it's not a batch
        _batchStates = [state] if state.ndim == 1 else state
        _qvalues = self._sess.run( self._opQhat_s, feed_dict = { self._tfStates : _batchStates } )

        return _qvalues

    def train( self, states, actions, targets ) :
        if not self._trainable :
            print( 'WARNING> tried training a non-trainable model' )
        else :
            # for gather functionality
            _actionsIndices = np.arange( actions.shape[0] )
            # dictionary to feed to placeholders
            _feedDict = { self._tfStates : states,
                          self._tfActions : actions,
                          self._tfActionsIndices : _actionsIndices,
                          self._tfQTargets : targets }

            if self._saveGradients and self._saveBellmanErrors :
                # ops to run (as extra, need gradients and bellman erros)
                _ops = [ self._opOptim, self._opLoss, self._opComputeGradients, self._opComputeBellmanErrors ]

                # run the session
                _, _loss, _gradients, _bellmanErrors = self._sess.run( _ops, _feedDict )

                # grab the bounty
                set_trace()
                self._gradients.append( _gradients )
                self._bellmanErrors.append( _bellmanErrors )

            elif self._saveGradients :
                # ops to run (as extra, need gradients only)
                _ops = [ self._opOptim, self._opLoss, self._opComputeGradients ]

                # run the session
                _, _loss, _gradients = self._sess.run( _ops, _feedDict )

                # grab the bounty
                self._gradients.append( _gradients )

            elif self._saveBellmanErrors :
                # ops to run (as extra, need bellmanErrors only)
                _ops = [ self._opOptim, self._opLoss, self._opComputeBellmanErrors ]

                # run the session
                _, _loss, _bellmanErrors = self._sess.run( _ops, _feedDict )

                self._bellmanErrors.append( _bellmanErrors )

            else :
                # ops to run (no stats required here)
                _ops = [ self._opOptim, self._opLoss ]

                # run the session
                _, _loss = self._sess.run( _ops, _feedDict )
    
            # grab loss for later statistics
            self._losses.append( _loss )

    def clone( self, other, tau = 1.0 ) :
        _srcWeights = self._nnetwork.get_weights()
        _dstWeights = other._nnetwork.get_weights()

        ## set_trace()

        _weights = []
        for i in range( len( _srcWeights ) ) :
            _weights.append( ( 1. - tau ) * _srcWeights[i] + ( tau ) * _dstWeights[i] )

        self._nnetwork.set_weights( _weights )

    def save( self, filename ) :
        self._nnetwork.save_weights( filename )

    def load( self, filename ) :
        self._nnetwork.load_weights( filename )


def BackendInitializer() :
    session = tf.InteractiveSession()
    session.run( tf.global_variables_initializer() )

    return { 'session' : session }

DqnModelBuilder = lambda name, config, trainable : DqnModelTensorflow( name, config, trainable )