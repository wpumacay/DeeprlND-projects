
import numpy as np

# debugging helpers
from IPython.core.debugger import set_trace

class IDqnModel( object ) :

    def __init__( self, name, modelConfig, trainable ) :
        super( IDqnModel, self ).__init__()

        # just an identifier
        self._name = name

        # whether this is an actor or target network
        self._trainable = trainable

        # save configuration data
        self._inputShape = modelConfig.inputShape
        self._outputShape = modelConfig.outputShape
        self._layersDefs = modelConfig.layers.copy()

        # save learning rate (copied from agent's configuration)
        self._lr = modelConfig._lr

        # whether or not use importance sampling on the (yy - yyhat)^2 of the loss
        self._useImpSampling = modelConfig._useImpSampling

        # losses storage (for learning stats)
        self._losses = []

        # gradients history
        self._gradients = []

        # bellman errors
        self._bellmanErrors = []

        # whether to save the gradients or not
        self._saveGradients = modelConfig.saveGradients

        # whether to save the bellman errors or not
        self._saveBellmanErrors = modelConfig.saveBellmanErrors

        self.build()

        # print configuration just in case
        self._printConfig()

    def build( self ) :
        raise NotImplementedError( 'IDqnModel::build> virtual method' )

    def eval( self, state ) :
        raise NotImplementedError( 'IDqnModel::eval> virtual method' )

    def train( self, states, actions, targets, impSampWeights = None ) :
        raise NotImplementedError( 'IDqnModel::train> virtual method' )

    def clone( self, other, tau = 1.0 ) :
        raise NotImplementedError( 'IDqnModel::clone> virtual method' )

    def save( self, filename ) :
        raise NotImplementedError( 'IDqnModel::save> virtual method' )

    def load( self, filename ) :
        raise NotImplementedError( 'IDqnModel::load> virtual method' )

    def initialize( self, args ) :
        raise NotImplementedError( 'IDqnModel::initialize> virtual method' )

    @property
    def losses( self ) :
        return self._losses

    @property
    def name( self ) :
        return self._name

    @property
    def trainable( self ) :
        return self._trainable

    @property
    def useImpSampling( self ) :
        return self._useImpSampling

    @property
    def gradients( self ) :
        return self._gradients

    @property
    def bellmanErrors( self ) :
        return self._bellmanErrors
    

    def _printConfig( self ) :
        # Each model could potentially override this with its own extra details
        print( '#############################################################' )
        print( '#                                                           #' )
        print( '#                 Model configuration                       #' )
        print( '#                                                           #' )
        print( '#############################################################' )

        print( 'model name          : ', self._name )
        print( 'input shape         : ', self._inputShape )
        print( 'output shape        : ', self._outputShape )
        print( 'learning rate       : ', self._lr )

        print( '#############################################################' )