
from navigation.dqn.core import model

import numpy as np

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from IPython.core.debugger import set_trace

class NetworkPytorchGeneric( nn.Module ) :

    def __init__( self, inputShape, outputShape, layersDefs ) :
        super( NetworkPytorchGeneric, self ).__init__()

        self._inputShape = inputShape
        self._outputShape = outputShape

        self._layersDefs = layersDefs.copy()
        self._layers = []

        self._build()

    def _build( self ) :
        _currShape = self._inputShape

        for layerDef in self._layersDefs :
            _layer, _currShape = self._createLayer( layerDef, _currShape )
            self._layers.append( _layer )

    def _createLayer( self, layerDef, currShape ) :
        _layer = None
        _nextShape = None
        
        if layerDef['type'] == 'fc' :
            # sanity check (should have a rank-1 tensor)
            assert len( currShape ) == 1, 'ERROR> must pass rank-1 tensor as input to fc layer'
            # grab the number of hidden units for this fc layer
            _nunits = layerDef['units']
            # and just create the layer
            _layer = nn.Linear( currShape[0], _nunits )
            _nextShape = ( _nunits )

        elif layerDef['type'] == 'conv2d' :
            # sanity check (should have at least a rank-2 tensor)
            assert len( currShape ) >= 2, 'ERROR> '
            

        elif layerDef['type'] == 'flatten' :
            _layer = lambda x : x.view( -1 )
            _nextShape = ( x.numel() )

        return _layer, _nextShape

    def forward( self, x ) :
        for stage in self._layers :
            # pass through current layer
            x = stage['layer'](x)

        return x

    def clone( self, other, tau ) :
        for _thisParams, _otherParams in zip( self.parameters(), other.parameters() ) :
            _thisParams.data.copy_( ( 1.| - tau ) * _thisParams.data + ( tau ) * _otherParams.data )

class NetworkPytorchCustom( nn.Module ) :

    def __init__( self, inputShape, outputShape, layersDefs ) :
        super( NetworkPytorchCustom, self ).__init__()

        # banana-raycast has a 37-vector as an observation (rank-1 tensor)
        assert len( inputShape ) == 1, 'ERROR> input should be a rank-1 tensor'
        # and also has a discrete set of actions, with a 4-vector for its qvalues
        assert len( outputShape ) == 1, 'ERROR> output should be rank-1 tensor'

        self._inputShape = inputShape
        self._outputShape = outputShape

        # define layers for this network
        self.fc1 = nn.Linear( self._inputShape[0], 128 )
        self.fc2 = nn.Linear( 128, 64 )
        self.fc3 = nn.Linear( 64, 16 )
        self.fc4 = nn.Linear( 16, self._outputShape[0] )

##         # initialize the weights
##         _layers = [ self.fc1, self.fc2, self.fc3 ]
##         for layer in _layers :
##             torch.nn.init.xavier_normal_( layer.weight )
##             torch.nn.init.zeros_( layer.bias )

        self.h1 = None
        self.h2 = None
        self.h3 = None
        self.out = None

    def forward( self, X ) :
        self.h1 = F.relu( self.fc1( X ) )
        self.h2 = F.relu( self.fc2( self.h1 ) )
        self.h3 = F.relu( self.fc3( self.h2 ) )

        self.out = self.fc4( self.h3 )

        return self.out

    def clone( self, other, tau ) :
        for _thisParams, _otherParams in zip( self.parameters(), other.parameters() ) :
            _thisParams.data.copy_( ( 1. - tau ) * _thisParams.data + ( tau ) * _otherParams.data )

# Just to clarify :
# yhat -> extimates computed by the network
# y -> targets passed (like labels in supervised learning)

class DqnModelPytorch( model.IDqnModel ) :

    def __init__( self, name, modelConfig, trainable ) :
        super( DqnModelPytorch, self ).__init__( name, modelConfig, trainable )

    def build( self ) :
        # @TEST: creating a custom fc network
        self._nnetwork = NetworkPytorchCustom( self._inputShape,
                                               self._outputShape,
                                               self._layersDefs )

    def initialize( self, args ) :
        # grab current pytorch device
        self._device = args['device']
        # send network to device
        self._nnetwork.to( self._device )
        # create train functionality if necessary
        if self._trainable :
            # check whether or not using importance sampling
            if self._useImpSampling :
                self._lossFcn = lambda yhat, y, w : torch.mean( w * ( ( y - yhat ) ** 2 ) )
            else :
                self._lossFcn = nn.MSELoss()
            self._optimizer = optim.Adam( self._nnetwork.parameters(), lr = self._lr )

    def eval( self, state, inference = False ) :
        _xx = torch.from_numpy( state ).float().to( self._device )

        self._nnetwork.eval()
        with torch.no_grad() :
            _qvalues = self._nnetwork.forward( _xx ).cpu().data.numpy()
        self._nnetwork.train()

        return _qvalues

    def train( self, states, actions, targets, impSampWeights = None ) :
        if not self._trainable :
            print( 'WARNING> tried training a non-trainable model' )
            return None
        else :
            _aa = torch.from_numpy( actions ).unsqueeze( 1 ).to( self._device )
            _xx = torch.from_numpy( states ).float().to( self._device )
            _yy = torch.from_numpy( targets ).float().unsqueeze( 1 ).to( self._device )

            # reset the gradients buffer
            self._optimizer.zero_grad()
    
            # do forward pass to compute q-target predictions
            _yyhat = self._nnetwork.forward( _xx ).gather( 1, _aa )
    
            ## set_trace()
    
            # and compute loss and gradients
            if self._useImpSampling :
                assert ( impSampWeights is not None ), \
                       'ERROR> should have passed importance sampling weights'

                # convert importance sampling weights to tensor
                _ISWeights = torch.from_numpy( impSampWeights ).float().unsqueeze( 1 ).to( self._device )

                # make a custom mse loss weighted using the importance samples weights
                _loss = self._lossFcn( _yyhat, _yy, _ISWeights )
                _loss.backward()
            else :
                # do the normal loss computation and backward pass
                _loss = self._lossFcn( _yyhat, _yy )
                _loss.backward()

            # compute bellman errors (either for saving or for prioritized exp. replay)
            with torch.no_grad() :
                _absBellmanErrors = torch.abs( _yy - _yyhat ).cpu().numpy()
    
            # run optimizer to update the weights
            self._optimizer.step()
    
            # grab loss for later statistics
            self._losses.append( _loss.item() )

            if self._saveGradients :
                # grab gradients for later
                _params = list( self._nnetwork.parameters() )
                _gradients = [ _params[i].grad for i in range( len( _params ) ) ]
                self._gradients.append( _gradients )

            if self._saveBellmanErrors :
                self._bellmanErrors.append( _absBellmanErrors )

            return _absBellmanErrors

    def clone( self, other, tau = 1.0 ) :
        self._nnetwork.clone( other._nnetwork, tau )

    def save( self, filename ) :
        if self._nnetwork :
            torch.save( self._nnetwork.state_dict(), filename )

    def load( self, filename ) :
        if self._nnetwork :
            self._nnetwork.load_state_dict( torch.load( filename ) )

DEVICE = torch.device( 'cuda:0' ) if torch.cuda.is_available() else 'cpu'

def BackendInitializer() :
    # nothing to initialize (no sessions, global variables, etc.)
    return { 'device' : DEVICE }

DqnModelBuilder = lambda name, config, trainable : DqnModelPytorch( name, config, trainable )