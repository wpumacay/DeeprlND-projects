
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
        for _localParams, _otherParams in zip( self.parameters(), other.parameters() ) :
            _localParams.data.copy_( tau * _localParams.data + ( 1.0 - tau ) * _otherParams.data )

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
        self.fc1 = nn.Linear( self._inputShape[0], 64 )
        self.fc2 = nn.Linear( 64, 64 )
        self.fc3 = nn.Linear( 64, self._outputShape[0] )

##         # initialize the weights
##         _layers = [ self.fc1, self.fc2, self.fc3 ]
##         for layer in _layers :
##             torch.nn.init.xavier_normal_( layer.weight )
##             torch.nn.init.zeros_( layer.bias )

        self.h1 = None
        self.h2 = None
        self.out = None

    def forward( self, X ) :
        self.h1 = F.relu( self.fc1( X ) )
        self.h2 = F.relu( self.fc2( self.h1 ) )

        self.out = self.fc3( self.h2 )

        return self.out

    def clone( self, other, tau ) :
        for _localParams, _otherParams in zip( self.parameters(), other.parameters() ) :
            _localParams.data.copy_( tau * _localParams.data + ( 1.0 - tau ) * _otherParams.data )

class DqnModelPytorch( model.IDqnModel ) :

    def __init__( self, name, modelConfig ) :
        super( DqnModelPytorch, self ).__init__( name, modelConfig )

    def build( self ) :
        self._device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )

        self._lossFcn = nn.MSELoss()
        self._losses = deque( maxlen = 100 )

        # @TEST: creating a custom fc network
        self._nnetwork = NetworkPytorchCustom( self._inputShape,
                                               self._outputShape,
                                               self._layersDefs )

        self._nnetwork.to( self._device )

        # @TODO: Add optimizer options to modelconfig
        self._optimizer = optim.Adam( self._nnetwork.parameters(), lr = self._lr )

    def initialize( self, args ) :
        # nothing to do here (pytorch is nice :D)
        pass

    def eval( self, state, inference = False ) :
        self._nnetwork.eval()

        _xx = torch.from_numpy( state ).float().to( self._device )

        return self._nnetwork.forward( _xx ).cpu().detach().numpy()

    def train( self, states, actions, targets ) :
        self._nnetwork.train()
        
        _aa = torch.from_numpy( actions ).unsqueeze( 1 ).to( self._device )
        _xx = torch.from_numpy( states ).float().to( self._device )
        _yy = torch.from_numpy( targets ).float().unsqueeze( 1 ).to( self._device )

        # reset the gradients buffer
        self._optimizer.zero_grad()

        # do forward pass to compute q-target predictions
        _yyhat = self._nnetwork.forward( _xx ).gather( 1, _aa )

        ## set_trace()

        # and compute loss and gradients
        _loss = self._lossFcn( _yyhat, _yy )
        _loss.backward()

        # run optimizer to update the weights
        self._optimizer.step()

        # grab loss for later statistics
        self._losses.append( _loss.item() )

    def clone( self, other, tau = 1.0 ) :
        self._nnetwork.clone( other._nnetwork, tau )

    def save( self, filename ) :
        if self._nnetwork :
            torch.save( self._nnetwork.state_dict(), filename )

    def load( self, filename ) :
        if self._nnetwork :
            self._nnetwork.load_state_dict( torch.load( filename ) )

def BackendInitializer() :
    # nothing to initialize (no sessions, global variables, etc.)
    return {}

DqnModelBuilder = lambda name, config : DqnModelPytorch( name, config )