
import abc
import numpy as np

from ccontrol.ddpg.core import model

import torch
from torch import nn
from torch import optim as opt
from torch.functional import F

from IPython.core.debugger import set_trace

DEFAULT_DEVICE = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )

##----------------------------------------------------------------------------##
##                         Backbones definitions                              ##
##----------------------------------------------------------------------------##

def lecunishUniformInitializer( layer ) :
    _fanIn = layer.weight.data.size()[0]
    _limit = np.sqrt( 2. / _fanIn )
    
    return ( -_limit, _limit )


class DDPGModelBackbonePytorch( model.DDPGModelBackbone, nn.Module ) :


    def __init__( self, backboneConfig, **kwargs ) :
        super( DDPGModelBackbonePytorch, self ).__init__( backboneConfig, **kwargs )

        self._seed = torch.manual_seed( self._config.seed )
        self._device = kwargs['device'] if 'device' in kwargs else DEFAULT_DEVICE

    @abc.abstractmethod
    def _resetParameters( self ) :
        r"""Resets the parameters of the network by using the appropriate initializers

        """
        pass


    def copy( self, other, tau = 1.0 ) :
        r"""Copies softly (with polyak averaging) model weights from another model

        Args:
            other (DDPGModelBackbone)   : model from whom to copy the weights
            tau (float)                 : averaging factory (soft-update with polyak averaging)

        """
        for paramsSelf, paramsOther in zip( self.parameters(), other.parameters() ) :
            paramsSelf.data.copy_( ( 1. - tau ) * paramsSelf.data + tau * paramsOther.data )


    def clone( self ) :
        r"""Creates an exact deep-replica of this model

        Returns:
            (DDPGModelBackbone) : replica of this model

        """
        _replica = self.__class__( self._config, device = self._device )
        _replica.to( self._device )
        _replica.copy( self )

        return _replica


class DDPGMlpModelBackboneActor( DDPGModelBackbonePytorch ) :

    def __init__( self, backboneConfig, **kwargs ) :
        super( DDPGMlpModelBackboneActor, self ).__init__( backboneConfig, **kwargs )

        self._fc1 = nn.Linear( self._config.inputShape[0], 256 )
        self._fc2 = nn.Linear( 256, 128 )
        self._fc3 = nn.Linear( 128, self._config.outputShape[0] )
        
        if self._config.useBatchnorm :
            self._bn0 = nn.BatchNorm1d( self._config.inputShape[0] )
            self._bn1 = nn.BatchNorm1d( 256 )
            self._bn2 = nn.BatchNorm1d( 128 )

        # initialize the parameters of the backbone
        self._resetParameters()


    def _resetParameters( self ) :
        self._fc1.weight.data.uniform_( *lecunishUniformInitializer( self._fc1 ) )
        self._fc2.weight.data.uniform_( *lecunishUniformInitializer( self._fc2 ) )
        self._fc3.weight.data.uniform_( -3e-3, 3e-3 )


    def forward( self, inputs ) :
        # sanity check: actor network is deterministic, and receives only states as input
        assert len( inputs ) == 1, 'ERROR> this network expects only one input (states)'

        # grab the actual input to the model (states)
        _states = inputs[0]

        if self._config.useBatchnorm :
            x = self._bn0( _states )
            x = F.relu( self._bn1( self._fc1( x ) ) )
            x = F.relu( self._bn2( self._fc2( x ) ) )
            x = F.tanh( self._fc3( x ) )
        else :
            x = F.relu( self._fc1( _states ) )
            x = F.relu( self._fc2( x ) )
            x = F.tanh( self._fc3( x ) )

        return x


class DDPGMlpModelBackboneCritic( DDPGModelBackbonePytorch ) :

    def __init__( self, backboneConfig, **kwargs ) :
        super( DDPGMlpModelBackboneCritic, self ).__init__( backboneConfig, **kwargs )

        # sanity check: ensure sizes are exactly what we expect for this case
        assert self._config.outputShape[0] == 1, \
            'ERROR> Critic model should only output 1 value for Q(s,a)'

        self._fc1 = nn.Linear( self._config.inputShape[0], 128 )
        self._fc2 = nn.Linear( 128 + self._config.actionsShape[0], 128 )
        self._fc3 = nn.Linear( 128, self._config.outputShape[0] )

        if self._config.useBatchnorm :
            self._bn0 = nn.BatchNorm1d( self._config.inputShape[0] )

        # initialize the parameters of the backbone
        self._resetParameters()


    def _resetParameters( self ) :
        self._fc1.weight.data.uniform_( *lecunishUniformInitializer( self._fc1 ) )
        self._fc2.weight.data.uniform_( *lecunishUniformInitializer( self._fc2 ) )
        self._fc3.weight.data.uniform_( -3e-3, 3e-3 )


    def forward( self, inputs ) :
        # sanity check: critic network expects both state and action batches as inputs
        assert len( inputs ) == 2, 'ERROR> this network expects two inputs (states,actions)'

        # grab the actual inputs to the model (states and actions)
        _states = inputs[0]
        _actions = inputs[1]

        if self._config.useBatchnorm :
            x = self._bn0( _states )
            x = F.relu( self._fc1( x ) )
            x = torch.cat( [x, _actions], dim = 1 )
            x = F.relu( self._fc2( x ) )
            x = self._fc3( x )
        else :
            x = F.relu( self._fc1( _states ) )
            x = torch.cat( [x, _actions], dim = 1 )
            x = F.relu( self._fc2( x ) )
            x = self._fc3( x )

        return x

##----------------------------------------------------------------------------##

##----------------------------------------------------------------------------##
##                     DDPG Actor model definition                            ##
##----------------------------------------------------------------------------##

class DDPGActor( model.IDDPGActor ) :

    def __init__( self, backbone, learningRate, **kwargs ) :
        super( DDPGActor, self ).__init__( backbone, learningRate, **kwargs )

        self._device = kwargs['device'] if 'device' in kwargs else DEFAULT_DEVICE
        # send the backbone model to the appropriate device
        self._backbone.to( self._device )

        self._optimizer = opt.Adam( self._backbone.parameters(), self._learningRate )


    def eval( self, state ) :
        # transform to torch tensors
        state = torch.from_numpy( state ).float().to( self._device )

        if not self._isTargetNetwork :
            # set in evaluation mode, as we might be using batch-norm
            self._backbone.eval()

        # do not compute gradients for the actor just yet
        with torch.no_grad() :
            _action = self._backbone( [state] )

        if not self._isTargetNetwork :
            # get back to training mode, as we might be using batch-norm
            self._backbone.train()

        return _action.cpu().data.numpy()


    def train( self, states, critic ) :
        # transform to torch tensors
        states = torch.from_numpy( states ).float().to( self._device )

        self._optimizer.zero_grad()
        # compute actions taken in these states by the actor
        _actionsPred = self._backbone( [states] )
        # compose the critic over the actor outputs (sandwich), which effectively does g(f(x))
        _lossActor = -critic( states, _actionsPred ).mean()
        _lossActor.backward()
        # take a step with the optimizer
        self._optimizer.step()


    def copy( self, other, tau = 1.0 ) :
        self._backbone.copy( other.backbone, tau )


    def clone( self ) :
        _replica = self.__class__( self._backbone.clone(),
                                   self._learningRate,
                                   device = self._device )

        return _replica


    def save( self ) :
        torch.save( self._backbone.state_dict(), 'checkpoint_actor.pth' )


    def load( self ) :
        self._backbone.load_state_dict( torch.load( 'checkpoint_actor.pth' ) )


    def __call__( self, states ) :
        return self._backbone( [states] )

##----------------------------------------------------------------------------##

##----------------------------------------------------------------------------##
##                     DDPG Critic model definition                           ##
##----------------------------------------------------------------------------##

class DDPGCritic( model.IDDPGCritic ) :

    def __init__( self, backbone, learningRate, **kwargs ) :
        super( DDPGCritic, self ).__init__( backbone, learningRate, **kwargs )

        self._device = kwargs['device'] if 'device' in kwargs else DEFAULT_DEVICE
        # send the backbone model to the appropriate device
        self._backbone.to( self._device )

        self._optimizer = opt.Adam( self._backbone.parameters(), self._learningRate )


    def eval( self, state, action ) :
        # transform to torch tensors
        state = torch.from_numpy( state ).float().to( self._device )
        action = torch.from_numpy( action ).float().to( self._device )

        if not self._isTargetNetwork :
            # set in evaluation mode, as we might be using batch-norm
            self._backbone.eval()

        # do not compute gradients for the critic in this stage
        with torch.no_grad() :
            _qvalue = self._backbone( [state, action] )

        if not self._isTargetNetwork :
            # get back to training mode, as we might be using batch-norm
            self._backbone.train()

        return _qvalue.cpu().data.numpy()


    def train( self, states, actions, qtargets ) :
        # transform to torch tensors
        states = torch.from_numpy( states ).float().to( self._device )
        actions = torch.from_numpy( actions ).float().to( self._device )
        qtargets = torch.from_numpy( qtargets ).float().to( self._device )

        # compute q-values for Q(s,a), where s,a come from the given ...
        # states and actions batches passed along the q-targets
        _qvalues = self._backbone( [states, actions] )

        # compute loss for the critic
        self._optimizer.zero_grad()
        _lossCritic = F.mse_loss( _qvalues, qtargets )
        _lossCritic.backward()
        if self._backbone.config.clipGradients :
            nn.utils.clip_grad_norm( self._backbone.parameters(), 
                                     self._backbone.config.gradientsClipNorm )
        # take a step with the optimizer
        self._optimizer.step()


    def copy( self, other, tau = 1.0 ) :
        self._backbone.copy( other.backbone, tau )


    def clone( self ) :
        _replica = self.__class__( self._backbone.clone(),
                                   self._learningRate,
                                   device = self._device )

        return _replica


    def save( self ) :
        torch.save( self._backbone.state_dict(), 'checkpoint_critic.pth' )


    def load( self ) :
        self._backbone.load_state_dict( torch.load( 'checkpoint_critic.pth' ) )


    def __call__( self, states, actions ) :
        return self._backbone( [states, actions] )

##----------------------------------------------------------------------------##