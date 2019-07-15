
import abc
import numpy as np

from ccontrol.ddpg.utils import config

from IPython.core.debugger import set_trace


class DDPGModelBackbone( abc.ABC ) :
    r"""DDPG backbone architecture for all models (either actor|critic)

    Args:
        backboneConfig (config.DDPGModelBackboneConfig) : configuration of the backbone to be used

    """
    def __init__( self, backboneConfig, **kwargs ) :
        super( DDPGModelBackbone, self ).__init__()

        self._config = backboneConfig


    @abc.abstractmethod
    def forward( self, inputs ) :
        r"""Executes a forward pass over the backbone at a given state (or state-batch)

        Args:
            inputs (list): a list of np.ndarrays, which represents the inputs to the model

        """
        pass


    @abc.abstractmethod
    def copy( self, other, tau = 1.0 ) :
        r"""Copies softly (with polyak averaging) model weights from another model

        Args:
            other (DDPGModelBackbone)   : model from whom to copy the weights
            tau (float)                 : averaging factory (soft-update with polyak averaging)

        """
        pass


    @abc.abstractmethod
    def clone( self ) :
        r"""Creates an exact deep-replica of this model

        Returns:
            (DDPGModelBackbone) : replica of this model

        """
        pass


    @property
    def config( self ) :
        r"""Returns the configuration properties of the backbone

        """
        return self._config


class IDDPGActor( abc.ABC ) :
    r"""DDPG core actor-model class

    Abstract class that represents an actor for a DDPG-based agent, 
    composed of operations on top of a backbone architecture defined
    by the user as he requests (either defining a custom model or
    through layer definitions)

    Args:
        backbone (DDPGModelBackbone)    : model backbone architecture
        learningRate (float)            : learning rate used for the optimizer

    """
    def __init__( self, backbone, learningRate, **kwargs ) :
        super( IDDPGActor, self ).__init__()

        self._backbone = backbone
        self._learningRate = learningRate
        self._isTargetNetwork = False
        self._savedir = './results/session_default'


    def setAsTargetNetwork( self, isTarget ) :
        r"""Sets the target-mode of the network

        It hints the network to whether or not the network is a target network,
        which changes the behaviour a bit of this network in evaluation mode,
        namely removing the constraint of deactivating the batch-norm when 
        calling eval on a batch of states

        Args:
            isTarget (boolean) : target-mode to be used by this network 

        """
        self._isTargetNetwork = isTarget


    @abc.abstractmethod
    def eval( self, state ) :
        r"""Returns the action to take at a given state (batch of states)

        Args:
            state (np.ndarray) : state (batch of states) at which we want to act

        Returns:
            (np.ndarray) : action(s) to take at the given state(s)

        """
        pass

    @abc.abstractmethod
    def train( self, states, critic ) :
        r"""Takes a learning step to update the parameters of the actor

        This method must implement the update of the parameters of the actor
        by using the deterministic policy gradients theorem, which computes
        the gradients through the critic

        Args:
            states (np.ndarray)     : batch of states sampled from the replay buffer
            critic (IDDPGCritic)    : appropriate critic to be used to compute gradients from

        """
        pass


    @abc.abstractmethod
    def copy( self, other, tau = 1.0 ) :
        r"""Updates the parameters of the actor from another one using polyak averaging

        Args:
            other (IDDPGActor)  : actor from whom we want to copy the parameters
            tau (float)         : polyak averaging factor for soft-updates

        """
        pass


    @abc.abstractmethod
    def clone( self ) :
        r"""Creates a replica of this actor (usually for a target actor)

        Returns:
            (IDDPGActor) : a replica of this actor

        """
        pass


    def setSaveDir( self, savedir ) :
        r"""Sets the directory where to save actor model

        Args:
            savedir (string) : folder where to save the actor model

        """
        self._savedir = savedir


    @abc.abstractmethod
    def save( self ) :
        r"""Saves the actor model into disk

        """
        pass


    @abc.abstractmethod
    def load( self ) :
        r"""Loads the actor model from disk

        """
        pass


    @property
    def backbone( self ) :
        r"""Returns a reference to the backbone model

        """
        return self._backbone



class IDDPGCritic( abc.ABC ) :
    r"""DDPG core critic-model class

    Abstract class that represents a critic for a DDPG-based agent, 
    composed of operations on top of a backbone architecture defined
    by the user as he requests (either defining a custom model or
    through layer definitions)

    Args:
        backbone (DDPGModelBackbone)    : model backbone architecture
        learningRate (float)            : learning rate used for the optimizer

    Args:

    """
    def __init__( self, backbone, learningRate, **kwargs ) :
        super( IDDPGCritic, self ).__init__()

        self._backbone = backbone
        self._learningRate = learningRate
        self._isTargetNetwork = False
        self._savedir = './results/session_default'


    def setAsTargetNetwork( self, isTarget ) :
        r"""Sets the target-mode of the network

        It hints the network to whether or not the network is a target network,
        which changes the behaviour a bit of this network in evaluation mode,
        namely removing the constraint of deactivating the batch-norm when 
        calling eval on a batch of states

        Args:
            isTarget (boolean) : target-mode to be used by this network 

        """
        self._isTargetNetwork = isTarget


    @abc.abstractmethod
    def eval( self, state, action ) :
        r"""Returns the q-values Q(s,a) at the given state(s) and action(s)

        Args:
            state (np.ndarray)  : state (batch of states) at which we want to evaluate Q(s,a)
            action (np.ndarray) : action (batch of actions) at which we want to evaluate Q(s,a)

        Returns:
            (np.ndarray) : q-value (batch of Q(s,a))

        """
        pass


    @abc.abstractmethod
    def train( self, states, actions, qtargets ) :
        r"""Takes a learning step to update the parameters of the critic

        This method must implement the update of the parameters of the critic
        by using fitted Q-learning, using the given Q-targets as the true values
        of Q(s,a) at those states and actions given

        Args:
            states (np.ndarray)     : batch of states to compute Q(s,a)
            actions (np.ndarray)    : batch of actions to compute Q(s,a)
            qtargets (np.ndarray)   : batch of qtargets Qhat(s,a) = r + gamma * Q(s,u(s))

        """
        pass


    @abc.abstractmethod
    def copy( self, other, tau = 1.0 ) :
        r"""Updates the parameters of the critic from another one using polyak averaging

        Args:
            other (IDDPGCritic) : critic from whom we want to copy the parameters
            tau (float)         : polyak averaging factor for soft-updates

        """
        pass


    @abc.abstractmethod
    def clone( self ) :
        r"""Creates a replica of this actor (usually for a target actor)

        Returns:
            (IDDPGActor) : a replica of this actor

        """
        pass


    def setSaveDir( self, savedir ) :
        r"""Sets the directory where to save critic model

        Args:
            savedir (string) : folder where to save the critic model

        """
        self._savedir = savedir


    @abc.abstractmethod
    def save( self ) :
        r"""Saves the critic model into disk

        """
        pass


    @abc.abstractmethod
    def load( self ) :
        r"""Loads the critic model from disk

        """
        pass


    @property
    def backbone( self ) :
        return self._backbone