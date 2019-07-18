
import numpy as np

from ccontrol.ddpg.utils import replaybuffer
from ccontrol.ddpg.utils import noise
from ccontrol.ddpg.utils import config

from ccontrol.ddpg.core.model import IDDPGActor
from ccontrol.ddpg.core.model import IDDPGCritic

from IPython.core.debugger import set_trace


class DDPGAgent( object ) : 
    r"""DDPG core agent class

    Implements a DDPG agent as in the paper 'Continuous control with deep reinforcement learning'
    by Lillicrap et. al., which implements an approximate DQN for continuous action
    spaces by making an 'actor' that approximates the maximizer used for a 'critic'. Both
    are trained using off-policy data from an exploratory policy based on the actor.

    Args:
        config (config.DDPGAgentConfig) : configuration object for this agent
        actorModel (model.IDDPGActor)   : model used for the actor of the ddpg-agent
        criticModel (model.IDDPGCritic) : model used for the critic of the ddpg agent

    """
    def __init__( self, agentConfig, actorModel, criticModel ) :
        super( DDPGAgent, self ).__init__()

        # keep the references to both actor and critic
        self._actor = actorModel
        self._critic = criticModel
        # and create some copies for the target networks
        self._actorTarget = self._actor.clone()
        self._criticTarget = self._critic.clone()
        # hint these target networks that they are actual target networks,
        # which is kind of a HACK to ensure batchnorm is called during eval
        # when using these networks to compute some required tensors
        self._actorTarget.setAsTargetNetwork( True )
        self._criticTarget.setAsTargetNetwork( True )

        # keep the reference to the configuration object
        self._config = agentConfig

        # directory where to save both actor and critic models
        self._savedir = './results/session_default'

        # step counter
        self._istep = 0

        # replay buffer to be used
        self._rbuffer = replaybuffer.DDPGReplayBuffer( self._config.replayBufferSize )

        # noise generator to be used
        if self._config.noiseType == 'ounoise' :
            self._noiseProcess = noise.OUNoise( self._config.actionsShape,
                                                self._config.noiseOUMu,
                                                self._config.noiseOUTheta,
                                                self._config.noiseOUSigma,
                                                self._config.seed )
        else :
            self._noiseProcess = noise.Normal( self._config.actionsShape,
                                               self._config.noiseNormalStddev,
                                               self._config.seed )

        # epsilon factor used to adjust exploration noise
        self._epsilon = 1.0

        ## # action scale factor
        ## self._actionScaler = 0.8

        # mode of the agent, either train or test
        self._mode = 'train'


    def setMode( self, mode ) :
        r"""Set the mode the agent will work with, either (train|test)

        Args:
            mode (str) : mode in which the agent should be working

        """
        self._mode = mode


    def act( self, state ) :
        r"""Returns an action to take in state(s) 'state'

        Args:
            state (np.ndarray) : state (batch of states) to be evaluated by the actor

        Returns:
            (np.ndarray) : action (batch of actions) to be taken at that situation

        """
        assert state.ndim > 1, 'ERROR> state should have a batch dimension (even if it is a single state)'

        _action = self._actor.eval( state )
        # during training add some noise (per action in the batch, to incentivize more exploration)
        if self._mode == 'train' :
            _noise = np.array( [ self._epsilon * self._noiseProcess.sample() \
                                    for _ in range( len( state ) ) ] ).reshape( _action.shape )
            _action += _noise
            ## _action = np.clip( _action, -self._actionScaler, self._actionScaler )
            _action = np.clip( _action, -1., 1. )

        return _action


    def update( self, transitions ) :
        r"""Updates the internals of the agent given some new batch of transitions

        Args:
            transitions (list) : a batch of transitions of the form (s,a,r,s',done)

        """
        for transition in transitions :
            self._rbuffer.store( transition )

        if self._istep >= self._config.trainingStartingStep and \
           self._istep % self._config.trainFrequencySteps == 0 and \
           len( self._rbuffer ) > self._config.batchSize :
            # do the required number of learning steps
            for _ in range( self._config.trainNumLearningSteps ) :
                self._learn()

            # save the current model
            self._actor.save()
            self._critic.save()

        self._istep += 1

        # update epsilon using the required schedule
        if self._config.epsilonSchedule == 'linear' :
            self._epsilon = max( 0.025, self._epsilon - self._config.epsilonFactorLinear )
            ## self._actionScaler = min( 1.0, self._actionScaler + self._config.epsilonFactorLinear )
        else :
            self._epsilon = max( 0.025, self._epsilon * self._config.epsilonFactorGeom )
            ## self._actionScaler = min( 1.0, self._actionScaler * self._config.epsilonFactorGeom )

    def _learn( self ) :
        r"""Takes a learning step on a batch from the replay buffer

        """
        # 0) grab some experience tuples from the replay buffer
        _states, _actions, _rewards, _statesNext, _dones = self._rbuffer.sample( self._config.batchSize )

        # 1) train the critic (fit q-values to q-targets)
        #
        #   minimize mse-loss of current q-value estimations and the ...
        #   corresponding TD(0)-estimates used as "true" q-values
        #
        #   * pi  -> actor parametrized by weights "theta"
        #       theta
        #
        #   * pi  -> actor target parametrized by weights "theta-t"
        #       theta-t
        #
        #   * Q   -> critic parametrized by weights "phi"
        #      phi
        #
        #   * Q   -> critic-target parametrized by weights "phi-t"
        #      phi-t
        #                           __                 ___                          2
        #   phi := phi + lrCritic * \/    ( 1 / |B| )  \    || Qhat(s,a) - Q(s,a) ||
        #                             phi              /__
        #                                         (s,a,r,s',d) in B
        #
        #   where:
        #      * Q(s,a) = Q (s,a) -> q-values from the critic
        #                phi
        #
        #      * a' = pi(s') -> max. actions from the target actor
        #               theta-t
        #
        #      * Qhat(s,a) = r + (1 - d) * gamma * Q (s',a') -> q-targets from the target critic
        #                                           phi-t
        #
        # so: compute q-target, and used them as true labels in a supervised-ish learning process
        #
        _actionsNext = self._actorTarget.eval( _statesNext )
        _qtargets = _rewards + ( 1. - _dones ) * self._config.gamma * self._criticTarget.eval( _statesNext, _actionsNext )
        self._critic.train( _states, _actions, _qtargets )

        # 2) train the actor (its gradient comes from the critic in a pathwise way)
        #
        #   compute gradients for the actor from gradients of the critic ...
        #   based on the deterministic policy gradients theorem:
        #
        #   dJ / d = E [ dQ / du * du / dtheta ]
        #
        #   __            __  
        #   \/  J   = E [ \/     Q( s, a ) |  ]
        #     theta        theta  phi      |s=st, a=pi(st)
        #                                             theta
        #
        #   which can be further reduced to :
        #
        #   __            __                            __
        #   \/  J   = E [ \/  Q( s, a ) |               \/  pi(s) |  ]
        #     theta        a   phi      |s=st, a=pi(st)   theta   |s=st
        #                                         theta
        #
        #   so: compute gradients of the actor from one of the expression above:
        #
        #    * for pytorch: just do composition Q(s,pi(s)), like f(g(x)), ...
        #                   and let pytorch's autograd do the job of ...
        #                   computing df/dg * dg/dx
        #
        #    * for tensorflow: compute gradients from both and combine them ...
        #                      using tf ops and tf.gradients
        #
        self._actor.train( _states, self._critic )
        
        # 3) apply soft-updates using polyak averaging
        self._actorTarget.copy( self._actor, self._config.tau )
        self._criticTarget.copy( self._critic, self._config.tau )


    def setSaveDir( self, savedir ) :
        r"""Sets the directory where to save actor and critic weights

        Args:
            savedir (string) : folder where to save both actor and critic models

        """
        self._savedir = savedir
        self._actor.setSaveDir( savedir )
        self._critic.setSaveDir( savedir )


    def load( self, savedir = None ) :
        r"""Loads the actor and critic models from a save directory

        Args:
            savedir (str?) : directory where to load the models from. If None
                             is given, then used own savedir

        """
        if savedir :
            self._actor.setSaveDir( savedir )
            self._critic.setSaveDir( savedir )

        self._actor.load()
        self._critic.load()

    @property
    def actor( self ) :
        return self._actor


    @property
    def critic( self ) :
        return self._critic
    

    @property
    def replayBuffer( self ) :
        return self._rbuffer


    @property
    def epsilon( self ) :
        return self._epsilon
    

    @property
    def noiseProcess( self ) :
        return self._noiseProcess