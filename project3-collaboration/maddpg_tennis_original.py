
import os
import sys
import gym
import copy
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
from torchsummary import summary

import torch
from torch import nn
from torch.functional import F
from torch import optim as opt

from collaboration.envs.mlagents import UnityEnvWrapper

from IPython.core.debugger import set_trace

# training parameters (not exposed through the command line)
NUM_AGENTS              = 2         # number of agents in the multiagent env. setup
GAMMA                   = 0.999     # discount factor applied to the rewards
LOG_WINDOW              = 100       # size of the smoothing window and logging window
TRAINING_EPISODES       = 2300      # number of training episodes
MAX_STEPS_IN_EPISODE    = 3000      # maximum number of steps in an episode
SEED                    = 0         # random seed to be used
EPSILON_SCHEDULE        = 'linear'  # type of shedule 
EPSILON_DECAY_FACTOR    = 0.999     # decay factor for e-greedy geometric schedule
EPSILON_DECAY_LINEAR    = 5e-6      # decay factor for e-greedy linear schedule
TRAINING_STARTING_STEP  = int(0)  # step index at which training should start

# configurable parameters through command line
TRAIN                   = True                      # whether or not to train our agent
TRAINING_SESSION_ID     = 'session_default'         # name of the training session
SESSION_FOLDER          = 'results/session_default' # folder where to save the results of the training session
REPLAY_BUFFER_SIZE      = 500000                    # size of the replay memory
BATCH_SIZE              = 256                       # batch size of data to grab for learning
LEARNING_RATE_ACTOR     = 0.001                     # learning rate used for actor network
LEARNING_RATE_CRITIC    = 0.0004                    # learning rate used for the critic network
TAU                     = 0.0008                    # soft update factor used for target-network updates
TRAIN_FREQUENCY_STEPS   = 1                         # learn every 10 steps (if there is data)
TRAIN_NUM_UPDATES       = 1                         # number of updates to do when doing a learning

DEVICE = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )

def lecunishUniformInitializer( layer ) :
    r"""Returns limits lecun-like initialization
    
    Args:
        layer 

    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt( fan_in / 2 )
    return ( -lim, lim )


class PiNetwork( nn.Module ) :
    r"""A simple deterministic policy network class to be used for the actor

    Args:
        observationShape (tuple): shape of the observations given to the network
        actionShape (tuple): shape of the actions to be computed by the network

    """
    def __init__( self, observationShape, actionShape, seed ) :
        super( PiNetwork, self ).__init__()

        self.seed = torch.manual_seed( seed )
        self.bn0 = nn.BatchNorm1d( observationShape[0] )
        self.fc1 = nn.Linear( observationShape[0], 256 )
        self.bn1 = nn.BatchNorm1d( 256 )
        self.fc2 = nn.Linear( 256, 128 )
        self.bn2 = nn.BatchNorm1d( 128 )
        self.fc3 = nn.Linear( 128, actionShape[0] )
        self._init()


    def _init( self ) :
        self.fc1.weight.data.uniform_( *lecunishUniformInitializer( self.fc1 ) )
        self.fc2.weight.data.uniform_( *lecunishUniformInitializer( self.fc2 ) )
        self.fc3.weight.data.uniform_( -3e-3, 3e-3 )


    def forward( self, observation ) :
        r"""Forward pass for this deterministic policy

        Args:
            observation (torch.tensor): observation used to decide the action

        """
        x = self.bn0( observation )
        x = F.relu( self.bn1( self.fc1( x ) ) )
        x = F.relu( self.bn2( self.fc2( x ) ) )
        x = F.tanh( self.fc3( x ) )

        return x


    def copy( self, other, tau = 1.0 ) :
        for paramsSelf, paramsOther in zip( self.parameters(), other.parameters() ) :
            paramsSelf.data.copy_( ( 1. - tau ) * paramsSelf.data + tau * paramsOther.data )


class Qnetwork( nn.Module ) :
    r"""A simple Q-network class to be used for the centralized critics

    Args:
        jointObservationShape (tuple): shape of the augmented state representation [o1,o2,...on]
        jointActionShape (tuple): shape of the augmented action representation [a1,a2,...,an]

    """
    def __init__( self, jointObservationShape, jointActionShape, seed ) :
        super( Qnetwork, self ).__init__()

        self.seed = torch.manual_seed( seed )

        self.bn0 = nn.BatchNorm1d( jointObservationShape[0] )
        self.fc1 = nn.Linear( jointObservationShape[0], 128 )
        self.fc2 = nn.Linear( 128 + jointActionShape[0], 128 )
        self.fc3 = nn.Linear( 128, 1 )
        self._init()


    def _init( self ) :
        self.fc1.weight.data.uniform_( *lecunishUniformInitializer( self.fc1 ) )
        self.fc2.weight.data.uniform_( *lecunishUniformInitializer( self.fc2 ) )
        self.fc3.weight.data.uniform_( -3e-3, 3e-3 )


    def forward( self, jointObservation, jointAction ) :
        r"""Forward pass for this critic at a given (x=[o1,...,an],aa=[a1...an]) pair

        Args:
            jointObservation (torch.tensor): augmented observation [o1,o2,...,on]
            jointAction (torch.tensor): augmented action [a1,a2,...,an]

        """
        _h = self.bn0( jointObservation )
        _h = F.relu( self.fc1( _h ) )
        _h = torch.cat( [_h, jointAction], dim = 1 )
        _h = F.relu( self.fc2( _h ) )
        _h = self.fc3( _h )

        return _h


    def copy( self, other, tau = 1.0 ) :
        for paramsSelf, paramsOther in zip( self.parameters(), other.parameters() ) :
            paramsSelf.data.copy_( ( 1. - tau ) * paramsSelf.data + tau * paramsOther.data )


class ReplayBuffer( object ) :
    r"""Replay buffer class used to train centralized critics.

    This replay buffer is the same as our old friend the replay-buffer from
    the vanilla dqn for a single agent, with some slight variations as the 
    tuples stored now consist in some cases in augmentations of the observations
    and action spaces:

    ([o1,...,on],[a1,...,an],[r1,...,rn],[o1',...,on'],[d1,...,dn])
          x                                    x'

    The usage depends on the network that will consume this data in its forward
    pass, which could be either a decentralized actor or a centralized critic.

    For a decentralized actor:

        u    ( oi ) requires the local observation for that actor
         theta-i

    For a centralized critic:

        Q     ( [o1,...,on], [a1,...,an] ) requires both the augmented observation
         phi-i   ----------  -----------   and the joint action from the actors
                     |            |
                     x        joint-action

    So, to make things simpler, as the environment is already returning packed
    numpy ndarrays with first dimension equal to the num-agents, we will store
    these as when sampling a minibatch we will actually returned an even more
    packed version, which would include a batch dimension on top of the over
    dimensions (n-agents,variable-shape), so we would have something like:
    
    e.g. storing:

        store( ( [obs1(33,),obs2(33,)], [a1(4,),a2(4,)], ... ) )
                 --------------------   ---------------
                    ndarray(2,33)         ndarray(2,4)

    e.g. sampling:
        batch -> ( batchObservations, batchActions, ... )
                   -----------------  ------------
                    tensor(128,2,33)   tensor(128,2,4)

    Args:
        bufferSize (int): max. number of experience tuples this buffer will hold
                          until it starts throwing away old experiences in a FIFO
                          way.
        numAgents (int): number of agents used during learning (for sanity-checks)

    """
    def __init__( self, bufferSize, numAgents ) :
        super( ReplayBuffer, self ).__init__()

        self._memory = deque( maxlen = bufferSize )
        self._numAgents = numAgents


    def store( self, transition ) :
        r"""Stores a transition tuple in memory

        The transition tuples to be stored must come in the form:

        ( [o1,...,on], [a1,...,an], [r1,...,rn], [o1',...,on'], [done1,...,donen] )

        Args:
            transition (tuple): a transition tuple to be stored in memory

        """
        # sanity-check: ensure first dimension of each transition component has the right size
        assert len( transition[0] ) == self._numAgents, 'ERROR> group observation size mismatch'
        assert len( transition[1] ) == self._numAgents, 'ERROR> group actions size mismatch'
        assert len( transition[2] ) == self._numAgents, 'ERROR> group rewards size mismatch'
        assert len( transition[3] ) == self._numAgents, 'ERROR> group next observations size mismatch'
        assert len( transition[4] ) == self._numAgents, 'ERROR> group dones size mismatch'

        self._memory.append( transition )


    def sample( self, batchSize ) :
        _batch = random.sample( self._memory, batchSize )

        _observations       = torch.tensor( [ _transition[0] for _transition in _batch ], dtype = torch.float ).to( DEVICE )
        _actions            = torch.tensor( [ _transition[1] for _transition in _batch ], dtype = torch.float ).to( DEVICE )
        _rewards            = torch.tensor( [ _transition[2] for _transition in _batch ], dtype = torch.float ).unsqueeze( 2 ).to( DEVICE )
        _observationsNext   = torch.tensor( [ _transition[3] for _transition in _batch ], dtype = torch.float ).to( DEVICE )
        _dones              = torch.tensor( [ _transition[4] for _transition in _batch ], dtype = torch.float ).unsqueeze( 2 ).to( DEVICE )

        return _observations, _actions, _rewards, _observationsNext, _dones


    def __len__( self ) :
        return len( self._memory )


class OUNoise( object ):
    """Ornstein-Uhlenbeck noise process
    
    Args:
        size (tuple): size of the noise to be generated
        seed (int): random seed for the rnd-generator
        mu (float): mu-param of the process
        theta (float): theta-param of the process
        sigma (float: sigma-param of the process
    """
    def __init__( self, size, seed, mu = 0., theta = 0.15, sigma = 0.2 ) :
        super( OUNoise, self ).__init__()

        self.mu = mu * np.ones( size )
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed( seed )
        self.reset()

    def reset( self ) :
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample( self ) :
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * ( self.mu - x ) + self.sigma * np.array( [ random.random() for i in range( len( x ) ) ] )
        self.state = x + dx

        return self.state


class NormalNoise( object ):
    """A simple normal-noise sampler
        
    Args:
        size (tuple): size of the noise to be generated
        seed (int): random seed for the rnd-generator
        sigma (float): standard deviation of the normal distribution
    """
    def __init__( self, size, seed, sigma = 0.2 ) :
        super( NormalNoise, self ).__init__()

        self._size = size
        self._seed = np.random.seed( seed )
        self._sigma = sigma

    def reset( self ) :
        pass

    def sample( self ) :
        _res =  self._sigma * np.random.randn( *self._size )
        return _res


def train( env, seed, num_episodes ) :
    ##------------- Create actor network (+its target counterpart)------------##
    actorsNetsLocal = [ PiNetwork( env.observation_space.shape,
                                   env.action_space.shape,
                                   seed ) for _ in range( NUM_AGENTS ) ]
    actorsNetsTarget = [ PiNetwork( env.observation_space.shape,
                                    env.action_space.shape,
                                    seed ) for _ in range( NUM_AGENTS ) ]
    for _netLocal, _netTarget in zip( actorsNetsLocal, actorsNetsTarget ) :
        _netTarget.copy( _netLocal )
        _netLocal.to( DEVICE )
        _netTarget.to( DEVICE )

    optimsActors = [ opt.Adam( _actorNet.parameters(), lr = LEARNING_RATE_ACTOR ) \
                        for _actorNet in actorsNetsLocal ]

    # print a brief summary of the network
    summary( actorsNetsLocal[0], env.observation_space.shape )
    print( actorsNetsLocal[0] )

    ##----------- Create critic network (+its target counterpart)-------------##
    criticsNetsLocal = [ Qnetwork( (NUM_AGENTS * env.observation_space.shape[0],),
                                   (NUM_AGENTS * env.action_space.shape[0],),
                                   seed ) for _ in range( NUM_AGENTS ) ]
    criticsNetsTarget = [ Qnetwork( (NUM_AGENTS * env.observation_space.shape[0],),
                                    (NUM_AGENTS * env.action_space.shape[0],),
                                    seed ) for _ in range( NUM_AGENTS ) ]
    for _netLocal, _netTarget in zip( criticsNetsLocal, criticsNetsTarget ) :
        _netTarget.copy( _netLocal )
        _netLocal.to( DEVICE )
        _netTarget.to( DEVICE )

    optimsCritics = [ opt.Adam( _criticNet.parameters(), lr = LEARNING_RATE_CRITIC ) \
                        for _criticNet in criticsNetsLocal ]

    # print a brief summary of the network
    summary( criticsNetsLocal[0], [(NUM_AGENTS * env.observation_space.shape[0],),
                                   (NUM_AGENTS * env.action_space.shape[0],)] )
    print( criticsNetsLocal[0] )
    ##------------------------------------------------------------------------##

    # Circular Replay buffer
    rbuffer = ReplayBuffer( REPLAY_BUFFER_SIZE, NUM_AGENTS )
    # Noise process
    ## noise = OUNoise( env.action_space.shape, seed )
    noise = NormalNoise( env.action_space.shape, seed )
    # Noise scaler factor (annealed with a schedule)
    epsilon = 1.0

    progressbar = tqdm( range( 1, num_episodes + 1 ), desc = 'Training>' )

    scoresAvgs = []
    scoresWindow = deque( maxlen = LOG_WINDOW )
    bestScore = -np.inf
    avgScore = -np.inf
    noiseNorm = -np.inf

    from tensorboardX import SummaryWriter
    writer = SummaryWriter( os.path.join( SESSION_FOLDER, 'tensorboard_summary' ) )
    istep = 0

    for iepisode in progressbar :

        noise.reset()
        _oo = env.reset()
        _scoreAgents = np.zeros( NUM_AGENTS )

        for i in range( MAX_STEPS_IN_EPISODE ) :
            # take full-random actions during these many steps
            if istep < TRAINING_STARTING_STEP :
                _aa = np.clip( np.random.randn( *((NUM_AGENTS,) + env.action_space.shape) ), -1., 1. )
            # take actions from exploratory policy
            else :
                # eval-mode (in case batchnorm is used)
                for _actorNet in actorsNetsLocal :
                    _actorNet.eval()

                # choose an action for each agent using its own actor network
                with torch.no_grad() :
                    _aa = []
                    for iactor, _actorNet in enumerate( actorsNetsLocal ) :
                        # evaluate action to take from each actor policy
                        _a = _actorNet( torch.from_numpy( _oo[iactor] ).unsqueeze( 0 ).float().to( DEVICE ) ).cpu().data.numpy().squeeze()
                        _aa.append( _a )
                    _aa = np.array( _aa )
                    # add some noise sampled from the noise process (each agent gets different sample)
                    _nn = np.array( [ epsilon * noise.sample() for _ in range( NUM_AGENTS ) ] ).reshape( _aa.shape )
                    # grab max noise norm for logging
                    _aa += _nn
                    # actions are speed-factors (range (-1,1)) in both x and y
                    _aa = np.clip( _aa, -1., 1. )
                    # save the max. norm of the noise so far
                    noiseNorm = np.max( np.linalg.norm( _nn, axis = 1 ) )

                # back to train-mode (in case batchnorm is used)
                for _actorNet in actorsNetsLocal :
                    _actorNet.train()

            # take action in the environment and grab bounty
            _oonext, _rr, _dd, _ = env.step( _aa )
            # store joint information (form (NAGENTS,) + MEASUREMENT-SHAPE)
            if i == MAX_STEPS_IN_EPISODE - 1 :
                rbuffer.store( ( _oo, _aa, _rr, _oonext, np.ones_like( _dd ) ) )
            else :
                rbuffer.store( ( _oo, _aa, _rr, _oonext, _dd ) )

            if len( rbuffer ) > BATCH_SIZE and istep % TRAIN_FREQUENCY_STEPS == 0 and \
               istep >= TRAINING_STARTING_STEP :
                for _ in range( TRAIN_NUM_UPDATES ) :
                    for iactor in range( NUM_AGENTS ) :
                        # grab a batch of data from the replay buffer
                        _observations, _actions, _rewards, _observationsNext, _dones = rbuffer.sample( BATCH_SIZE )
    
                        # compute joint observations and actions to be passed ...
                        # to the critic, which basically consists of keep the ...
                        # batch dimension and vectorize everything else into one ...
                        # single dimension [o1,...,on] and [a1,...,an]
                        _batchJointObservations = _observations.reshape( _observations.shape[0], -1 )
                        _batchJointObservationsNext = _observationsNext.reshape( _observationsNext.shape[0], -1 )
                        _batchJointActions = _actions.reshape( _actions.shape[0], -1 )
    
                        # compute the joint next actions required for the centralized ...
                        # critics q-target computation
                        with torch.no_grad() :
                            _batchJointActionsNext = torch.stack( [ actorsNetsTarget[iactorIndx]( _observationsNext[:,iactorIndx,:] )  \
                                                                    for iactorIndx in range( NUM_AGENTS ) ], dim = 1 )
                            _batchJointActionsNext = _batchJointActionsNext.reshape( _batchJointActionsNext.shape[0], -1 )

                        # extract local observations to be fed to the actors, ...
                        # as well as local rewards and dones to be used for local 
                        # q-targets computation using critics
                        _batchLocalObservations = _observations[:,iactor,:]
                        _batchLocalRewards = _rewards[:,iactor,:]
                        _batchLocalDones = _dones[:,iactor,:]

                        #---------------------- TRAIN CRITICS  --------------------#

                        # compute current q-values for the joint-actions taken ...
                        # at joint-observations using the critic, as explained ...
                        # in the MADDPG algorithm:
                        #
                        # Q(x,a1,a2,...,an) -> Q( [o1,o2,...,on], [a1,a2,...,an] )
                        #                       phi-i
                        _qvalues = criticsNetsLocal[iactor]( _batchJointObservations, _batchJointActions )
                        # compute target q-values using both decentralized ...
                        # target actor and centralized target critic for this ...
                        # current actor, as explained in the MADDPG algorithm:
                        #
                        # Q-targets  = r  + ( 1 - done ) * gamma * Q  ( [o1',...,on'], [a1',...,an'] )
                        #          i    i             i             phi-target-i
                        # 
                        # 
                        with torch.no_grad() :
                            _qvaluesTarget = _batchLocalRewards + ( 1. - _batchLocalDones ) \
                                                * GAMMA * criticsNetsTarget[iactor]( _batchJointObservationsNext, 
                                                                                     _batchJointActionsNext )
        
                        # compute loss for the critic
                        optimsCritics[iactor].zero_grad()
                        _lossCritic = F.mse_loss( _qvalues, _qvaluesTarget )
                        _lossCritic.backward()
                        torch.nn.utils.clip_grad_norm( criticsNetsLocal[iactor].parameters(), 1 )
                        optimsCritics[iactor].step()
    
                        #---------------------- TRAIN ACTORS  ---------------------#
    
                        # compute loss for the actor, from the objective to "maximize":
                        #
                        # dJ / dtheta = E [ dQ / du * du / dtheta ]
                        #
                        # where:
                        #   * theta: weights of the actor
                        #   * dQ / du : gradient of Q w.r.t. u (actions taken)
                        #   * du / dtheta : gradient of the Actor's weights
        
                        optimsActors[iactor].zero_grad()

                        # compute predicted actions for current local observations ...
                        # as we will need them for computing the gradients of the ...
                        # actor. Recall that these gradients depend on the gradients ...
                        # of its own related centralized critic, which need the joint ...
                        # actions to work. Keep with grads here as we have to build ...
                        # the computation graph with these operations
                        _batchJointActionsPred = torch.stack( [ actorsNetsLocal[indexActor]( _observations[:,indexActor,:] )  \
                                                                  if indexActor == iactor else _actions[:,indexActor,:] \
                                                                    for indexActor in range( NUM_AGENTS ) ], dim = 1 )
                        _batchJointActionsPred = _batchJointActionsPred.reshape( _batchJointActionsPred.shape[0], -1 )

                        # compose the critic over the actor outputs (sandwich), which effectively does g(f(x))
                        _lossActor = -criticsNetsLocal[iactor]( _batchJointObservations, _batchJointActionsPred ).mean()
                        _lossActor.backward()
                        optimsActors[iactor].step()
        
                        # update target networks
                        actorsNetsTarget[iactor].copy( actorsNetsLocal[iactor], TAU )
                        criticsNetsTarget[iactor].copy( criticsNetsLocal[iactor], TAU )
    
                    # update epsilon using schedule
                    if EPSILON_SCHEDULE == 'linear' :
                        epsilon = max( 0.1, epsilon - EPSILON_DECAY_LINEAR )
                    else :
                        epsilon = max( 0.1, epsilon * EPSILON_DECAY_FACTOR )

                for iactor in range( NUM_AGENTS ) :
                    torch.save( actorsNetsLocal[iactor].state_dict(), 
                                os.path.join( SESSION_FOLDER, 'maddpg_actor_reacher_' + str(iactor) + '.pth' ) )
                    torch.save( criticsNetsLocal[iactor].state_dict(), 
                                os.path.join( SESSION_FOLDER, 'maddpg_critic_reacher_' + str(iactor) + '.pth' ) )

            # book keeping for next iteration
            _oo = _oonext
            _scoreAgents += _rr
            istep += 1

            if _dd.any() :
                break

        # update some info for logging
        _score = np.max( _scoreAgents ) # score of the game is the max over both agents' scores
        bestScore = max( bestScore, _score ) # max game score so far
        scoresWindow.append( _score )

        if iepisode >= LOG_WINDOW :
            avgScore = np.mean( scoresWindow )
            scoresAvgs.append( avgScore )
            message = 'Training> best: %.2f - mean: %.2f - current: %.2f'
            progressbar.set_description( message % ( bestScore, avgScore, _score ) )
            progressbar.refresh()
        else :
            message = 'Training> best: %.2f - current : %.2f'
            progressbar.set_description( message % ( bestScore, _score ) )
            progressbar.refresh()

        writer.add_scalar( 'log1_score', _score, iepisode )
        writer.add_scalar( 'log2_avg_score', np.mean( scoresWindow ), iepisode )
        writer.add_scalar( 'log3_buffer_size', len( rbuffer ), iepisode )
        writer.add_scalar( 'log4_epsilon', epsilon, iepisode )
        if noiseNorm != -np.inf :
            writer.add_scalar( 'log5_noise_norm', noiseNorm, iepisode )

    for iactor in range( NUM_AGENTS ) :
        torch.save( actorsNetsLocal[iactor].state_dict(), 
                    os.path.join( SESSION_FOLDER, 'maddpg_actor_reacher_' + str(iactor) + '.pth' ) )
        torch.save( criticsNetsLocal[iactor].state_dict(), 
                    os.path.join( SESSION_FOLDER, 'maddpg_critic_reacher_' + str(iactor) + '.pth' ) )


def test( env, seed, num_episodes ) :
    actorsNets = [ PiNetwork( env.observation_space.shape,
                              env.action_space.shape,
                              seed ) for _ in range( NUM_AGENTS ) ]
    for iactor, _actorNet in enumerate( actorsNets ) :
        ## _actorNet.load_state_dict( torch.load( './results/maddpg_actor_reacher_' + str( iactor ) + '_' + TRAINING_SESSION_ID + '.pth' ) )
        _actorNet.load_state_dict( torch.load( os.path.join( SESSION_FOLDER, 'maddpg_actor_reacher_' + str(iactor) + '.pth' ), map_location='cpu' ) )
        _actorNet.eval()

    progressbar = tqdm( range( 1, num_episodes + 1 ), desc = 'Testing>' )

    for _ in progressbar :
        _done = False
        _oo = env.reset()
        _scoreAgents = np.zeros( NUM_AGENTS )

        while True :
            # compute actions for each actor
            _aa = []
            for iactor, _actorNet in enumerate( actorsNets ) :
                _a = _actorNet( torch.from_numpy( _oo[iactor] ).unsqueeze( 0 ).float() ).data.numpy().squeeze()
                _aa.append( _a )
            _aa = np.array( _aa )

            _oo, _rr, _dd, _ = env.step( _aa )
            env.render()

            _scoreAgents += _rr

            if _dd.any() :
                break

        _score = np.max( _scoreAgents ) # score of the game is the max over both agents' scores

        progressbar.set_description( 'Testing> score: %.2f' % ( _score ) )
        progressbar.refresh()


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument( 'mode', help='mode to run the script (train|test)', type=str, choices=['train','test'], default='train' )
    parser.add_argument( '--sessionId', help='unique identifier of this training run', type=str, default='session_default' )
    parser.add_argument( '--seed', help='random seed for the rnd-generators', type=int, default=SEED )
    parser.add_argument( '--hp_replay_buffer_size', help='size of the replay buffer to be used', type=int, default=REPLAY_BUFFER_SIZE )
    parser.add_argument( '--hp_batch_size', help='batch size for updates on both the actor and critic', type=int, default=BATCH_SIZE )
    parser.add_argument( '--hp_lrate_actor', help='learning rate used for the actor', type=float, default=LEARNING_RATE_ACTOR )
    parser.add_argument( '--hp_lrate_critic', help='learning rate used for the critic', type=float, default=LEARNING_RATE_CRITIC )
    parser.add_argument( '--hp_tau', help='soft update parameter (polyak averaging)', type=float, default=TAU )
    parser.add_argument( '--hp_train_update_freq', help='how often to do a learning step', type=int, default=TRAIN_FREQUENCY_STEPS )
    parser.add_argument( '--hp_train_num_updates', help='how many updates to do per learning step', type=int, default=TRAIN_NUM_UPDATES )

    args = parser.parse_args()

    SEED                    = args.seed
    TRAIN                   = ( args.mode.lower() == 'train' )
    TRAINING_SESSION_ID     = args.sessionId
    SESSION_FOLDER          = os.path.join( './results', TRAINING_SESSION_ID )
    REPLAY_BUFFER_SIZE      = args.hp_replay_buffer_size
    BATCH_SIZE              = args.hp_batch_size
    LEARNING_RATE_ACTOR     = args.hp_lrate_actor
    LEARNING_RATE_CRITIC    = args.hp_lrate_critic
    TAU                     = args.hp_tau
    TRAIN_FREQUENCY_STEPS   = args.hp_train_update_freq
    TRAIN_NUM_UPDATES       = args.hp_train_num_updates

    if not os.path.exists( SESSION_FOLDER ) :
        os.makedirs( SESSION_FOLDER )

    # in case the results directory for this session does not exist, create a new one
    
    if not os.path.exists( SESSION_FOLDER ) :
        os.makedirs( SESSION_FOLDER )

    print( '#############################################################' )
    print( '#                                                           #' )
    print( '#            Environment and agent setup                    #' )
    print( '#                                                           #' )
    print( '#############################################################' )
    print( 'Mode                    : ', args.mode.lower() )
    print( 'SessionId               : ', args.sessionId )
    print( 'Seed                    : ', SEED )
    print( 'Replay buffer size      : ', args.hp_replay_buffer_size )
    print( 'Batch size              : ', args.hp_batch_size )
    print( 'Learning-rate actor     : ', args.hp_lrate_actor )
    print( 'Learning-rate critic    : ', args.hp_lrate_critic )
    print( 'Tau                     : ', args.hp_tau )
    print( 'Train update freq       : ', args.hp_train_update_freq )
    print( 'Train num updates       : ', args.hp_train_num_updates )
    print( '#############################################################' )

    # create the environment
    executableFullPath = os.path.join( os.getcwd(), './executables/Tennis_Linux/Tennis.x86_64' )
    env = UnityEnvWrapper( executableFullPath,
                           numAgents = 2, 
                           mode = 'training' if TRAIN else 'testing', 
                           workerID = 100, 
                           seed = SEED )

    env.seed( SEED )
    random.seed( SEED )
    np.random.seed( SEED )
    torch.manual_seed( SEED )

    if TRAIN :
        train( env, SEED, TRAINING_EPISODES )
    else :
        test( env, SEED, 10 )