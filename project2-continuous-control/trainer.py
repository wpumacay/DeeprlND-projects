
import os
import gym
import gin
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import deque
from tensorboardX import SummaryWriter

from ccontrol.envs.mlagents import UnityEnvWrapper

from ccontrol.ddpg.core.agent import DDPGAgent

from ccontrol.ddpg.utils.config import DDPGAgentConfig
from ccontrol.ddpg.utils.config import DDPGModelBackboneConfig

# use pytorch backend (currently only one supported for now)
from ccontrol.ddpg.models.pytorch import DDPGMlpModelBackboneActor
from ccontrol.ddpg.models.pytorch import DDPGMlpModelBackboneCritic
from ccontrol.ddpg.models.pytorch import DDPGActor
from ccontrol.ddpg.models.pytorch import DDPGCritic

from IPython.core.debugger import set_trace

TRAIN                   = True      # whether or not to train our agent
LOG_WINDOW              = 100       # size of the smoothing window and logging window
TRAINING_EPISODES       = 2000      # number of training episodes
MAX_STEPS_IN_EPISODE    = 3000      # maximum number of steps in an episode
TRAINING_SESSION_ID     = 'sess_0'  # name of the training session

def train( env, agent, num_episodes = 2000 ) :
    progressbar = tqdm( range( 1, num_episodes + 1 ), desc = 'Training>' )

    scoresAvgs = []
    scoresWindow = deque( maxlen = LOG_WINDOW )
    bestScore = -np.inf
    avgScore = -np.inf

    writer = SummaryWriter( 'summary_' + TRAINING_SESSION_ID + '_reacher' )

    for iepisode in progressbar :

        _score = 0.
        _ss = env.reset()
        agent.noiseProcess.reset()

        for i in range( MAX_STEPS_IN_EPISODE ) :
            # get the action(s) to take
            _aa = agent.act( _ss )

            # take action in the environment and grab bounty
            _ssnext, _rr, _dd, _ = env.step( _aa )
            # pack the transitions for the agent
            _transitions = []
            for _s, _a, _r, _snext, _done in zip( _ss, _aa, _rr, _ssnext, _dd ) :
                if i == MAX_STEPS_IN_EPISODE - 1 :
                    _transitions.append( ( _s, _a, _r, _snext, True ) )
                else :
                    _transitions.append( ( _s, _a, _r, _snext, _done ) )
            # update the agent
            agent.update( _transitions )

            # book keeping for next iteration
            _ss = _ssnext
            _score += np.mean( _rr )

            if _dd.any() :
                break

        # update some info for logging
        bestScore = max( bestScore, _score )
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

        writer.add_scalar( 'score', _score, iepisode )
        writer.add_scalar( 'avg_score', np.mean( scoresWindow ), iepisode )
        writer.add_scalar( 'buffer_size', len( agent.replayBuffer ), iepisode )
        writer.add_scalar( 'epsilon', agent.epsilon, iepisode )


def test( env, agent, num_episodes = 10 ) :
    agent.load()

    for _ in tqdm( range( num_episodes ), desc = 'Testing> ' ) :
        _done = False
        _ss = env.reset()

        while not _done :
            _aa = agent.act( _ss )
            _ss, _rr, _dd, _ = env.step( _aa )
            env.render()


if __name__ == '__main__' :
    executableFullPath = './executables/Reacher_Linux_multi/Reacher.x86_64'
    numberOfAgents = 20

    env = UnityEnvWrapper( executableFullPath,
                           numAgents = numberOfAgents, mode = 'training', workerID = 1, seed = 0 )

    gin.parse_config_file( './configs/ddpg_reacher_multi.gin' )

    # grab configuration from gin
    with gin.config_scope( 'actor' ) :
        actorBackboneConfig = DDPGModelBackboneConfig()
    with gin.config_scope( 'critic' ) :
        criticBackboneConfig = DDPGModelBackboneConfig()
    agentConfig = DDPGAgentConfig()

    # create the backbones for both actor and critic
    actorBackbone = DDPGMlpModelBackboneActor( actorBackboneConfig )
    criticBackbone = DDPGMlpModelBackboneCritic( criticBackboneConfig )

    # create both actor and critics
    actor = DDPGActor( actorBackbone, agentConfig.lrActor )
    critic = DDPGCritic( criticBackbone, agentConfig.lrCritic )

    # create the agent
    agent = DDPGAgent( agentConfig, actor, critic )

    env.seed( 0 )
    random.seed( 0 )
    np.random.seed( 0 )

    if TRAIN :
        train( env, agent )
    else :
        test( env, agent )
