
import os
import sys
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

from ccontrol.ddpg.utils.config import DDPGTrainerConfig
from ccontrol.ddpg.utils.config import DDPGAgentConfig
from ccontrol.ddpg.utils.config import DDPGModelBackboneConfig

# use pytorch backend (currently only one supported for now)
from ccontrol.ddpg.models.pytorch import DDPGMlpModelBackboneActor
from ccontrol.ddpg.models.pytorch import DDPGMlpModelBackboneCritic
from ccontrol.ddpg.models.pytorch import DDPGActor
from ccontrol.ddpg.models.pytorch import DDPGCritic

from IPython.core.debugger import set_trace

TRAIN                   = True                          # whether or not to train our agent
LOG_WINDOW              = 100                           # size of the smoothing window and logging window
TRAINING_EPISODES       = 2000                          # number of training episodes
MAX_STEPS_IN_EPISODE    = 3000                          # maximum number of steps in an episode
TRAINING_SESSION_ID     = 'session_default'             # name of the training session
SESSION_FOLDER          = './results/session_default'   # folder where to save training results
NUMBER_OF_AGENTS        = 20                            # number of agents in the environment
EXECUTABLE_FULL_PATH    = './executables/Reacher_Linux_multi/Reacher.x86_64'

def train( env, agent, numEpisodes ) :
    r"""Training loop for our given agent in a given environment

    Args:
        env (gym.Env)               : Gym-like environment (or wrapper) used to train the agent
        agent (ddpg.core.DDPGAgent) : ddpg-based agent to be trained
        numEpisodes (int)           : number of episodes to train

    """
    progressbar = tqdm( range( 1, numEpisodes + 1 ), desc = 'Training>' )

    scoresAvgs = []
    scoresWindow = deque( maxlen = LOG_WINDOW )
    bestMeanScore = -np.inf
    bestSingleScore = -np.inf
    avgScore = -np.inf

    writer = SummaryWriter( os.path.join( SESSION_FOLDER, 'tensorboard_summary' ) )

    for iepisode in progressbar :

        _scoresPerAgent = np.zeros( NUMBER_OF_AGENTS )
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
                    if _dd.any() :
                        # this is used for environments in which different ...
                        #'done' signals are received for different agents, ...
                        # like the crawler case (some might end, but some not)
                        _transitions.append( ( _s, _a, _r, _snext, True ) )
                    else :
                        _transitions.append( ( _s, _a, _r, _snext, _done ) )

            # update the agent
            agent.update( _transitions )

            # book keeping for next iteration
            _ss = _ssnext
            _scoresPerAgent += _rr

            if _dd.any() :
                break

        # update some info for logging
        _meanScore = np.mean( _scoresPerAgent )
        bestMeanScore = max( bestMeanScore, _meanScore )
        bestSingleScore = max( bestSingleScore, np.max( _scoresPerAgent ) )
        scoresWindow.append( _meanScore )

        if iepisode >= LOG_WINDOW :
            avgScore = np.mean( scoresWindow )
            scoresAvgs.append( avgScore )
            message = 'Training> best-mean: %.2f - best-single: %.2f - current-mean-window: %.2f - current-mean: %.2f'
            progressbar.set_description( message % ( bestMeanScore, bestSingleScore, avgScore, _meanScore ) )
            progressbar.refresh()
        else :
            message = 'Training> best-mean: %.2f - best-single: %.2f - current-mean : %.2f'
            progressbar.set_description( message % ( bestMeanScore, bestSingleScore, _meanScore ) )
            progressbar.refresh()

        writer.add_scalar( 'log_1_mean_score', _meanScore, iepisode )
        writer.add_scalar( 'log_2_mean_score_window', np.mean( scoresWindow ), iepisode )
        writer.add_scalar( 'log_3_buffer_size', len( agent.replayBuffer ), iepisode )
        writer.add_scalar( 'log_4_epsilon', agent.epsilon, iepisode )


def test( env, agent, numEpisodes = 10 ) :
    r"""Test an agent in a given environment

    Args:
        env (gym.Env)               : Gym-like environment (or wrapper) in which to test the agent
        agent (ddpg.core.DDPGAgent) : ddpg-based trained agent we want to test
        numEpisodes (int)           : number of episodes to test

    """
    progressbar = tqdm( range( 1, numEpisodes + 1 ), desc = 'Testing>' )

    agent.setMode( 'test' )
    agent.load()

    for iepisode in progressbar :
        _score = 0.
        _ss = env.reset()
        _nsteps = 0

        while True :
            _aa = agent.act( _ss )
            _ss, _rr, _dd, _ = env.step( _aa )
            env.render()

            _score += np.mean( _rr )
            _nsteps += 1

            if _dd.any() :
                break

        progressbar.set_description( 'Testing> score: %.2f, nsteps: %.2f' % ( _score, _nsteps ) )
        progressbar.refresh()


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument( 'mode', help='mode to run the script (train|test)', type=str, choices=['train','test'], default='train' )
    parser.add_argument( '--config', help='gin-config file with the trainer configuration', type=str, default='./configs/ddpg_reacher_multi_default.gin' )
    parser.add_argument( '--seed', help='random seed to be used all over the place', type=int, default=None )

    args = parser.parse_args()

    gin.parse_config_file( args.config )

    # grab training mode
    TRAIN = ( args.mode == 'train' )

    # grab configuration from gin
    trainerConfig = DDPGTrainerConfig()
    agentConfig = DDPGAgentConfig()
    with gin.config_scope( 'actor' ) :
        actorBackboneConfig = DDPGModelBackboneConfig()
    with gin.config_scope( 'critic' ) :
        criticBackboneConfig = DDPGModelBackboneConfig()

    # check for command line seed. if so, override all seeds in gin files
    if args.seed is not None :
        trainerConfig.seed          = args.seed
        agentConfig.seed            = args.seed
        actorBackboneConfig.seed    = args.seed
        criticBackboneConfig.seed   = args.seed
        # slightly modify the session id
        trainerConfig.sessionID += ( '_seed_' + str( args.seed ) )

    env = UnityEnvWrapper( EXECUTABLE_FULL_PATH,
                           numAgents = NUMBER_OF_AGENTS, 
                           mode = 'training' if TRAIN else 'testing', 
                           workerID = 0, 
                           seed = trainerConfig.seed )

    # in case the results directory for this session does not exist, create a new one
    SESSION_FOLDER = os.path.join( './results', trainerConfig.sessionID )
    if not os.path.exists( SESSION_FOLDER ) :
        os.makedirs( SESSION_FOLDER )

    TRAINING_EPISODES = trainerConfig.numTrainingEpisodes
    MAX_STEPS_IN_EPISODE = trainerConfig.maxStepsInEpisode
    TRAINING_SESSION_ID = trainerConfig.sessionID

    # create the backbones for both actor and critic
    actorBackbone = DDPGMlpModelBackboneActor( actorBackboneConfig )
    criticBackbone = DDPGMlpModelBackboneCritic( criticBackboneConfig )

    # create both actor and critics
    actor = DDPGActor( actorBackbone, agentConfig.lrActor )
    critic = DDPGCritic( criticBackbone, agentConfig.lrCritic )

    # create the agent
    agent = DDPGAgent( agentConfig, actor, critic )
    agent.setSaveDir( SESSION_FOLDER )

    env.seed( trainerConfig.seed )
    random.seed( trainerConfig.seed )
    np.random.seed( trainerConfig.seed )

    if TRAIN :
        train( env, agent, TRAINING_EPISODES )
    else :
        test( env, agent )
