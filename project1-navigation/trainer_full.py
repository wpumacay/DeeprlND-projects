
import os
import gym
import time
import argparse
import numpy as np
from tqdm import tqdm
from collections import deque
from collections import defaultdict


# import simple gridworld for testing purposes
from navigation.envs import mlagents

# @DEBUG: test environment - gridworld (sanity check) --------------------------
from navigation.envs import gridworld
from navigation.envs import gridworld_utils
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------

# import banana agents (raycast, and visual)
from navigation import agent_raycast
from navigation import agent_visual

# @DEBUG: gridworl and gym agents for testing
from navigation import agent_gridworld
from navigation import agent_gym_control

# import config utils
from navigation.dqn.utils import config

from navigation.dqn.utils import plot

# import model builder functionality (pytorch as backend)
from navigation import model_pytorch
from navigation import model_tensorflow

from IPython.core.debugger import set_trace

# logging functionality
import logger

GRIDWORLD       = False     # global variable, set by the argparser
GYM             = True      # global variable, whether or not use gym envs.
GYM_ENV         = ''        # global variable, set by the argparser
TEST            = True      # global variable, set by the argparser
TIME_START      = 0         # global variable, set in __main__
RESULTS_FOLDER  = 'results' # global variable, where to place the results of training
SEED            = 0         # global variable, set by argparser
VISUAL          = False     # @TODO: Do not use this option yet. Not fully supported
CONFIG_AGENT    = ''        # global variable, set by argparser
CONFIG_MODEL    = ''        # global variable, set by argparser

USE_DOUBLE_DQN                      = False # global variable, set by argparser
USE_PRIORITIZED_EXPERIENCE_REPLAY   = False # global variable, set by argparser
USE_DUELING_DQN                     = False # global variable, set by argparser

# @DEBUG: test method for gridworld --------------------------------------------
def plotQTable( envGridworld, agentGridworld ) :
    plt.ion()
    # evaluate the agents model for each action state
    _qvals_sa = defaultdict( lambda : np.zeros( envGridworld.nA ) )
    for s in range( envGridworld.nS ) :
        if envGridworld._isBlocked( s ) :
            _qvals_sa[s] = np.zeros( envGridworld.nA )
        elif envGridworld._isTerminal( s ) :
            _qvals_sa[s] = np.ones( envGridworld.nA ) * envGridworld.getRewardAt( s )
        else :
            _qvals_sa[s] = agentGridworld._qmodel_actor.eval( agentGridworld._preprocess( s ) )

    gridworld_utils.plotQTableInGrid( _qvals_sa, envGridworld.rows, envGridworld.cols )

# ------------------------------------------------------------------------------

def train( env, agent, sessionId, savefile, resultsFilename, replayFilename ) :
    MAX_EPISODES = agent.learningMaxSteps
    MAX_STEPS_EPISODE = 1000
    LOG_WINDOW_SIZE = 100

    _progressbar = tqdm( range( 1, MAX_EPISODES + 1 ), desc = 'Training>', leave = True )
    _maxAvgScore = -np.inf
    _scoresWindow = deque( maxlen = LOG_WINDOW_SIZE )
    _scores = []
    _scoresAvgs = []
    _stepsWindow = deque( maxlen = LOG_WINDOW_SIZE )

    _timeStart = TIME_START

    for iepisode in _progressbar :

        if GRIDWORLD or GYM:
            _state = env.reset()
        else :
            _state = env.reset( training = True )
        _score = 0
        _nsteps = 0

        while True :
            # grab action from dqn agent: runs through model, e-greedy, etc.
            _action = agent.act( _state, inference = False )
            # apply action in simulator to get the transition
            _snext, _reward, _done, _ = env.step( _action )
            ## env.render()
            _transition = ( _state, _action, _snext, _reward, _done )
            # send this transition back to the agent (to learn when he pleases)
            ## set_trace()
            agent.step( _transition )

            # prepare for next iteration
            _state = _snext
            _score += _reward
            _nsteps += 1

            if _done :
                break

        _scores.append( _score )
        _scoresWindow.append( _score )
        _stepsWindow.append( _nsteps )

        if iepisode >= LOG_WINDOW_SIZE :
            _avgScore = np.mean( _scoresWindow )
            _avgSteps = np.mean( _stepsWindow )

            _scoresAvgs.append( _avgScore )

            if _avgScore > _maxAvgScore :
                _maxAvgScore = _avgScore

            # log results
            if iepisode % LOG_WINDOW_SIZE == 0 :
                if agent._usePrioritizedExpReplay :
                    _progressbar.set_description( 'Training> Max-Avg=%.2f, Curr-Avg=%.2f, Curr=%.2f, Eps=%.2f, Beta=%.2f' % (_maxAvgScore, _avgScore, _score, agent.epsilon, agent._rbuffer.beta ) )
                else :
                    _progressbar.set_description( 'Training> Max-Avg=%.2f, Curr-Avg=%.2f, Curr=%.2f, Eps=%.2f' % (_maxAvgScore, _avgScore, _score, agent.epsilon ) )
                _progressbar.refresh()

    # save trained model
    agent.save( savefile )

    if GRIDWORLD :
        # @DEBUG: gridworl visualization of q-table---------------------------------
        plotQTable( env, agent )
        _ = input( 'Press ENTER to continue ...' )
        # --------------------------------------------------------------------------
    else :
        _timeStop = int( time.time() )
        _trainingTime = _timeStop - _timeStart

        # save training results for later visualization and analysis
        logger.saveTrainingResults( resultsFilename,
                                    sessionId,
                                    _timeStart,
                                    _scores,
                                    _scoresAvgs,
                                    agent.actorModel.losses,
                                    agent.actorModel.bellmanErrors,
                                    agent.actorModel.gradients )

        # save replay batch for later visualization and analysis
        _minibatch = agent.replayBuffer.sample( 100 )
        _ss, _aa, _rr, _ssnext = _minibatch[0], _minibatch[1], _minibatch[2], _minibatch[3]
        _q_s_batch = [ agent.actorModel.eval( agent._preprocess( state ) ) \
                       for state in _ss ]
        _replayBatch = { 'states' : _ss, 'actions' : _aa, 'rewards' : _rr, 'nextStates' : _ssnext }

        logger.saveReplayBatch( replayFilename,
                                sessionId,
                                TIME_START,
                                _replayBatch,
                                _q_s_batch )

def test( env, agent ) :

    if GYM : 
        # replace description with the appropriate one
        _descriptions = ['NOP', 'LEFT-ENGINE', 'MAIN-ENGINE', 'RIGHT-ENGINE']
    else :
        _descriptions = agent.actionsDescs

    ## _qViz = plot.QvaluesVisualizer( _descriptions )
    ## _vViz = plot.TimeSeriesVisualizer()

    _ = input( 'Ready for testing. Press ENTER to continue' )

    _progressbar = tqdm( range( 1, 10 + 1 ), desc = 'Testing>', leave = True )
    for _ in _progressbar :

        if GRIDWORLD or GYM :
            _state = env.reset()
        else :
            _state = env.reset( training = False )

        _score = 0.0
        _goodBananas = 0
        _badBananas = 0

        while True :
            _action = agent.act( _state, inference = True )
            _state, _reward, _done, _ = env.step( _action )

            _qvalues = agent.actorModel.eval( agent._preprocess( _state ) )
            ## _qViz.update( _qvalues )
            ## _vViz.update( np.max( _qvalues ) )

            if GRIDWORLD or GYM :
                env.render()
            else :
                if _reward > 0 :
                    _goodBananas += 1
                    _progressbar.write( 'Got banana! :D. So far: %d' % _goodBananas )
                elif _reward < 0 :
                    _badBananas += 1
                    _progressbar.write( 'Got bad banana :/. So far: %d' % _badBananas )

            _score += _reward

            if GRIDWORLD:
                _ = input( 'Press ENTER to continue ...' )

            if _done :
                break

        _progressbar.set_description( 'Testing> Score=%.2f' % ( _score ) )
        _progressbar.refresh()

def experiment( sessionId, 
                library, 
                savefile, 
                resultsFilename, 
                replayFilename, 
                agentConfigFilename, 
                modelConfigFilename ) :

    # grab factory-method for the model according to the library requested
    _modelBuilder = model_pytorch.DqnModelBuilder if library == 'pytorch' \
                        else model_tensorflow.DqnModelBuilder

    # grab initialization-method for the model according to the library requested
    _backendInitializer = model_pytorch.BackendInitializer if library == 'pytorch' \
                            else model_pytorch.BackendInitializer

    if not GRIDWORLD and not GYM :
        # paths to the environment executables
        _bananaExecPath = os.path.join( os.getcwd(), 'executables/Banana_Linux/Banana.x86_64' )
        _bananaHeadlessExecPath = os.path.join( os.getcwd(), 'executables/Banana_Linux_NoVis/Banana.x86_64' )
        _bananaVisualExecPath = os.path.join( os.getcwd(), 'executables/VisualBanana_Linux/Banana.x86_64' )
        ## _bananaVisualExecPath = os.path.join( os.getcwd(), 'executables/VisualBanana/VisualBanana.x86_64' )

        # instantiate the environment
        if VISUAL :
            _env = mlagents.createDiscreteActionsEnv( _bananaVisualExecPath, envType = 'visual', seed = SEED )
        else :
            _env = mlagents.createDiscreteActionsEnv( _bananaExecPath, seed = SEED )

        # instantiate the agent accordingly (visual or non-visual based environment)

        if VISUAL :

            # set the seed for the agent
            agent_visual.AGENT_CONFIG.seed = SEED
            # set whether or not to use visual-based model
            agent_visual.AGENT_CONFIG.useConvolutionalBasedModel = True

            # set improvement flags
            agent_visual.AGENT_CONFIG.useDoubleDqn             = USE_DOUBLE_DQN
            agent_visual.AGENT_CONFIG.usePrioritizedExpReplay  = USE_PRIORITIZED_EXPERIENCE_REPLAY
            agent_visual.AGENT_CONFIG.useDuelingDqn            = USE_DUELING_DQN

            _agent = agent_visual.CreateAgent( agent_visual.AGENT_CONFIG,
                                               agent_visual.MODEL_CONFIG,
                                               _modelBuilder,
                                               _backendInitializer )

            # save agent and model configurations
            config.DqnAgentConfig.save( agent_visual.AGENT_CONFIG, agentConfigFilename )
            config.DqnModelConfig.save( agent_visual.MODEL_CONFIG, modelConfigFilename )

        else :
            # set the seed for the agent
            agent_raycast.AGENT_CONFIG.seed = SEED

            # set improvement flags
            agent_raycast.AGENT_CONFIG.useDoubleDqn             = USE_DOUBLE_DQN
            agent_raycast.AGENT_CONFIG.usePrioritizedExpReplay  = USE_PRIORITIZED_EXPERIENCE_REPLAY
            agent_raycast.AGENT_CONFIG.useDuelingDqn            = USE_DUELING_DQN

            _agent = agent_raycast.CreateAgent( agent_raycast.AGENT_CONFIG,
                                                agent_raycast.MODEL_CONFIG,
                                                _modelBuilder,
                                                _backendInitializer )

            # save agent and model configurations
            config.DqnAgentConfig.save( agent_raycast.AGENT_CONFIG, agentConfigFilename )
            config.DqnModelConfig.save( agent_raycast.MODEL_CONFIG, modelConfigFilename )

    elif GRIDWORLD :
        # @DEBUG: gridworld test environment------------------------------------
        _env = gridworld.GridWorldEnv( gridworld.BOOK_LAYOUT,
                                       ## gridworld.DRLBOOTCAMP_LAYOUT,
                                       ## gridworld.DEFAULT_LAYOUT,
                                       noise = 0.0,
                                       ## noise = 0.2,
                                       ## noise = 0.0,
                                       rewardAtGoal = -1.0,
                                       rewardAtHole = -1.0,
                                       ## rewardAtGoal = 1.0,
                                       ## rewardAtHole = -1.0,
                                       ## rewardAtGoal = 10.0,
                                       ## rewardAtHole = -10.0,
                                       rewardPerStep = -1.0,
                                       ## rewardPerStep = 0.0,
                                       ## rewardPerStep = 0.0,
                                       renderInteractive = TEST,
                                       randomSeed = 0 )

        agent_gridworld.AGENT_CONFIG.stateDim = _env.nS
        agent_gridworld.AGENT_CONFIG.nActions = _env.nA
    
        agent_gridworld.MODEL_CONFIG.inputShape   = ( _env.nS, )
        agent_gridworld.MODEL_CONFIG.outputShape  = ( _env.nA, )

        # set improvement flags
        agent_gridworld.AGENT_CONFIG.useDoubleDqn             = USE_DOUBLE_DQN
        agent_gridworld.AGENT_CONFIG.usePrioritizedExpReplay  = USE_PRIORITIZED_EXPERIENCE_REPLAY
        agent_gridworld.AGENT_CONFIG.useDuelingDqn            = USE_DUELING_DQN

        _agent = agent_gridworld.CreateAgent( agent_gridworld.AGENT_CONFIG,
                                              agent_gridworld.MODEL_CONFIG,
                                              _modelBuilder,
                                              _backendInitializer )

        # ----------------------------------------------------------------------

    elif GYM :
        # @DEBUG: gym test environment------------------------------------------
        _env = gym.make( GYM_ENV )

        agent_gym_control.AGENT_CONFIG.stateDim = _env.observation_space.shape
        agent_gym_control.AGENT_CONFIG.nActions = _env.action_space.n
    
        agent_gym_control.MODEL_CONFIG.inputShape   = _env.observation_space.shape
        agent_gym_control.MODEL_CONFIG.outputShape  = ( _env.action_space.n, )

        # set improvement flags
        agent_gym_control.AGENT_CONFIG.useDoubleDqn             = USE_DOUBLE_DQN
        agent_gym_control.AGENT_CONFIG.usePrioritizedExpReplay  = USE_PRIORITIZED_EXPERIENCE_REPLAY
        agent_gym_control.AGENT_CONFIG.useDuelingDqn            = USE_DUELING_DQN

        _agent = agent_gym_control.CreateAgent( agent_gym_control.AGENT_CONFIG,
                                                agent_gym_control.MODEL_CONFIG,
                                                _modelBuilder,
                                                _backendInitializer )

        # ----------------------------------------------------------------------

    if not TEST :
        train( _env, _agent, sessionId, savefile, resultsFilename, replayFilename )
    else :
        _agent.load( _savefile )
        test( _env, _agent )

if __name__ == '__main__' :
    _parser = argparse.ArgumentParser()
    _parser.add_argument( 'mode',
                          help = 'mode of execution (train|test)',
                          type = str,
                          choices = [ 'train', 'test' ] )
    _parser.add_argument( '--library', 
                          help = 'deep learning library to use (pytorch|tensorflow)', 
                          type = str, 
                          choices = [ 'pytorch','tensorflow' ], 
                          default = 'pytorch' )
    _parser.add_argument( '--sessionId', 
                          help = 'identifier of this training run', 
                          type = str, 
                          default = 'banana_simple' )
    _parser.add_argument( '--seed',
                          help = 'random seed for the environment and generators',
                          type = int,
                          default = 0 )
    _parser.add_argument( '--gridworld',
                          help = 'whether or not to test the implementation in a gridworld env.',
                          type = str,
                          default = 'false' )
    _parser.add_argument( '--gym',
                          help = 'gym environment to use',
                          type = str,
                          default = '' )
    _parser.add_argument( '--visual',
                          help = 'whether or not use the visual-banana environment',
                          type = str,
                          default = 'false' )
    _parser.add_argument( '--ddqn',
                          help = 'whether or not to use double dqn (true|false)',
                          type = str,
                          default = 'false' )
    _parser.add_argument( '--prioritizedExpReplay',
                          help = 'whether or not to use prioritized experience replay (true|false)',
                          type = str,
                          default = 'false' )
    _parser.add_argument( '--duelingDqn',
                          help = 'whether or not to use dueling dqn (true|false)',
                          type = str,
                          default = 'false' )
    _parser.add_argument( '--configAgent',
                          help = 'configuration file for the agent (hyperparameters, etc.)',
                          type = str,
                          default = '' )
    _parser.add_argument( '--configModel',
                          help = 'configuration file for the model (architecture, etc.)',
                          type = str,
                          default = '' )

    _args = _parser.parse_args()

    # whether or not use the toy gridworld test environment
    GRIDWORLD = ( _args.gridworld.lower() == 'true' )

    # whether or not use a gym-environment
    GYM     = ( _args.gym != '' )
    GYM_ENV = _args.gym

    # whether or not we are in test mode
    TEST = ( _args.mode == 'test' )
    # the actual seed for the environment
    SEED = _args.seed
    # timestamp of the start of execution
    TIME_START = int( time.time() )

    _sessionfolder = os.path.join( RESULTS_FOLDER, _args.sessionId )
    if not os.path.exists( _sessionfolder ) :
        os.makedirs( _sessionfolder )

    # file where to save the trained model
    _savefile = _args.sessionId
    _savefile += '_model_'
    _savefile += _args.library
    _savefile += ( '.pth' if _args.library == 'pytorch' else '.h5' )
    _savefile = os.path.join( _sessionfolder, _savefile )

    # file where to save the training results statistics
    _resultsFilename = os.path.join( _sessionfolder, 
                                     _args.sessionId + '_results.pkl' )

    # file where to save the replay information (for further extra analysis)
    _replayFilename = os.path.join( _sessionfolder,
                                    _args.sessionId + '_replay.pkl' )

    # configuration files for this training session
    _agentConfigFilename = os.path.join( _sessionfolder, _args.sessionId + '_agentconfig.json' )
    _modelConfigFilename = os.path.join( _sessionfolder, _args.sessionId + '_modelconfig.json' )

    # whether or not use the visual-banana environment
    VISUAL = ( _args.visual.lower() == 'true' )

    # DQN improvements options
    USE_DOUBLE_DQN                      = ( _args.ddqn.lower() == 'true' )
    USE_PRIORITIZED_EXPERIENCE_REPLAY   = ( _args.prioritizedExpReplay.lower() == 'true' )
    USE_DUELING_DQN                     = ( _args.duelingDqn.lower() == 'true' )

    # Configuration files with training information (provided by the user)
    CONFIG_AGENT = _args.configAgent
    CONFIG_MODEL = _args.configModel

    print( '#############################################################' )
    print( '#                                                           #' )
    print( '#            Environment and agent setup                    #' )
    print( '#                                                           #' )
    print( '#############################################################' )
    print( 'Mode                    : ', _args.mode )
    print( 'Library                 : ', _args.library )
    print( 'SessionId               : ', _args.sessionId )
    print( 'Savefile                : ', _savefile )
    print( 'ResultsFilename         : ', _resultsFilename )
    print( 'ReplayFilename          : ', _replayFilename )
    print( 'AgentConfigFilename     : ', _agentConfigFilename )
    print( 'ModelConfigFilename     : ', _modelConfigFilename )
    print( 'Gridworld               : ', _args.gridworld )
    print( 'Gym                     : ', GYM )
    print( 'Gym-env                 : ', 'None' if not GYM else _args.gym )
    print( 'VisualBanana            : ', _args.visual )
    print( 'DoubleDqn               : ', _args.ddqn )
    print( 'PrioritizedExpReplay    : ', _args.prioritizedExpReplay )
    print( 'DuelingDqn              : ', _args.duelingDqn )
    print( 'Agent config file       : ', 'None' if _args.configAgent == '' else _args.configAgent )
    print( 'Model config file       : ', 'None' if _args.configModel == '' else _args.configModel )
    print( '#############################################################' )

    experiment( _args.sessionId, 
                _args.library,
                _savefile,
                _resultsFilename,
                _replayFilename,
                _agentConfigFilename,
                _modelConfigFilename )