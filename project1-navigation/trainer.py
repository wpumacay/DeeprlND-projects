
import os
import numpy as np
import argparse
import time
from tqdm import tqdm
from collections import deque
from collections import defaultdict

# import simple gridworld for testing purposes
from navigation.envs import mlagents

# @DEBUG: test environment - gridworld (sanity check) --------------------------
from rl.envs import gridworld
from rl.envs import gridworld_utils
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------

# import banana agent (raycast )
from navigation import agent_raycast
# @DEBUG: gridworl agent for testing
from navigation import agent_gridworld

# import config utils
from navigation.dqn.utils import config

# import model builder functionality (pytorch as backend)
from navigation import model_pytorch
from navigation import model_tensorflow

from IPython.core.debugger import set_trace

# logging functionality
import logger

GRIDWORLD = False
TEST = True
TIME_START = 0
RESULTS_FOLDER = 'results'
SEED = 0

USE_DOUBLE_DQN = False
USE_PRIORITIZED_EXPERIENCE_REPLAY = False
USE_DUELING_DQN = False

# @DEBUG: test method for gridworld --------------------------------------------
def plotQTable( envGridworld, agentGridworld ) :
    plt.ion()
    # evaluate the agents model for each action state
    _qvals_sa = defaultdict( lambda : np.zeros( envGridworld.nA ) )
    for s in range( envGridworld.nS ) :
        if envGridworld._isTerminal( s ) :
            _qvals_sa[s] = np.zeros( envGridworld.nA )
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
                ## set_trace()
                _progressbar.set_description( 'Training> Max-Avg=%.2f, Curr-Avg=%.2f, Curr=%.2f, Eps=%.2f, Nsteps=%d' % (_maxAvgScore, _avgScore, _score, agent.epsilon, _nsteps ) )
                _progressbar.refresh()

    if GRIDWORLD :
        # @DEBUG: gridworl visualization of q-table---------------------------------
        plotQTable( env, agent )
        _ = input( 'Press ENTER to continue ...' )
        # --------------------------------------------------------------------------
    else :
        # save trained model
        agent.save( savefile )

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
        _ss, _aa, _rr, _ssnext, _ = agent.replayBuffer.sample( 100 )
        _q_s_batch = [ agent.actorModel.eval( agent._preprocess( state ) ) \
                       for state in _ss ]
        _replayBatch = { 'states' : _ss, 'actions' : _aa, 'rewards' : _rr, 'nextStates' : _ssnext }

        logger.saveReplayBatch( replayFilename,
                                sessionId,
                                TIME_START,
                                _replayBatch,
                                _q_s_batch )

def test( env, agent ) :
    _progressbar = tqdm( range( 1, 10 + 1 ), desc = 'Testing>', leave = True )
    for _ in _progressbar :

        _state = env.reset( training = False )
        _score = 0.0
        _goodBananas = 0
        _badBananas = 0

        while True :
            _action = agent.act( _state, inference = True )
            _state, _reward, _done, _ = env.step( _action )

            if _reward > 0 :
                _goodBananas += 1
                _progressbar.write( 'Got banana! :D. So far: %d' % _goodBananas )
            elif _reward < 0 :
                _badBananas += 1
                _progressbar.write( 'Got bad banana :/. So far: %d' % _badBananas )

            _score += _reward

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
                            else model_tensorflow.BackendInitializer

    if not GRIDWORLD :
        # paths to the environment executables
        _bananaExecPath = os.path.join( os.getcwd(), 'executables/Banana_Linux/Banana.x86_64' )
        _bananaHeadlessExecPath = os.path.join( os.getcwd(), 'executables/Banana_Linux_NoVis/Banana.x86_64' )

        # instantiate the environment
        _env = mlagents.createDiscreteActionsEnv( _bananaExecPath, seed = SEED )

        # set the seed for the agent
        agent_raycast.AGENT_CONFIG.seed = SEED

        # set improvement flags
        agent_raycast.AGENT_CONFIG.useDoubleDqn             = USE_DOUBLE_DQN
        agent_raycast.AGENT_CONFIG.usePrioritizedExpReplay  = USE_PRIORITIZED_EXPERIENCE_REPLAY
        agent_raycast.AGENT_CONFIG.useDuelingDqn            = USE_DUELING_DQN

        # instantiate the agent
        _agent = agent_raycast.CreateAgent( agent_raycast.AGENT_CONFIG,
                                            agent_raycast.MODEL_CONFIG,
                                            _modelBuilder,
                                            _backendInitializer )

        # save agent and model configurations
        config.DqnAgentConfig.save( agent_raycast.AGENT_CONFIG, agentConfigFilename )
        config.DqnModelConfig.save( agent_raycast.MODEL_CONFIG, modelConfigFilename )

    else :
        # @DEBUG: gridworld test environment------------------------------------
        _env = gridworld.GridWorldEnv( gridworld.BOOK_LAYOUT, # DEFAULT_LAYOUT
                                       noise = 0.0,
                                       rewardAtGoal = -1.0, # 10.0
                                       rewardAtHole = -1.0, # -10.0
                                       rewardPerStep = -1.0,
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

    _args = _parser.parse_args()

    GRIDWORLD = ( _args.gridworld.lower() == 'true' )

    TEST = ( _args.mode == 'test' )
    SEED = _args.seed
    TIME_START = int( time.time() )

    _sessionfolder = os.path.join( RESULTS_FOLDER, _args.sessionId )
    if not os.path.exists( _sessionfolder ) :
        os.makedirs( _sessionfolder )

    _savefile = _args.sessionId
    _savefile += '_model_'
    _savefile += _args.library
    _savefile += ( '.pth' if _args.library == 'pytorch' else '.h5' )
    _savefile = os.path.join( _sessionfolder, _savefile )

    _resultsFilename = os.path.join( _sessionfolder, 
                                     _args.sessionId + '_results.pkl' )

    _replayFilename = os.path.join( _sessionfolder,
                                    _args.sessionId + '_replay.pkl' )

    _agentConfigFilename = os.path.join( _sessionfolder, _args.sessionId + '_agentconfig.json' )
    _modelConfigFilename = os.path.join( _sessionfolder, _args.sessionId + '_modelconfig.json' )

    USE_DOUBLE_DQN                      = ( _args.ddqn.lower() == 'true' )
    USE_PRIORITIZED_EXPERIENCE_REPLAY   = ( _args.prioritizedExpReplay.lower() == 'true' )
    USE_DUELING_DQN                     = ( _args.duelingDqn.lower() == 'true' )

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
    print( 'DoubleDqn               : ', _args.ddqn )
    print( 'PrioritizedExpReplay    : ', _args.prioritizedExpReplay )
    print( 'DuelingDqn              : ', _args.duelingDqn )
    print( '#############################################################' )

    experiment( _args.sessionId, 
                _args.library,
                _savefile,
                _resultsFilename,
                _replayFilename,
                _agentConfigFilename,
                _modelConfigFilename )