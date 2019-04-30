
import os
import numpy as np
import argparse
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

# import model builder functionality (pytorch as backend)
from navigation import model_pytorch

from IPython.core.debugger import set_trace

GRIDWORLD = False
TEST = False

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

def train( env, agent, savefile ) :
    MAX_EPISODES = agent.learningMaxSteps
    MAX_STEPS_EPISODE = 1000
    LOG_WINDOW_SIZE = 100

    _progressbar = tqdm( range( 1, MAX_EPISODES + 1 ), desc = 'Training>', leave = True )
    _maxAvgScore = -np.inf
    _scoresWindow = deque( maxlen = LOG_WINDOW_SIZE )
    _scores = []
    _stepsWindow = deque( maxlen = LOG_WINDOW_SIZE )

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
        if savefile is not None :
            agent.save( savefile )

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

def experiment( savefile ) :
    # paths to the environment executables
    _bananaExecPath = os.path.join( os.getcwd(), 'executables/Banana_Linux/Banana.x86_64' )
    _bananaHeadlessExecPath = os.path.join( os.getcwd(), 'executables/Banana_Linux_NoVis/Banana.x86_64' )

    if not GRIDWORLD :
        # instantiate the environment
        _env = mlagents.createDiscreteActionsEnv( _bananaExecPath )

        # instantiate the agent
        _agent = agent_raycast.CreateAgent( agent_raycast.AGENT_CONFIG,
                                            agent_raycast.MODEL_CONFIG,
                                            model_pytorch.DqnModelBuilder,
                                            model_pytorch.BackendInitializer )
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

        _agent = agent_gridworld.CreateAgent( agent_gridworld.AGENT_CONFIG,
                                              agent_gridworld.MODEL_CONFIG,
                                              model_pytorch.DqnModelBuilder,
                                              model_pytorch.BackendInitializer )

        # ----------------------------------------------------------------------

    if not TEST :
        train( _env, _agent, savefile )
    else :
        _agent.load( savefile )
        test( _env, _agent )

if __name__ == '__main__' :
    _parser = argparse.ArgumentParser()
    _parser.add_argument( '--filename', help='file to save|load the model', type=str, default='banana_model_weights.pth' )

    _args = _parser.parse_args()

    experiment( _args.filename )