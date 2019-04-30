
import os
import numpy as np
from tqdm import tqdm
from collections import deque

# import simple gridworld for testing purposes
from navigation.envs import mlagents

# import banana agent (raycast )
from navigation import agent_raycast

# import model builder functionality (pytorch as backend)
from navigation import model_pytorch

from IPython.core.debugger import set_trace

TEST = False



def train( env, agent, savefile ) :
    MAX_EPISODES = agent.learningMaxSteps
    MAX_STEPS_EPISODE = 1000
    LOG_WINDOW_SIZE = 100

    _progressbar = tqdm( range( 1, MAX_EPISODES + 1 ), desc = 'Training>', leave = True )
    _maxAvgScore = -np.inf
    _scoresWindow = deque( maxlen = LOG_WINDOW_SIZE )
    _scores = []

    for iepisode in _progressbar :

        _state = env.reset( training = True )
        _score = 0

        for istep in range( MAX_STEPS_EPISODE ) :
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

            if _done :
                break

        _scores.append( _score )
        _scoresWindow.append( _score )

        if iepisode >= LOG_WINDOW_SIZE :
            _avgScore = np.mean( _scoresWindow )
            if _avgScore > _maxAvgScore :
                _maxAvgScore = _avgScore

            # log results
            if iepisode % LOG_WINDOW_SIZE == 0 :
                ## set_trace()
                _progressbar.set_description( 'Training> Max-Avg=%.2f, Curr=%.2f, Eps=%.2f' % (_maxAvgScore, _score, agent.epsilon) )
                _progressbar.refresh()

    if savefile is not None :
        agent.save( savefile )

def test( env, agent ) :
    for _ in range( 10 ) :

        _state = env.reset( training = False )

        while True :
            _action = agent.act( _state, inference = True )
            _state, _, _done, _ = env.step( _action )

            if _done :
                break

def experiment() :
    # paths to the environment executables
    _bananaExecPath = os.path.join( os.getcwd(), 'executables/Banana_Linux/Banana.x86_64' )
    _bananaHeadlessExecPath = os.path.join( os.getcwd(), 'executables/Banana_Linux_NoVis/Banana.x86_64' )

    # instantiate the environment
    _env = mlagents.createDiscreteActionsEnv( _bananaExecPath if TEST else _bananaHeadlessExecPath )

    # instantiate the agent
    _agent = agent_raycast.CreateAgent( model_pytorch.DqnModelBuilder,
                                        model_pytorch.BackendInitializer )

    if not TEST :
        train( _env, _agent, 'banana_model_weights.pth' )
    else :
        test( _env, _agent )

if __name__ == '__main__' :
    experiment()