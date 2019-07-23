
import csv
import time
import pickle
import numpy as np


def saveTrainingResults( filename, sessionId, timestamp, scoresAll, scoresAvg, losses, bellmanErrors, gradients ) :
    """Saves training information into pickle file for later usage

    Args:
        filename        (str)   : file where to save the data
        sessionId       (str)   : an extra identifier to remind us of the run
        timestamp       (float) : time stamp represented as seconds since 1970
        scoresAll       (list)  : all scores over all episodes
        scoresAvg       (list)  : average scores over a window of 100 episodes
        losses          (list)  : all training losses for each minibatch
        bellmanErrors   (list)  : all bellman errors saved during training
        gradients       (list)  : all training gradients for each minibatch

    """
    _data = { 'sessionId' : sessionId,
              'timestamp' : timestamp,
              'scoresAll' : scoresAll,
              'scoresAvg' : scoresAvg,
              'losses' : losses,
              'bellmanErrors' : bellmanErrors,
              'gradients' : gradients }

    with open( filename, 'wb' ) as file :
        pickle.dump( _data, file )

def loadTrainingResults( filename ) :
    """Loads a pickled file with training information (usually for plotting)

    Args:
        filename (str) : file from where we will load the training data

    """
    with open( filename, 'rb' ) as file :
        _data = pickle.load( file )

    ## print( 'INFO> loaded training data from session [%s] with timestamp [%s]' % \
    ##             ( _data['sessionId'], time.ctime( int( _data['timestamp'] ) ) ) )

    return _data

def saveReplayBatch( filename, sessionId, timestamp, replayBatch, qvalsBatch ) :
    """Saves a small batch from the replaybuffer for later checks|visualization

    Args:
        filename    (str)   : file where to save the batch
        sessionId   (str)   : an extra identifier to remind us of the run
        timestamp   (float) : time stamp represented as seconds since 1970
        replayBatch (list)  : a list with all experience objects (s,a,s',r)
        qvalsBatch  (list)  : a list with the Qs evaluated at each (s,a)

    """
    _dataBatch = { 'sessionId' : sessionId,
                   'timestamp' : timestamp,
                   'experiences' : replayBatch,
                   'qvalues' : qvalsBatch }

    with open( filename, 'wb' ) as file :
        pickle.dump( _dataBatch, file )

def loadReplayBatch( filename ) :
    """Loads a pickled file with a batch from the replay, and some more (above)

    Args:
        filename (str) : file from where we will load the replay batch

    """
    with open( filename, 'rb' ) as file :
        _dataBatch = pickle.load( filename )

    ## print( 'INFO> loaded replay batch from session [%s] with timestamp [%s]' % \
    ##             ( _dataBatch['sessionId'], time.ctime( int( _dataBatch['timestamp'] ) ) ) )

    return _data

def loadTrainingResultsCsv( filename ) :
    """Loads exported .csv files from tensorboard

    Args:
        filename (str): csv file where tensorboard exported results

    """
    _steps = []
    _data = []
    with open( filename, 'r' ) as file :
        _csvDictReader = csv.DictReader( file )
        for row in _csvDictReader :
            _steps.append( int( row['Step'] ) )
            _data.append( float( row['Value'] ) )

    return [ _steps, _data ]