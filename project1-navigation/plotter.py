
import numpy as np
import matplotlib.pyplot as plt


def drawStdPlot( batchResults, title, xlabel, ylabel, color = 'r', axes = None ) : 
    """Draws an std-plot from a batch of results from various experiments
    
    Args:
        batchResults (list): list of experiments, each element containing results
                             for each expriment (training run) as a list of elements

    """
    # sanity-check: should have at least pass some data
    assert len( batchResults ) > 0, 'ERROR> should have passed at least some data'

    # convert batch to user friendly np.ndarray
    batchResults = np.array( batchResults )

    # if no axes given, create a new one
    if axes is None :
        fig, axes = plt.subplots()

    # each element has _niters elements on it (iters: episodes, steps, etc.)
    _niters = batchResults.shape[1]
    _xx = np.arange( _niters )

    # grab mean and std over all runs
    _mean = batchResults.mean( axis = 0 )
    _std = batchResults.std( axis = 0 )

    # do the plotting
    axes.plot( _xx, _mean, color = color, linestyle = '-' )
    axes.fill_between( _xx, _mean - 2. * _std, _mean + 2. * _std, color = color, alpha = 0.5 )

    return axes

def drawBatchResults( batchResults, title, xlabel, ylabel, color = 'r', axes = None ) :
    """Draws a single set of line-plots, one line per run

    Args:
        batchResults (list): list of experiments, each element containing results
                             for each expriment (training run) as a list of elements

    """

    # sanity-check: should have at least pass some data
    assert len( batchResults ) > 0, 'ERROR> should have passed at least some data'

    if axes is None :
        fig = plt.figure()
        axes = fig.add_subplot( 111 )

    # each element has _niters elements on it (iters: episodes, steps, etc.)
    _nruns = len( batchResults )
    _niters = len( batchResults[0] )
    _xx = np.arange( _niters )

    # plot using matplotlib (just like that :o)
    for i in range( _nruns ) :
        plt.plot( _xx, batchResults[i], color = np.random.rand( 3, ) )

    return axes