
import itertools
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

def _ndarrayToDict( ndarray, dimToStop = -1 ) :
    _ndim = ndarray.ndim

    if _ndim == dimToStop :
        return ndarray
    elif _ndim == 1 :
        return { k : ndarray[k] for k in range( len( ndarray ) ) }
    else :
        _dict = {}
    
        for i in range( len( ndarray ) ) :
            _dict[i] = _ndarrayToDict( ndarray[i], dimToStop )

    return _dict

def _plotStateTableInGrid( table, rows, cols, title = 'State table', extFig = None, extAxes = None ) : 

    # just in case, if ndarray, convert it to a dictionary
    if type( table ) == np.ndarray :
        table = _ndarrayToDict( table )

    if ( extFig is None ) or ( extAxes is None ) :
        _fig = plt.figure()
        _axes = _fig.add_axes( [ 0, 0, 1, 1 ] )
    else :
        _fig = extFig
        _axes = extAxes

    _matrix = np.zeros( ( rows, cols ), dtype=np.float32 )

    for _s in table.keys() :
        # convert state id to grid position
        _row = _s // cols
        _col = _s % cols
        # set the value V(s) to the matrix
        _matrix[_row,_col] = table[_s]

    _axes.imshow( _matrix, 
                  interpolation = 'nearest', 
                  cmap = plt.cm.Blues )

    for i, j in itertools.product( range( _matrix.shape[0] ),
                                   range( _matrix.shape[1] ) ) :
        _x = ( float(j) + 0.5 ) / _matrix.shape[1]
        _y = 1 - ( float(i) + 0.5 ) / _matrix.shape[0]

        _axes.text( _x, _y, '{0:.2f}'.format( _matrix[i, j] ),
                    horizontalalignment = 'center',
                    transform = _axes.transAxes,
                    color = 'black' )

    _fig.show()

def plotVTableInGrid( vtable, rows, cols ) :
    _plotStateTableInGrid( vtable, rows, cols, 'V-value for each state V(s)' )

def plotVisitsInGrid( visitsTable, rows, cols ) :
    _plotStateTableInGrid( visitsTable, rows, cols, 'Number of visits for each state N(s)' )

def plotVisitsInHistogram( visitsTable, nStates ) :
    _centers = np.arange( 0, nStates )

    _hist = np.zeros( _centers.shape )

    for _s in visitsTable.keys() :
        _hist[_s] = visitsTable[_s]

    plt.figure()
    plt.title( 'Visits histogram' )
    plt.bar( _centers, _hist, align = 'center', width = 1 )
    plt.show()

def _plotStateActionTableInGrid( table, rows, cols, title = 'StateAction table', extFig = None, extAxes = None ) :

    # just in case, if ndarray, convert it to a dictionary
    if type( table ) == np.ndarray :
        table = _ndarrayToDict( table, dimToStop = 1 )

    if ( extFig is None ) or ( extAxes is None ) :
        _fig = plt.figure()
        _axes = _fig.add_axes( [ 0, 0, 1, 1 ] )
    else :
        _fig = extFig
        _axes = extAxes

    _matrix = np.zeros( ( rows, cols ), dtype=np.float32 )

    for _s in table.keys() :
        # convert state id to grid position
        _row = _s // cols
        _col = _s % cols
        # set the average over all action values for '_s' to the matrix
        _matrix[_row,_col] = np.mean( table[_s] )

    _axes.imshow( _matrix, 
                  interpolation = 'nearest', 
                  cmap = plt.cm.Blues )

    for i, j in itertools.product( range( _matrix.shape[0] ),
                                   range( _matrix.shape[1] ) ) :

        # convert i,j to state id
        _s = i * _matrix.shape[1] + j

        _dx = 1. / _matrix.shape[1]
        _dy = 1. / _matrix.shape[0]

        # draw the lines
        _p0 = [ j * _dx, 1 - i * _dy ]
        _p1 = [ j * _dx, 1 - ( i + 1 ) * _dy ]
        _p2 = [ ( j + 1 ) * _dx, 1 - i * _dy ]
        _p3 = [ ( j + 1 ) * _dx, 1 - ( i + 1 ) * _dy ]

        _axes.plot( [_p0[0], _p3[0]], [_p0[1], _p3[1]], 'k', transform = _axes.transAxes )
        _axes.plot( [_p1[0], _p2[0]], [_p1[1], _p2[1]], 'k', transform = _axes.transAxes )

        # draw the text for each slot in each cell
        _actions = [0, 1, 2, 3] # there are 3 actions in this environment
        _offsets = [ (-0.7, -1.6), (-0.7, 1.5), (-1.7, 0), (0.6, 0) ] # offsets for each slot

        _xc = ( j + 0.5 ) * _dx
        _yc = 1 - ( i + 0.5 ) * _dy
        
        for k in range( len( _actions ) ) :
            _a = _actions[k]
            _off = _offsets[k]

            _x = _xc + _off[0] * _dx * 0.25
            _y = _yc - _off[1] * _dy * 0.25

            _axes.text( _x, _y, '{0:.2f}'.format( table[_s][_a] ),
                        transform = _axes.transAxes,
                        color = 'black' )

    _fig.show()

def plotQTableInGrid( qtable, rows, cols ) :
    _plotStateActionTableInGrid( qtable, rows, cols, 'Q-value for each state-action Q(s,a)' )


class VTableVisualizer( object ) :

    def __init__( self, rows, cols ) :
        super( VTableVisualizer, self ).__init__()

        # force interactive mode in case not enabled
        plt.ion()

        # plotting resources
        self.m_fig = plt.figure()
        self.m_axs = self.m_fig.add_axes( [ 0, 0, 1, 1 ] )

        # gridworld properties
        self.m_rows = rows
        self.m_cols = cols

    def update( self, vtable ) :
        self.m_axs.cla()

        _plotStateTableInGrid( vtable, self.m_rows, self.m_cols, 
                               'State-value functon V(s)',
                               self.m_fig, self.m_axs )
        plt.pause( 0.001 )

class QTableVisualizer( object ) :

    def __init__( self, rows, cols ) :
        super( VTableVisualizer, self ).__init__()

        # force interactive mode in case not enabled
        plt.ion()

        # plotting resources
        self.m_fig = plt.figure()
        self.m_axs = self.m_fig.add_axes( [ 0, 0, 1, 1 ] )

        # gridworld properties
        self.m_rows = rows
        self.m_cols = cols

    def update( self, qtable ) :
        self.m_axs.cla()

        _plotStateActionTableInGrid( qtable, self.m_rows, self.m_cols, 
                                     'State-Action-value functon Q(s,a)',
                                     self.m_fig, self.m_axs )
        plt.pause( 0.001 )


class HyperparameterScheduleVisualizer( object ) :

    def __init__( self, rmin, rmax ) :
        super( HyperparameterScheduleVisualizer, self ).__init__()

        # force interactive mode in case not enable
        plt.ion()

        # plotting resources
        self.m_fig = plt.figure()
        self.m_axs = self.m_fig.add_subplot( 111 )

        # min-max range
        self.m_rmin = rmin
        self.m_rmax = rmax

        # and a queue for maxlen=100 values
        self.m_deque = deque( maxlen = 100 )

        # configure the limits of the figure
        self.m_axs.set_ylim( self.m_rmin, self.m_rmax )

        # a step counter
        self.m_tstep = 0

    def update( self, value ) :
        self.m_tstep += 1
        self.m_deque.append( value )

        _values = list( self.m_deque )
        _ticks = np.arange( self.m_tstep - len( _values ) + 1,
                            self.m_tstep + 1, 1 )

        self.m_axs.cla()
        self.m_axs.plot( _ticks, _values )
        plt.pause( 0.001 )