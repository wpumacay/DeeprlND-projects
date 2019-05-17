
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

class TimeSeriesVisualizer( object ) :

    def __init__( self, rmin = None, rmax = None ) :
        super( TimeSeriesVisualizer, self ).__init__()

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

        if rmin is not None and rmax is not None :
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
        plt.pause( 0.0001 )

class QvaluesVisualizer( object ) :

    def __init__( self, actions, rmin = None, rmax = None ) :
        super( QvaluesVisualizer, self ).__init__()

        # force interactive mode in case not enable
        plt.ion()

        # descriptions of the actions
        self.m_actions = actions

        # plotting resources
        self.m_fig = plt.figure()
        self.m_axs = self.m_fig.add_subplot( 111 )

        # min-max range
        self.m_rmin = rmin
        self.m_rmax = rmax

        if rmin is not None and rmax is not None :
            # configure the limits of the figure
            self.m_axs.set_ylim( self.m_rmin, self.m_rmax )

    def update( self, qValues ) :

        self.m_axs.cla()
        _bars = self.m_axs.bar( self.m_actions, qValues )

        _maxIndx    = np.argmax( qValues )
        _maxValue   = qValues[_maxIndx]
        _bars[_maxIndx].set_color( 'r' )

        self.m_axs.axhline( _maxValue, linestyle = '--', color = 'k' )

        plt.pause( 0.0001 )