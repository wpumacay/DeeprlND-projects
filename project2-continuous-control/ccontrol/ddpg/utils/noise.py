
import random
import numpy as np

from IPython.core.debugger import set_trace

class OUNoise( object ) :

    def __init__( self, noiseShape, ouMu, ouTheta, ouSigma, seed ) :
        super( OUNoise, self ).__init__()

        self._mu = ouMu * np.ones( noiseShape )
        self._theta = ouTheta
        self._sigma = ouSigma
        self._state = self._mu.copy()
        self._seed = random.seed( seed )


    def reset( self ) :
        self._state = self._mu.copy()


    def sample( self ) :
        x = self._state
        dx = self._theta * ( self._mu - x ) + self._sigma * np.array( [ random.random() for i in range( len( x ) ) ] )
        self._state = x + dx

        return self._state.copy()


class Normal( object ) :

    def __init__( self, noiseShape, stddev, seed ) :
        super( Normal, self ).__init__()

        self._mu = np.zeros( noiseShape )
        self._std = stddev
        self._seed = np.random.seed( seed )


    def reset( self ) :
        pass


    def sample( self ) :
        return self._mu + self._std * np.random.randn( *self._mu.shape )