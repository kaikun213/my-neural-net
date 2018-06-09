
import numpy as np

class SpatialPooler:

    def __init__(self, inputDimensions = (32, 32),
                       columnDimensions = (64, 64),
                       potentialConnectionsPct = 0.5,
                       connectionThreshold = 0.1,
                       permanenceInc = 0.1,
                       permanenceDec = 0.05,
                       overlapPct = 0.2
                       ):

        inputDimensions = np.array(inputDimensions)
        columnDimensions = np.array(columnDimensions)

        self._numInputs = inputDimensions.prod()
        self._numColumns = columnDimensions.prod()
        self._potentialConnectionsPct = potentialConnectionsPct
        self._connectionThreshold = connectionThreshold
        self._permanenceInc = permanenceInc
        self._permanenceDec = permanenceDec
        self._overlapPct = overlapPct

        # Potential pool to input space for each minicolumn.
        # Represented as a binary matrix (numColumns, numInputs).
        # A potential connection from the i'th minicolumn to the j'th input-bit can be made if [i,j] = 1
        self._potentialPools = np.zeros((numColumns, numInputs))

        # The permanence values for each minicolumns potential connections.
        # Represented as a matrix with float values 0-1, dimensions (numColumns, numInputs)
        # Only if the potentialPool value is 1 it can have a permanence value.
        self._permanences = np.zeros((numColumns, numInputs), dtype = np.float32)

        # A tiny random tie breaker. This is used to determine winning
        # columns where the overlaps are identical.
        self._tieBreaker = numpy.array([0.01 * self._random.getReal64() for i in
                                        xrange(self._numColumns)], dtype=realDType)

        # The connections from each minicolumn to the input space
        # It is essentially the same as the 'self._permanences' matrix (numColumns, numInputs)
        # Represented as a binary matrix instead of float values we will have
        # '1' if it is above 'self._connectionThreshold' and zero otherwise.
        self._connections = np.zeros((numColumns, numInputs))

        # Amount of connections counted for each cortical column.
        # Information is contained in 'self._connections' but used for simplicity and efficiency
        self._connectedCount = numpy.zeros(numColumns)

        # Initialize permanence values
        for columnIndex in xrange(numColumns):
            
