import numpy as np

# input and column dimensions
inDim = (4,4)
colDim = (5,5)

# Initialize
numRows = colDim[0]
numCols = colDim[1]
count_per_input = np.zeros(inDim)

# Only for quadratic dimensions 67x67 or even ones 86x84
def potentialPool(receptiveFieldDimension):
    fieldSize = receptiveFieldDimension[0] * receptiveFieldDimension[1]
    zeros = np.zeros(fieldSize/2)
    ones = np.append(zeros, np.ones(fieldSize/2))
    rand = np.random.permutation(ones).reshape(receptiveFieldDimension)
    return rand

# calculate flying average over input_bits number of connections to columns
for row in xrange(numRows):
    for col in xrange(numCols):
        rand = potentialPool(inDim)
        # percentage per column is 0.5 -> connected to half of the input space
        pct_per_column = np.sum(rand)*1.0 / (inDim[0]*inDim[1])
        count_per_input += rand
        print(pct_per_column)

# print percentage connections per input / total number of columns
pct_per_input = count_per_input / (numRows * numCols)
print("Percentage per input bit of connections to total number of columns \n", pct_per_input)
