import csv
import matplotlib.pyplot as plt
import numpy as np

def runningMean(a,n):
    return np.convolve(a, np.ones((n,))/n, mode='valid')

# with open('results.csv', 'rb') as csvfile:
#     resultsreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     for row in resultsreader:
#         print ', '.join(row)

# load results
results = np.genfromtxt('results_static_experiment/results.csv', delimiter=',')
neuralActivations = np.genfromtxt('results_static_experiment/neural-activations.csv', delimiter=',')
neuralPredictions = np.genfromtxt('results_static_experiment/neural-predictions.csv', delimiter=',')
correctPredictions = np.genfromtxt('results_static_experiment/correct-predictions.csv', delimiter=',')
errors = np.genfromtxt('results_static_experiment/TDErrors.csv', delimiter=',')
motorActive = np.genfromtxt('results_static_experiment/voluntaryActivation.csv', delimiter=',')
actions = np.genfromtxt('results_static_experiment/actions.csv', delimiter=',')

N = 50

# plot points
plt.subplot(3,3,1)
plt.scatter(results[N-1:,1],runningMean(results[:,0],N))
# plot as line
# plt.subplot(3,3,2)
# plt.plot(x,y)
plt.title('rewards')

plt.subplot(3,3,2)
x = range(np.size(errors)-N+1)
plt.plot(x[:], runningMean(errors[:],N))
plt.title('errors')

# actionSpace = ['click', 'right', 'left', 'top', 'bottom', 'none']
plt.subplot(3,3,3)
x = range(np.size(actions))
plt.bar(x, actions, color="blue")
plt.title('click|right|left|top|bottom|none')

x = range(np.size(neuralActivations,0)-N+1)
# plt.plot(x, neuralActivations[:,0],
#          x, neuralActivations[:,1],
#          x, neuralActivations[:,2])
plt.subplot(3,3,4)
plt.plot(x[:], runningMean(neuralActivations[:,0], N))
plt.title('neural activations L4')
# plt.subplot(3,3,5)
# plt.plot(x[:], runningMean(neuralActivations[:,1], N))
# plt.title('neural activations L5')
plt.subplot(3,3,5)
plt.plot(x[:], runningMean(neuralActivations[:,2], N))
plt.title('neural activations D1')
plt.subplot(3,3,6)
plt.plot(x, runningMean(correctPredictions[:,2],N))
plt.title('correct predictions D1')


# plt.subplot(3,3,3)
# x = range(np.size(neuralPredictions,0))
# plt.plot(x, neuralPredictions[:,0],
#          x, neuralPredictions[:,1],
#          x, neuralPredictions[:,2])
# plt.title('neural predictions')

x = range(np.size(correctPredictions,0)-N+1)
plt.subplot(3,3,7)
plt.plot(x, runningMean(correctPredictions[:,0],N))
plt.title('correct predictions L4')
plt.subplot(3,3,8)
plt.plot(x, runningMean(correctPredictions[:,1],N))
plt.title('correct predictions L5')
# plt.subplot(3,3,9)
# plt.plot(x, neuralPredictions[:,2])
# plt.title('neural predictions D1')

plt.subplot(3,3,9)
x = range(np.size(motorActive)-N+1)
plt.plot(x, runningMean(motorActive[:], N))
plt.title('voluntary active motor cells')


plt.show()
