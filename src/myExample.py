import gym
import universe # register the universe environments
import numpy as np
import os
import time
import datetime
import matplotlib.pyplot as plt
import csv

from network import createNetwork, saveNetwork, loadNetwork
from config.modelParams import MOTOR_PARAMS, SENSOR_PARAMS

# TODO:
#       - Serialization -> SparseMatrixConnections in MotorTM (how?)
#       - TD Error error goes to infinity (forum post - check decay function etc.)
#       - Test if FOCUS helps recognize mouse movements better (Include mouse coords)
#       - Make encoder nicer (with coordinates, mouse parameters etc)
#       - Motor Layer: Learning apical D1-L5 and L5-Motor use same threshold, connected, initial perm etc.
#       - GitHub pullrequest for HTM-Research apical_distal_motor_memory is not punishing uncorrectly active segments

#       - RESET signal correct order -> do not interpret reward with reseted state or
#           confuse as prevActive in reinforce is also reset and will effect TD-Calculation
#       - Why activationThreshold/minThreshold so low in ETM layer 5 and D1,D2?


# [STATS] Save results and network
NETWORK_DIR_NAME = "networks"
SAVED_NETWORK_PATH = NETWORK_DIR_NAME + "/2018_06_03_17_59_11_agent_net.nta"


def randCoords():
    radius = SENSOR_PARAMS['radius'];
    return {
        'x': np.random.randint(0+radius, 160-radius) + 10,
        'y': np.random.randint(0+radius, 160-radius) + 75 + 50
    };

def init():
  """ Initialize parameters for the program

  """

  # initialize global variables
  radius = SENSOR_PARAMS['radius'];

  #(TODO: refractor global to params)
  global boundary
  global actionSpace
  global currentAction
  global coords
  global actionsTotal
  global RUN
  boundary = {    # 5 pixel from the bounds
      'x': (10+radius,170-radius),
      'y': (125+radius, 285-radius) # only bottom row
  };

  actionSpace = ['click', 'right', 'left', 'top', 'bottom', 'none'];
  currentAction = 'initial';
  coords = randCoords(); # mouse coordinates - Initially random coordinates in the agents visual field

  # (TODO: Add support for random reset) Determines if mouse cursor is randomly inialized after sequence end
  # global RANDOM_RESET = False;

  # Initialize for DEBUG
  actionsTotal = [0,0,0,0,0,0]
  RUN = 0

  # Create RESULTS_DIR
  if not os.path.exists('results'):
      os.makedirs('results');
  datetimestr = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S");
  RESULTS_DIR_NAME = 'results/%s' % datetimestr;
  os.makedirs(RESULTS_DIR_NAME);


def selectAction(winnerCells, motorSize):
  """
  Selects the action from the given obseration and the output motor cells.

  :param winnerCells: The indicies of the chosen winner cells for this iteration
  :param motorSize: The size of motorCells, gives the range for the winner indicies
  """

  # map winnerRange constantly to one of the actions
  actions = np.zeros(len(actionSpace));
  for winner in winnerCells:
      translatedIndex = translate(winner, 0, motorSize, 0, len(actionSpace))
      # sum over active motorcells and which action they belong to
      actions[translatedIndex] += 1
  # Take the action with maximum active motor cells
  print 'Take max from mapping .. ', actions
  actions_max = np.random.choice(np.flatnonzero(actions == actions.max()))

  # Map selected action to environment command
  global currentAction  # TODO: Make nicer than global
  global actionsTotal
  currentAction = actionSpace[actions_max]
  actionsTotal[actions_max] += 1
  if currentAction == 'click':
      action = click()
  elif currentAction == 'none':
      action = []
  else:
      action = move(currentAction)

  return action

def translate(value, leftMin, leftMax, rightMin, rightMax):
    """
    Function to map a range to another
    """
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return int(rightMin + (valueScaled * rightSpan))

def click():
    # 1. move to x,y with left button released, and click there (2. and 3.)
    return [universe.spaces.PointerEvent(coords['x'],coords['y'], 0),
            universe.spaces.PointerEvent(coords['x'],coords['y'], 1),
            universe.spaces.PointerEvent(coords['x'],coords['y'], 0)]

def move(direction):
    """
      Move the mouse pointer without clicking.
      Only discrete moving (left, right, top, bottom) but with an increment
      of 10px each time so every character is reachable.

      It ensures that the mouse stays always in the 160x160 agent window
      from the received (768,1024,3) uint8 screen from the environment.

      The browser window indents the origin of MiniWob by 75 pixels from top and
      10 pixels from the left. The first 50 pixels along height are the query.
    """
    # Get x,y
    xcoord = coords['x']
    ycoord = coords['y']
    stepSize = 15   # how many pixels to move

    # Discrete action space
    if direction == 'right':
        xcoord = xcoord + stepSize
        xcoord = min(xcoord, boundary['x'][1])
    elif direction == 'left':
        xcoord = xcoord - stepSize
        xcoord = max(xcoord, boundary['x'][0])
    elif direction == 'top':
        ycoord = ycoord - stepSize
        ycoord = max(ycoord, boundary['y'][0])
    else:   # bottom
        ycoord = ycoord + stepSize
        ycoord = min(ycoord, boundary['y'][1])

    # Set x,y
    coords['x'] = xcoord
    coords['y'] = ycoord

    # Move to x,y with left button released
    return [universe.spaces.PointerEvent(xcoord, ycoord, 0)]


def printDebugPlot(layers):
    # Load radius from params
    radius = SENSOR_PARAMS['radius'];
    plt.ion();

    # Sensor layer
    plt.subplot(3,2,1)
    if radius>0:
        sensor = np.reshape(layers['sensor'].encoder.getState()[0:160*160], (160,160))
        focus = np.reshape(layers['sensor'].encoder.getState()[160*160:], (radius*2,radius*2))
        plt.imshow(sensor)
        plt.subplot(3,2,2)
        plt.imshow(focus)
    else:
        plt.imshow(np.reshape(layers['sensor'].encoder.getState(), (160,160)))
    # show when using receptive field
    #plt.imshow(np.reshape(layers['sensor'].encoder.getState(), (SENSOR_PARAMS['radius']*2,SENSOR_PARAMS['radius']*2)))

    # Plot L4_TM activations
    ax1 = plt.subplot(3,2,3)
    ax1.title.set_text('Layer 4')
    active_L4 = np.zeros(512*32)
    active_L4[layers['L4_TM']._tm.getActiveCells()] = 1
    plt.imshow(np.reshape(active_L4, (32, 512)), aspect='auto')

    # Plot L5_TM
    ax2 = plt.subplot(3,2,4)
    ax2.title.set_text('Layer 5')
    active_L5 = np.zeros(512*32)
    active_L5[layers['L5_TM']._tm.getActiveCells()] = 1
    plt.imshow(np.reshape(active_L5, (32, 512)), aspect='auto')

    # Plot D1
    ax3 = plt.subplot(3,2,5)
    ax3.title.set_text('Layer D1')
    active_D1 = np.zeros(512*32)
    active_D1[layers['D1_TM']._tm.getActiveCells()] = 1
    plt.imshow(np.reshape(active_D1, (32, 512)), aspect='auto')

    # Barchart about activity
    # L4
    # plt.subplot(3,2,2)
    # active_L4 = len(layers['L4_TM']._tm.getActiveCells())
    # depolarized_L4 = len(layers['L4_TM']._tm.getPredictiveCells())
    # plt.bar(range(2), [active_L4, depolarized_L4])
    #
    # # L4
    # plt.subplot(3,2,3)
    # active_L5 = len(layers['L5_TM']._tm.getActiveCells())
    # depolarized_L5 = len(layers['L5_TM']._tm.getPredictiveCells())
    # plt.bar(range(2), [active_L5, depolarized_L5])

    plt.show()
    plt.pause(0.0001)

def saveResults(results, filename):
    # open in binary mode to append
    filename = '%s/%s' % RESULTS_DIR_NAME,filename
    f = open(filename, 'ab')
    np.savetxt(f, results , delimiter=',')
    f.close()

def trainNetwork(env, net, layers, iterations):
    """
    Train the agent on the given environment.

    :param env: Environment to train the agent in
    :param net: Network object
    :param layers: Access to the internal layers of the network
    :param iterations: Defines for how many step iterations it should run
    """

    # initial step
    observation_n, reward_n, done_n, info = env.step([[]])

    # run until first observation (skip environment initialization)
    while not observation_n or not observation_n[0]:
        observation_n, reward_n, done_n, info = env.step([[]])


    # save results -> rewards:iterations
    results = []
    neuralActivations = []  # L4, L5, D1
    neuralPredictions = [] # L4, L5, D1
    correctPredictions = []
    errors = []
    motorActive = []

    # train the network x iterations (Could be changed to time)
    for i in range(0,iterations):
      if observation_n and observation_n[0]:

          # run Network
          sensedValue = {
            "observation": observation_n[0]['vision'],
            "reward": reward_n,
            "mouse": coords,
            "coordinates": coords,
            "done": False, #(done_n and RANDOM_RESET) # If task end - reset sequence states
          }
          layers['sensor'].setSensedValue(sensedValue);
          net.run(1);

          # read winner cells
          winnerCells = layers['Motor'].getWinnerCells()
          action = selectAction(winnerCells, MOTOR_PARAMS['motorCount'])

          # save results
          if reward_n[0] != 0:
              results.append(reward_n[0])
              results.append(i + RUN*iterations)
          neuralActivation = [len(layers['L4_TM']._tm.getActiveCells()),
                              len(layers['L5_TM']._tm.getActiveCells()),
                              len(layers['D1_TM']._tm.getActiveCells())]
          neuralPrediction = [len(layers['L4_TM']._tm.getPredictiveCells()),
                              len(layers['L5_TM']._tm.getPredictiveCells()),
                              len(layers['D1_TM']._tm.getPredictiveCells())]
          correctPrediction = [len(np.intersect1d(layers['L4_TM']._tm.getPredictiveCells(),layers['L4_TM']._tm.getActiveCells())),
                              len(np.intersect1d(layers['L5_TM']._tm.getPredictiveCells(),layers['L5_TM']._tm.getActiveCells())),
                              len(np.intersect1d(layers['D1_TM']._tm.getPredictiveCells(),layers['D1_TM']._tm.getActiveCells()))]
          neuralActivations.append(neuralActivation)
          neuralPredictions.append(neuralPrediction)
          correctPredictions.append(correctPrediction)
          errors.append(layers['D1_TM'].getTDError())
          motorActive.append(len(layers['Motor']._tm.getVoluntaryActiveCells()))

        #   print DEBUG
        #   if i> 980:
        #       printDebugPlot(layers);
        #   printDebugPlot(layers);
          #time.sleep(0.5)
      else:
          print 'Environment did not process observation.'
          action = []   # no observation rendered

      # Render environment
      observation_n, reward_n, done_n, info = env.step([action])
      env.render()

      # DEBUG Printout
      print '------------------------- DEBUG SUMMARY -------------------------'
      print "Reward_n", reward_n
    #   Debug why reward not [-1,1]
    #   if (reward_n[0] < (-1)):
    #     time.sleep(10)
      print "done_n", done_n
      print 'WINNER CELLS', winnerCells, 'in range', MOTOR_PARAMS['motorCount']
      print 'Selected Action', currentAction

    results = np.reshape(results, newshape=(len(results)/2,2))
    saveResults(results, 'results.csv');
    saveResults(neuralActivations, 'neural-activations.csv');
    saveResults(neuralPredictions, 'neural-predictions.csv');
    saveResults(correctPredictions, 'correct-predictions.csv');
    saveResults(errors, 'TDErrors.csv');
    saveResults(motorActive, 'voluntaryActivation.csv')
    # update clicks (not append)
    f = open('results/actions.csv', 'wb')
    np.savetxt(f, actionsTotal , delimiter=',')
    f.close()

    global RUN
    RUN += 1;


def runDemo():
    # Setup environment
    #env = gym.make('wob.mini.AnExperiment-v0')
    env = gym.make('wob.mini.ClickColor-v0')
    #env = gym.make('wob.mini.AnEasyExperiment-v0')
    #env = gym.make('wob.mini.CharacterExp-v0')
    #env = gym.make('wob.mini.ChaseCircle-v0')

    env.configure(remotes=1, fps=5,
                  vnc_driver='go',
                  vnc_kwargs={'encoding': 'tight', 'compress_level': 0,
                              'fine_quality_level': 100, 'subsample_level': 0})

    # Initialize network
    if not os.path.exists(NETWORK_DIR_NAME) or not os.path.isfile(SAVED_NETWORK_PATH):
        print 'Initialize a new network'
        (net, layers) = createNetwork();
        net.initialize();
    else:
        print 'Loading network from %s', SAVED_NETWORK_PATH
        (net, layers) = loadNetwork(SAVED_NETWORK_PATH);

    # Starts new environment. As it runs in real time
    # and connects directly the initial observation will be None
    observation_n = env.reset()

    # Train network
    iterations = 1000      # 60frames/sec is theoretical maximum from env.
    while True:
        trainNetwork(env, net, layers, iterations)

    # Save network
    print 'Saving network on filesystem'
    saved_path = saveNetwork(layers)
    (net,layers) = loadNetwork(saved_path)

    iterations = 20      # 60frames/sec is theoretical maximum from env.
    trainNetwork(env, net, layers, iterations)

    # Print some stats

if __name__ == "__main__":
  init();
  runDemo();
