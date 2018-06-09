# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015-2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import copy
import numpy as np

from nupic.bindings.regions.PyRegion import PyRegion

from htmresearch.algorithms.temporal_memory_factory import createModel

# deserialization
from htmresearch_core.experimental import ExtendedTemporalMemory



#DEBUG
from pprint import pprint
import time
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ReinforcementRegion(PyRegion):
  """
  The ReinforcementRegion extends the ExtendedTMRegion with TD-Error computation.
  Documentation is mostly taken from TM-Region. Used for D1 simulation.

  The ExtendedTMRegion's computation implementations come from the
  nupic.research class ExtendedTemporalMemory.

  The region supports external basal and apical inputs.

  The main difference between the ExtendedTMRegion and the TMRegion is that the
  ExtendedTMRegion uses the basal / apical input to predict cells for the
  current time step, while the TMRegion uses them to predict cells for the next
  time step. The ExtendedTMRegion can't output predictions for the next input.
  """

  @classmethod
  def getSpec(cls):
    """
    Return the Spec for ExtendedTMRegion.
    """
    spec = dict(
      description=ReinforcementRegion.__doc__,
      singleNodeOnly=True,
      inputs=dict(
        activeColumns=dict(
          description=("An array of 0's and 1's representing the active "
                       "minicolumns, i.e. the input to the TemporalMemory"),
          dataType="Real32",
          count=0,
          required=True,
          regionLevel=True,
          isDefaultInput=True,
          requireSplitterMap=False),

        reward=dict(
          description="Reward signal for TD-Learning this iteration.",
          dataType='Real32',
          count=1,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        resetIn=dict(
          description="A boolean flag that indicates whether"
                      " or not the input vector received in this compute cycle"
                      " represents the first presentation in a"
                      " new temporal sequence.",
          dataType='Real32',
          count=1,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        basalInput=dict(
          description="An array of 0's and 1's representing external input"
                      " such as motor commands that are available to basal"
                      " segments",
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        apicalInput=dict(
          description="An array of 0's and 1's representing top down input."
                      " The input will be provided to apical dendrites.",
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

      ),
      outputs=dict(
        TDError=dict(
          description="TDError for this iteration.",
          dataType="Real32",
          count=1,
          regionLevel=True,
          isDefaultOutput=False),

        predictedCells=dict(
          description="A binary output containing a 1 for every"
                      " cell that was predicted for this timestep.",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False),

        predictedActiveCells=dict(
          description="A binary output containing a 1 for every"
                      " cell that transitioned from predicted to active.",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False),

        activeCells=dict(
          description="A binary output containing a 1 for every"
                      " cell that is currently active.",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False),

        winnerCells=dict(
          description=("A binary output containing a 1 for every "
                       "'winner' cell in the TM."),
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False),

      ),

      parameters=dict(
        learn=dict(
          description="True if the node is learning (default true).",
          accessMode="ReadWrite",
          dataType="Bool",
          count=1,
          defaultValue="true"),
        TDLearningRate=dict(
          description="TD Learning rate used how fast new state values"
                      "are learned in the update of neuron values.",
          dataType="Real32",
          accessMode="Read",
          count=1,
          constraints=""),
        TDTraceDecay=dict(
          description="TD Trace Decay used in eligability traces computation.",
          dataType="Real32",
          accessMode="Read",
          count=1,
          constraints=""),
        TDDiscount=dict(
          description="TD Discount used in TD-Error computation.",
          dataType="Real32",
          accessMode="Read",
          count=1,
          constraints=""),
        globalValueDecay=dict(
          description="Global decay applied to neuron values to prevent infinite growth."
                      "If 0.01 then each neuron value would be decayed one percent each iteration.",
          dataType="Real32",
          accessMode="Read",
          count=1,
          constraints=""),
        columnCount=dict(
          description="Number of columns in this temporal memory",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        columnDimensions=dict(
          description="Number of colums in this temporal memory (vector"
                      " version).",
          dataType="Real32",
          accessMode="Read",
          count=0,
          regionLevel=True,
          isDefaultOutput=True),
        basalInputWidth=dict(
          description='Number of basal inputs to the TM.',
          accessMode='Read',
          dataType='UInt32',
          count=1,
          constraints=''),
        apicalInputWidth=dict(
          description='Number of apical inputs to the TM.',
          accessMode='Read',
          dataType='UInt32',
          count=1,
          constraints=''),
        cellsPerColumn=dict(
          description="Number of cells per column",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        activationThreshold=dict(
          description="If the number of active connected synapses on a "
                      "segment is at least this threshold, the segment "
                      "is said to be active.",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        initialPermanence=dict(
          description="Initial permanence of a new synapse.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        connectedPermanence=dict(
          description="If the permanence value for a synapse is greater "
                      "than this value, it is said to be connected.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        minThreshold=dict(
          description="If the number of synapses active on a segment is at "
                      "least this threshold, it is selected as the best "
                      "matching cell in a bursting column.",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        maxNewSynapseCount=dict(
          description="The maximum number of synapses added to a segment "
                      "during learning.",
          accessMode="Read",
          dataType="UInt32",
          count=1),
        maxSegmentsPerCell=dict(
          description="The maximum number of segments per cell",
          accessMode="Read",
          dataType="UInt32",
          count=1),
        maxSynapsesPerSegment=dict(
          description="The maximum number of synapses per segment",
          accessMode="Read",
          dataType="UInt32",
          count=1),
        permanenceIncrement=dict(
          description="Amount by which permanences of synapses are "
                      "incremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        permanenceDecrement=dict(
          description="Amount by which permanences of synapses are "
                      "decremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        predictedSegmentDecrement=dict(
          description="Amount by which active permanences of synapses of "
                      "previously predicted but inactive segments are "
                      "decremented.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        seed=dict(
          description="Seed for the random number generator.",
          accessMode="Read",
          dataType="UInt32",
          count=1),
        formInternalBasalConnections=dict(
          description="Flag to determine whether to form basal connections "
                      "with internal cells within this temporal memory",
          accessMode="Read",
          dataType="Bool",
          count=1,
          defaultValue="true"),
        learnOnOneCell=dict(
          description="If True, the winner cell for each column will be"
                      " fixed between resets.",
          accessMode="Read",
          dataType="Bool",
          count=1,
          defaultValue="false"),
        implementation=dict(
          description="ETM implementation",
          accessMode="Read",
          dataType="Byte",
          count=0,
          constraints=("enum: etm, monitored_etm"),
          defaultValue="py"),
        checkInputs=dict(
          description="Set to False to disable input checking (for speed-up)",
          accessMode="ReadWrite",
          dataType="Bool",
          count=1,
          defaultValue="true"),
      ),
      commands=dict(
        reset=dict(description="Explicitly reset TM states now."),
      )
    )

    return spec


  def __init__(self,

               # Modified ETM params
               columnCount=2048,
               basalInputWidth=0,
               apicalInputWidth=0,
               TDTraceDecay=0.0,
               TDDiscount=0.0,
               TDLearningRate=0.1,
               globalValueDecay=0.0,

               # ETM params
               cellsPerColumn=32,
               activationThreshold=13,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               minThreshold=10,
               maxNewSynapseCount=20,
               permanenceIncrement=0.10,
               permanenceDecrement=0.10,
               predictedSegmentDecrement=0.0,
               formInternalBasalConnections=True,
               learnOnOneCell=False,
               maxSegmentsPerCell=255,
               maxSynapsesPerSegment=255,
               seed=42,
               checkInputs=True,

               # Region params
               implementation="etm",
               learn=True,
               **kwargs):

    # Input sizes (the network API doesn't provide these during initialize)
    self.columnCount = columnCount
    self.basalInputWidth = basalInputWidth
    self.apicalInputWidth = apicalInputWidth

    # TM params
    self.cellsPerColumn = cellsPerColumn
    self.activationThreshold = activationThreshold
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.minThreshold = minThreshold
    self.maxNewSynapseCount = maxNewSynapseCount
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.predictedSegmentDecrement = predictedSegmentDecrement
    self.formInternalBasalConnections = formInternalBasalConnections
    self.learnOnOneCell = learnOnOneCell
    self.maxSegmentsPerCell = maxSegmentsPerCell
    self.maxSynapsesPerSegment = maxSynapsesPerSegment
    self.seed = seed
    self.checkInputs = checkInputs

    # Region params
    self.implementation = implementation
    self.learn = learn

    PyRegion.__init__(self, **kwargs)

    # TM instance
    self._tm = None

    # Reinforcement Learning variables (Eligability traces and values for each neuron)
    self.TDDiscount = TDDiscount
    self.TDLearningRate = TDLearningRate
    self.TDTraceDecay = TDTraceDecay
    self.globalValueDecay = globalValueDecay
    self.traces = np.zeros(columnCount*cellsPerColumn)
    self.values = np.zeros(columnCount*cellsPerColumn)
    self.stateValue = 0
    self.prevActiveCells = []
    # Save prev. distal input for calculation with L5(t-1) distal input
    self.prevActiveCellsExternalBasal = []
    # For Debug save errors
    self.TDError = 0

  def initialize(self):
    """
    Initialize the self._tm if not already initialized. We need to figure out
    the constructor parameters for each class, and send it to that constructor.
    """
    if self._tm is None:
      params = {
        "columnDimensions": (self.columnCount,),
        "basalInputDimensions": (self.basalInputWidth,),
        "apicalInputDimensions": (self.apicalInputWidth,),
        "cellsPerColumn": self.cellsPerColumn,
        "activationThreshold": self.activationThreshold,
        "initialPermanence": self.initialPermanence,
        "connectedPermanence": self.connectedPermanence,
        "minThreshold": self.minThreshold,
        "maxNewSynapseCount": self.maxNewSynapseCount,
        "permanenceIncrement": self.permanenceIncrement,
        "permanenceDecrement": self.permanenceDecrement,
        "predictedSegmentDecrement": self.predictedSegmentDecrement,
        "formInternalBasalConnections": self.formInternalBasalConnections,
        "learnOnOneCell": self.learnOnOneCell,
        "maxSegmentsPerCell": self.maxSegmentsPerCell,
        "maxSynapsesPerSegment": self.maxSynapsesPerSegment,
        "seed": self.seed,
        "checkInputs": self.checkInputs,
      }
      self._tm = createModel(self.implementation, **params)


  def compute(self, inputs, outputs):
    """
    Run one iteration of TM's compute.

    Note that if the reset signal is True (1) we assume this iteration
    represents the *end* of a sequence. The output will contain the TM
    representation to this point and any history will then be reset. The output
    at the next compute will start fresh, presumably with bursting columns.
    """

    # Handle reset first (should be sent with an empty signal)
    if "resetIn" in inputs:
      assert len(inputs["resetIn"]) == 1
      if inputs["resetIn"][0] != 0:
        # send empty output
        self.reset()
        outputs["activeCells"][:] = 0
        outputs["predictedCells"][:] = 0
        outputs["predictedActiveCells"][:] = 0
        outputs["winnerCells"][:] = 0
        return

    activeColumns = inputs["activeColumns"].nonzero()[0]

    if "basalInput" in inputs:
      activeCellsExternalBasal = self.prevActiveCellsExternalBasal
      self.prevActiveCellsExternalBasal = inputs["basalInput"].nonzero()[0]
    else:
      activeCellsExternalBasal = ()

    if "apicalInput" in inputs:
      activeCellsExternalApical = inputs["apicalInput"].nonzero()[0]
    else:
      activeCellsExternalApical = ()

    # Run the TM for one time step.
    self._tm.depolarizeCells(
      activeCellsExternalBasal,
      activeCellsExternalApical,
      learn=self.learn)
    self._tm.activateCells(
      activeColumns,
      reinforceCandidatesExternalBasal=activeCellsExternalBasal,
      reinforceCandidatesExternalApical=activeCellsExternalApical,
      growthCandidatesExternalBasal=activeCellsExternalBasal,
      growthCandidatesExternalApical=activeCellsExternalApical,
      learn=self.learn)
    # Current neural activation
    activeCells = self._tm.getActiveCells()

    # TD learning
    reward = inputs["reward"][0]
    # Discount reward to be [-0.1,0.1] range
    reward = reward
    # - Eligability Trace decay
    self.traces = self.traces * self.TDDiscount * self.TDTraceDecay
    # - Calculate avg. state value (current estimates)
    prevStateValue = self.stateValue;
    self.stateValue = self._calculateStateValue(activeCells,
                                                self._tm.getPredictiveCells())
    # NOTE: Option NeuronValues/AverageError
    # - Calculate TD-Error from Estimate at time t and t+1
    #TDError = self._calculateError(reward, self.prevActiveCells, self.stateValue)
    self.TDError = self._calculateErrorAverage(reward, prevStateValue, self.stateValue);
    # - Replacing traces
    self.traces[activeCells] = 1
    # - Update State values of each neuron
    self._updateValues(self.TDError)
    # Save prev. neural activation for next iteration
    self.prevActiveCells = activeCells



    # Extract the active / predicted cells and put them into binary arrays.
    outputs["activeCells"][:] = 0
    outputs["activeCells"][activeCells] = 1
    outputs["predictedCells"][:] = 0
    outputs["predictedCells"][self._tm.getPredictiveCells()] = 1
    outputs["predictedActiveCells"][:] = (outputs["activeCells"] *
                                          outputs["predictedCells"])
    outputs["winnerCells"][:] = 0
    outputs["winnerCells"][self._tm.getWinnerCells()] = 1

    outputs["TDError"][:] = self.TDError

    print '[D1] Input SP indicies', inputs["activeColumns"].nonzero()[0]
    print '[D1] Basal Input L5 indicies', inputs["basalInput"].nonzero()[0]
    print '[D1] activeCells indicies', self._tm.getActiveCells()
    print '[D1] predictedCells indicies', self._tm.getPredictiveCells()


  def _updateValues(self, TDError):
      """ Update the values of all neurons depending on their activity.
          All neuron values together represent the current state.

          The learning is weighted depending on the eligability trace.
          This implies actions that were further in the past were less important.
      """
      self.values = self.values * (1-self.globalValueDecay)
      self.values = self.values + (self.TDLearningRate * TDError * self.traces)

  def _calculateError(self, reward, prevActiveCells, avgStateValue):
      """ Calculate the TD Error for the current timestep.
          It compares the average estimate (state value) of the current step (t+1)
          with every neuron value of the previous timestep and takes the average error.
      """

      # Calculate average error (filter prev. active cells for difference)
      TDErrors = reward + self.TDDiscount * (avgStateValue - self.values[prevActiveCells])
      if len(prevActiveCells)>0:
          avgError = sum(TDErrors) / len(prevActiveCells)
      else:
          avgError = 0

      # DEBUG
      print bcolors.FAIL
      print '[D1] New State Average', avgStateValue
      print '[D1] Reward', reward
      print '[D1] Avg. Error', avgError
      print bcolors.ENDC
    #   time.sleep(1)

      return avgError

  def _calculateErrorAverage(self, reward, prevStateValue, avgStateValue):
      """ Alternative way to calculate the TD error

          Calculate the TD Error for the current timestep.
          It compares the average estimate (state value) of the current step (t+1)
          with the average value from the previous timestep.
      """

      # Calculate average error (filter prev. active cells for difference)
      TDError = reward + self.TDDiscount * (avgStateValue - prevStateValue)

      # DEBUG
      print bcolors.FAIL
      print '[D1] Old State Average', prevStateValue
      print '[D1] New State Average', avgStateValue
      print '[D1] Reward', reward
      print '[D1] Avg. Error', TDError
      print bcolors.ENDC
    #   time.sleep(1)

      return TDError

  def _calculateStateValue(self, activeCells, predictedCells):
      """ Calculate the weighted average for all active neurons.
          Predicted neurons have a 10x factor, unpredicted (bursted) 1x.
          The result is the current state value.
      """
      factors = np.zeros(len(self.values))
      factors[activeCells] = 1
      activePredictedCells = np.intersect1d(activeCells,predictedCells)
      factors[activePredictedCells] = 1
      avgValue = sum(self.values * factors) / len(activeCells)
      return avgValue


  def reset(self):
    """ Reset the state of the TM """
    if self._tm is not None:
      self._tm.reset()
    self.traces = np.zeros(self.columnCount*self.cellsPerColumn)
    self.prevActiveCells = []
    self.prevActiveCellsExternalBasal = []

  def getTDError(self):
    """ Get TD error for debugging """
    return self.TDError


  def debugPlot(self, name):
    self._tm.mmGetCellActivityPlot(activityType="activeCells",
                                   showReset=True,
                                   resetShading=0.75,
                                   title="ac-{name}".format(name=name))
    self._tm.mmGetCellActivityPlot(activityType="predictedCells",
                                   showReset=True,
                                   resetShading=0.75,
                                   title="p1-{name}".format(name=name))
    self._tm.mmGetCellActivityPlot(activityType="predictedActiveCells",
                                   showReset=True,
                                   resetShading=0.75,
                                   title="pa-{name}".format(name=name))

  def getAlgorithmInstance(self):
    """
    :returns: temporal memory instance depending on the implementation parameter
              of the underlying algorithm object.
              Used for serialization of the network.
    """
    return self._tm

  def setAlgorithmInstance(self, instance):
    """
    :set instance of the underlying algorithm object.
    """
    self._tm = instance;

  def getParameter(self, parameterName, index=-1):
    """
      Get the value of a NodeSpec parameter. Most parameters are handled
      automatically by PyRegion's parameter get mechanism. The ones that need
      special treatment are explicitly handled here.
    """
    return PyRegion.getParameter(self, parameterName, index)


  def setParameter(self, parameterName, index, parameterValue):
    """
    Set the value of a Spec parameter. Most parameters are handled
    automatically by PyRegion's parameter set mechanism. The ones that need
    special treatment are explicitly handled here.
    """
    if hasattr(self, parameterName):
      setattr(self, parameterName, parameterValue)
    else:
      raise Exception("Unknown parameter: " + parameterName)


  def getOutputElementCount(self, name):
    """
    Return the number of elements for the given output.
    """
    if name in ["predictedActiveCells", "predictedCells", "activeCells",
                "winnerCells"]:
      return self.cellsPerColumn * self.columnCount
    else:
      raise Exception("Invalid output name specified: %s" % name)


  def prettyPrintTraces(self):
    if "mixin" in self.temporalImp.lower():
      print self._tm.mmPrettyPrintTraces([
        self._tm.mmGetTraceNumSegments(),
        self._tm.mmGetTraceNumSynapses(),
      ])


  def writeToProto(self, proto):
    """
    Overrides :meth:`~nupic.bindings.regions.PyRegion.PyRegion.writeToProto`.

    Write state to proto object. The algorithm instance is serialized separately.

    :param proto: ReinforcementRegionProto capnproto object
    """

    proto.implementation = self.implementation
    proto.learn = bool(self.learn)

    # TM instance
    self._tm.write(proto.etm)

    # Reinforcement Learning variables (Eligability traces and values for each neuron)
    proto.discount = float(self.TDDiscount)
    proto.learningRate = float(self.TDLearningRate)
    proto.traceDecay = float(self.TDTraceDecay)
    proto.stateValue = float(self.stateValue)

    # Trace and neuron values for TD-Error
    tracesProto = proto.init("traces", len(self.traces))
    for i, trace in enumerate(self.traces):
        tracesProto[i] = float(trace)
    valuesProto = proto.init("values", len(self.values))
    for i, value in enumerate(self.values):
        valuesProto[i] = float(value)

    # Previous indices for next calculation
    prevActiveCellsProto = proto.init("prevActiveCells", len(self.prevActiveCells))
    for i, cell in enumerate(self.prevActiveCells):
        prevActiveCellsProto[i] = int(cell)
    prevActiveCellsExternalBasalProto = proto.init("prevActiveCellsExternalBasal", len(self.prevActiveCellsExternalBasal))
    for i, cell in enumerate(self.prevActiveCellsExternalBasal):
        prevActiveCellsExternalBasalProto[i] = int(cell)

  @classmethod
  def readFromProto(cls, proto):
    """
    Overrides :meth:`~nupic.bindings.regions.PyRegion.PyRegion.readFromProto`.

    Read state to proto object. The algorithm instance is serialized separately.

    :param proto: ReinforcementRegionProto capnproto object
    """
    instance = cls.__new__(cls)

    instance.implementation = proto.implementation
    instance.learn = proto.learn

    # TODO: Only ETM implementation supported
    instance._tm = ExtendedTemporalMemory.read(proto.etm)

    instance.TDDiscount = proto.discount
    instance.TDLearningRate = proto.learningRate
    instance.TDTraceDecay = proto.traceDecay
    instance.stateValue = proto.stateValue

    instance.traces = np.array(proto.traces, dtype=np.float64)
    instance.values = np.array(proto.values, dtype=np.float64)

    instance.prevActiveCells = np.array(proto.prevActiveCells, dtype='uint32')
    instance.prevActiveCellsExternalBasal = np.array(proto.prevActiveCellsExternalBasal, dtype='uint32')

    return instance
