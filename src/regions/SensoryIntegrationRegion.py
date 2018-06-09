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

from algorithms.apical_weighted_temporal_memory import ApicalWeightedTemporalMemory

# deserialization
from htmresearch_core.experimental import ExtendedTemporalMemory



class SensoryIntegrationRegion(PyRegion):
  """
  The SensoryIntegrationRegion based on ExtendedTMRegion with light modifications. (original documentation)

  It makes basal connections internally and to the motor basal cells.
  The motor basal cells are weighted differently as they are much less cell activity but important.

  The ExtendedTMRegion implements temporal memory for the HTM network API.

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
    Return the Spec for SensoryIntegrationRegion.
    """
    spec = dict(
      description=SensoryIntegrationRegion.__doc__,
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
                      " segments from the Motor region",
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

      ),
      outputs=dict(

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
        activationThresholdBasal=dict(
          description="If the number of active connected basal synapses on a "
                      "segment is at least this threshold, the segment "
                      "is said to be active.",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        minThresholdBasal=dict(
          description="If the number of basal synapses active on a segment is at "
                      "least this threshold, it is selected as the best "
                      "matching cell in a bursting column.",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        basalPredictedSegmentDecrement=dict(
          description="Amount by which active permanences of basal synapses of "
                      "previously predicted but inactive segments are "
                      "decremented.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        activationThresholdApical=dict(
          description="If the number of active connected apical synapses on a "
                      "segment is at least this threshold, the segment "
                      "is said to be active.",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        minThresholdApical=dict(
          description="If the number of apical synapses active on a segment is at "
                      "least this threshold, it is selected as the best "
                      "matching cell in a bursting column.",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        apicalPredictedSegmentDecrement=dict(
          description="Amount by which active permanences of apical synapses of "
                      "previously predicted but inactive segments are "
                      "decremented.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        learn=dict(
          description="True if the node is learning (default true).",
          accessMode="ReadWrite",
          dataType="Bool",
          count=1,
          defaultValue="true"),
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
        seed=dict(
          description="Seed for the random number generator.",
          accessMode="Read",
          dataType="UInt32",
          count=1),
        implementation=dict(
          description="ETM implementation",
          accessMode="Read",
          dataType="Byte",
          count=0,
          constraints=("enum: etm, monitored_etm"),
          defaultValue="py"),
        sampleSize=dict(
          description="How much of the active SDR to sample with synapses.",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          defaultValue="20"),
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

               # ETM params
               cellsPerColumn=32,

               initialPermanence=0.21,
               connectedPermanence=0.50,
               permanenceIncrement=0.10,
               permanenceDecrement=0.10,
               predictedSegmentDecrement=0.0,
               maxSynapsesPerSegment=255,
               seed=42,
               sampleSize=20,

               # apical, basal weighting
               activationThresholdBasal=13,
               activationThresholdApical=2,
               minThresholdBasal=10,
               minThresholdApical=1,
               basalPredictedSegmentDecrement=0.001,
               apicalPredictedSegmentDecrement=0.001,

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
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.predictedSegmentDecrement = predictedSegmentDecrement
    self.maxSynapsesPerSegment = maxSynapsesPerSegment
    self.seed = seed
    self.sampleSize = sampleSize

    # TM weight distal and apical differently
    self.minThresholdBasal = minThresholdBasal
    self.minThresholdApical = minThresholdApical
    self.activationThresholdBasal = activationThresholdBasal
    self.activationThresholdApical = activationThresholdApical
    self.basalPredictedSegmentDecrement = basalPredictedSegmentDecrement
    self.apicalPredictedSegmentDecrement = apicalPredictedSegmentDecrement


    # Region params
    self.implementation = implementation
    self.learn = learn

    PyRegion.__init__(self, **kwargs)

    # TM instance
    self._tm = None

    # Custom use internal activation t-1 as basal input
    self.prevActivation = np.array([])


  def initialize(self):
    """
    Initialize the self._tm if not already initialized. We need to figure out
    the constructor parameters for each class, and send it to that constructor.
    """
    if self._tm is None:
      params = {
        "columnCount": self.columnCount,
        "cellsPerColumn": self.cellsPerColumn,
        "basalInputSize": self.basalInputWidth,
        "apicalInputSize": self.apicalInputWidth,
        "sampleSize": self.sampleSize,

        # new for weighted version
        "activationThresholdBasal": self.activationThresholdBasal,
        "minThresholdBasal": self.minThresholdBasal,
        "activationThresholdApical": self.activationThresholdApical,
        "minThresholdApical": self.minThresholdApical,
        "basalPredictedSegmentDecrement": self.basalPredictedSegmentDecrement,
        "apicalPredictedSegmentDecrement": self.apicalPredictedSegmentDecrement,

        "permanenceIncrement": self.permanenceIncrement,
        "permanenceDecrement": self.permanenceDecrement,
        "initialPermanence": self.initialPermanence,
        "connectedPermanence": self.connectedPermanence,
        "maxSynapsesPerSegment": self.maxSynapsesPerSegment,
        "seed": self.seed,
      }
      self._tm = ApicalWeightedTemporalMemory(**params);

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

    # Use prev. L4(t-1) for basal (context interpreting)
    if self.prevActivation.any():
      activeCellsExternalBasal = self.prevActivation.nonzero()[0]
    else:
      activeCellsExternalBasal = ()

    # Motor input for apical input
    if "apicalInput" in inputs:
      activeCellsExternalApical = inputs["apicalInput"].nonzero()[0]
    else:
      activeCellsExternalApical = ()


    # Run the TM for one time step.
    self._tm.compute(activeColumns,
                     activeCellsExternalBasal,
                     activeCellsExternalApical,
                     basalGrowthCandidates=None,
                     apicalGrowthCandidates=None,
                     learn=self.learn)

    # Extract the active / predicted cells and put them into binary arrays.
    outputs["activeCells"][:] = 0
    outputs["activeCells"][self._tm.getActiveCells()] = 1
    self.prevActivation = outputs["activeCells"][:]
    outputs["predictedCells"][:] = 0
    outputs["predictedCells"][self._tm.getPredictiveCells()] = 1
    outputs["predictedActiveCells"][:] = (outputs["activeCells"] *
                                          outputs["predictedCells"])
    outputs["winnerCells"][:] = 0
    outputs["winnerCells"][self._tm.getWinnerCells()] = 1

    # Temporary debug output
    print '~~~~~~~ L4 ~~~~~~~~'
    print 'Proximal input (columns)', activeColumns
    print 'Distal input', activeCellsExternalBasal
    print 'Active cells', self._tm.getActiveCells()
    print 'Predicted cells', self._tm.getPredictiveCells()
    print '~~~~~~~~~~~~~~~~~~~~~~~~~'


  def reset(self):
    """ Reset the state of the TM """
    if self._tm is not None:
      self._tm.reset()
    self.prevActivation = ()


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
