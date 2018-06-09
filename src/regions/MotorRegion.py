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
import numpy

from nupic.bindings.regions.PyRegion import PyRegion

from algorithms.apical_distal_motor_memory import MotorTM
from htmresearch.algorithms.temporal_memory_factory import createModel

#DEBUG
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



class MRegion(PyRegion):
  """
  Motor Region based using the apical_distal_motor_memory implementation.

  Apical connections come from D1 and D2. If a cell is depolarized by
  distal and apical connections it will excite/inhibit the motor neurons
  depending on the apical connection respectively.

  Apical permanences are learned using the TD-Error calculated in D1/D2 Region.
  The output is a set of active motor neurons. It will rank them on excitation
  level and choose the NUM_WINNERS highest to become active. The parameter is
  defined at region initialization.


  """

  @classmethod
  def getSpec(cls):
    """
    Return the Spec for MRegion.
    """
    spec = dict(
      description=MRegion.__doc__,
      singleNodeOnly=True,
      inputs=dict(
        TDErrorD1=dict(
            description="The TDError calculated at D1 used for learning the apical connections",
            dataType='Real32',
            count=1,
            required=True,
            regionLevel=True,
            isDefaultInput=False,
            requireSplitterMap=False),
        TDErrorD2=dict(
            description="The TDError calculated at D2 used for learning the apical connections",
            dataType='Real32',
            count=1,
            required=True,
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

        activeColumns=dict(
          description="An array of 0's and 1's representing active columns."
                      " It is a copy of the active columns from L5.",
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        depolarizedBasalCells=dict(
          description="An array of 0's and 1's representing external input."
                      " It is a copy of the depolarized cells from L5."
                      " (Depolarized through basal input from L2/L4)",
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        activeCells=dict(
          description="List of active cells from layer 5 needed "
                      " to learn apical connections from Motor-Layer5. ",
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        apicalInputD1=dict(
          description="An array of 0's and 1's representing top down input from D1."
                      " The input will be used for culculating motor excitation.",
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        apicalInputD2=dict(
          description="An array of 0's and 1's representing top down input from D2."
                      " The input will be used for calculating motor inhibition.",
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

      ),
      outputs=dict(

        voluntaryActiveCells=dict(
          description="A binary output containing a 1 for every"
                      "cell which was depolarized by basal *and* apical input",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False),

        winnerCells=dict(
          description=("A binary output containing a 1 for every "
                       "'winner' cell after choosing the K-Winners."),
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False),

        depolarizedApicalCells=dict(
          description="A binary output containing a 1 for every"
                      " cell that was depolarized from the apical connections.",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False),

        activeMotorCells=dict(
          description="A binary output containing a 1 for every"
                      " motor cells that got excited/inhibited.",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False),

      ),

      parameters=dict(
        motorCount=dict(
          description="Number of neurons in the motor layer.",
          accessMode="Read",
          dataType="UInt32",
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
        cellsPerColumn=dict(
          description="Number of cells per column",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        apicalGlobalDecay=dict(
          description='Global Decay for all apical synapses.',
          accessMode='Read',
          dataType='Real32',
          count=1,
          constraints=''),
        TDLearningRate=dict(
          description='Learning rate applied to permanences with TD error',
          accessMode='Read',
          dataType='Real32',
          count=1,
          constraints=''),
        winnerSize=dict(
          description='Amount of winner cells chosen for motor behavior representation.',
          accessMode='Read',
          dataType='UInt32',
          count=1,
          constraints=''),
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
        activationThreshold=dict(
          description="If the number of active connected synapses on a "
                      "segment is at least this threshold, the segment "
                      "is said to be active.",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        synPermActiveIncMotor=dict(
          description="Amount by which permanences of L5-Motor synapses are incremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        synPermInactiveDecMotor=dict(
          description="Amount by which permanences of L5-Motor synapses are incremented during learning.",
          accessMode="Read",
          dataType="Real32",
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
        maxSynapsesPerSegment=dict(
          description="The maximum number of synapses per segment",
          accessMode="Read",
          dataType="UInt32",
          count=1),
        punishPredDec=dict(
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
      ),
      commands=dict(
        reset=dict(description="Explicitly reset TM states now."),
      )
    )

    return spec


  def __init__(self,

               # Modified TM params
               basalInputWidth=0,
               apicalInputWidth=0,
               apicalGlobalDecay=0.000001,
               TDLearningRate=0.5,
               winnerSize=4,
               motorCount=32,

               synPermActiveIncMotor=0.04,
               synPermInactiveDecMotor=0.008,

               # TM params
               columnCount=2048,
               cellsPerColumn=32,
               activationThreshold=13,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               minThreshold=10,
               maxNewSynapseCount=20,
               punishPredDec=0.0,
               maxSynapsesPerSegment=255,
               seed=42,

               # Region params
               learn=True,
               **kwargs):

    # Input sizes (the network API doesn't provide these during initialize)
    self.columnCount = columnCount
    self.basalInputWidth = basalInputWidth
    self.apicalInputWidth = apicalInputWidth

    # TM params
    self.columnCount = columnCount
    self.cellsPerColumn = cellsPerColumn
    self.basalInputWidth = basalInputWidth
    self.apicalInputWidth = apicalInputWidth
    self.activationThreshold = activationThreshold
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.minThreshold = minThreshold
    self.maxSynapsesPerSegment = maxSynapsesPerSegment
    self.seed = seed

    # Motor specific
    self.TDLearningRate = TDLearningRate
    self.apicalGlobalDecay = apicalGlobalDecay
    self.winnerSize = winnerSize
    self.punishPredDec = punishPredDec
    self.motorCount = motorCount
    self.synPermActiveIncMotor = synPermActiveIncMotor
    self.synPermInactiveDecMotor = synPermInactiveDecMotor

    # Region params
    self.learn = learn

    PyRegion.__init__(self, **kwargs)

    # TM instance
    self._tm = None


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
        "activationThreshold": self.activationThreshold,
        "synPermActiveIncMotor":self.synPermActiveIncMotor,
        "synPermInactiveDecMotor":self.synPermInactiveDecMotor,
        "initialPermanence": self.initialPermanence,
        "connectedPermanence": self.connectedPermanence,
        "punishPredDec":self.punishPredDec,
        "minThreshold": self.minThreshold,
        "maxSynapsesPerSegment": self.maxSynapsesPerSegment,
        "seed": self.seed,
        "TDLearningRate":self.TDLearningRate,
        "apicalGlobalDecay":self.apicalGlobalDecay,
        "winnerSize":self.winnerSize,
        "motorCount": self.motorCount,
      }
      self._tm = MotorTM(**params);


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
        outputs["voluntaryActiveCells"][:] = 0
        outputs["depolarizedApicalCells"][:] = 0
        outputs["activeMotorCells"][:] = 0
        outputs["winnerCells"][:] = 0
        return

    activeColumns = inputs["activeColumns"].nonzero()[0]
    activeCells = inputs["activeCells"].nonzero()[0]
    depolarizedBasalCells = inputs["depolarizedBasalCells"].nonzero()[0]
    apicalInputD1 = inputs["apicalInputD1"].nonzero()[0]
    apicalInputD2 = inputs["apicalInputD2"].nonzero()[0]
    TDError = inputs["TDErrorD1"][0]

    # ~~~~~~~~~~~~~~~ Generate Motor Behavior ~~~~~~~~~~~~~~~
    # calculate intersection of cells depolarized from basal input and D1/D2
    # Run the TM for one time step.
    self._tm.compute(
      activeColumns=activeColumns,
      depolarizedBasalCells=depolarizedBasalCells,
      activeCells=activeCells,
      apicalInputD1=apicalInputD1,
      apicalInputD2=apicalInputD2,
      TDError=TDError,
      learn=self.learn)

    # Extract the voluntary active cells (basal&apical depolarized) and put them into binary arrays.
    outputs["voluntaryActiveCells"][:] = 0
    outputs["voluntaryActiveCells"][self._tm.getVoluntaryActiveCells()] = 1

    outputs["depolarizedApicalCells"][:] = 0
    outputs["depolarizedApicalCells"][self._tm.getDepolarizedApicalCells()] = 1
    outputs["activeMotorCells"][:] = 0
    outputs["activeMotorCells"][self._tm.getActiveMotorCells()] = 1

    # Winner cells determine action
    outputs["winnerCells"][:] = 0
    outputs["winnerCells"][self._tm.getWinnerCells()] = 1

    # DEBUG print
    print '~~~~~ Motor Summary ~~~~~'
    print '[Motor TM] D1 Apical Input total size', len(inputs['apicalInputD1'])
    print '[Motor] Motor cells', self._tm.getMotorCells()
    print '[Motor] Depolarized basal input from L5(t)', depolarizedBasalCells
    print '[Motor] Depolarized Basal cells', self._tm.getDepolarizedBasalCells()
    print '[Motor] Depolarized Apical cells', self._tm.getDepolarizedApicalCells()
    if self._tm.getActiveMotorCells().any():
        print bcolors.WARNING
        print '[Motor] Active cells (voluntary, apical&basal Depolarized)', self._tm.getVoluntaryActiveCells()
        print '[Motor] Excited/Inhibited cell motor-values', self._tm.getMotorCells()[self._tm.getActiveMotorCells()]
        print bcolors.ENDC
    print '[Motor] Winner cells', self._tm.getWinnerCells()

  def reset(self):
    """ Reset the state of the TM """
    if self._tm is not None:
      self._tm.reset()


  def debugPlot(self, name):
    self._tm.mmGetCellActivityPlot(activityType="voluntaryActiveCells",
                                   showReset=True,
                                   resetShading=0.75,
                                   title="ac-{name}".format(name=name))
    self._tm.mmGetCellActivityPlot(activityType="depolarizedApicalCells",
                                   showReset=True,
                                   resetShading=0.75,
                                   title="p1-{name}".format(name=name))
    self._tm.mmGetCellActivityPlot(activityType="activeMotorCells",
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
    if name in ["activeMotorCells", "depolarizedApicalCells", "voluntaryActiveCells",
                "winnerCells"]:
      return self.cellsPerColumn * self.columnCount
    else:
      raise Exception("Invalid output name specified: %s" % name)

  def getMotorCells(self):
    """
    Return the excitation values for the motor cells of the current iteration.
    """
    return self._tm.getMotorCells()

  def getWinnerCells(self):
    """
    Return the k-winner cells with the highest excitation for the current iteration. (k is given winnerSize parameter)
    """
    return self._tm.getWinnerCells()

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

    :param proto: MotorRegion capnproto object
    """

    proto.learn = bool(self.learn)

    # TM instance
    self._tm.write(proto.motorTM)


  @classmethod
  def readFromProto(cls, proto):
    """
    Overrides :meth:`~nupic.bindings.regions.PyRegion.PyRegion.readFromProto`.

    Read state to proto object. The algorithm instance is serialized separately.

    :param proto: MotorRegion capnproto object
    """
    instance = cls.__new__(cls)
    instance.learn = proto.learn

    instance._tm = MotorTM.read(proto.motorTM)

    return instance
