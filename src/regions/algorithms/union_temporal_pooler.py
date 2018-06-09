# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import random
import copy
import numpy
from nupic.bindings.algorithms import SpatialPooler
# Uncomment below line to use python SP
# from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.bindings.math import (SM32 as SparseMatrix,
                                 SM_01_32_32 as SparseBinaryMatrix,
                                 GetNTAReal,
                                 Random as NupicRandom)
from htmresearch.frameworks.union_temporal_pooling.activation.excite_functions.excite_functions_all import (
  LogisticExciteFunction, FixedExciteFunction, LinearExciteFunction)

from htmresearch.frameworks.union_temporal_pooling.activation.decay_functions.decay_functions_all import (
  ExponentialDecayFunction, NoDecayFunction, LinearDecayFunction)


REAL_DTYPE = GetNTAReal()
UINT_DTYPE = "uint32"
_TIE_BREAKER_FACTOR = 0.000001



class UnionTemporalPooler(SpatialPooler):
  """
  Experimental Union Temporal Pooler Python implementation. The Union Temporal
  Pooler builds a "union SDR" of the most recent sets of active columns. It is
  driven by active-cell input and, more strongly, by predictive-active cell
  input. The latter is more likely to produce active columns. Such winning
  columns will also tend to persist longer in the union SDR.
  """


  def __init__(self,
               # union_temporal_pooler.py parameters
               activeOverlapWeight=1.0,
               predictedActiveOverlapWeight=0.0,
               maxUnionActivity=0.20,
               exciteFunctionType='Fixed',
               decayFunctionType='NoDecay',
               decayTimeConst=20.0,
               decayLinearConst=1.0,
               synPermPredActiveInc=0.0,
               synPermPreviousPredActiveInc=0.0,
               historyLength=0,
               minHistory=0,
               **kwargs):
    """
    Please see spatial_pooler.py in NuPIC for super class parameter
    descriptions.

    Class-specific parameters:
    -------------------------------------

    @param activeOverlapWeight: A multiplicative weight applied to
        the overlap between connected synapses and active-cell input

    @param predictedActiveOverlapWeight: A multiplicative weight applied to
        the overlap between connected synapses and predicted-active-cell input

    @param fixedPoolingActivationBurst: A Boolean, which, if True, has the
        Union Temporal Pooler grant a fixed amount of pooling activation to
        columns whenever they win the inhibition step. If False, columns'
        pooling activation is calculated based on their current overlap.

    @param exciteFunction: If fixedPoolingActivationBurst is False,
        this specifies the ExciteFunctionBase used to excite pooling
        activation.

    @param decayFunction: Specifies the DecayFunctionBase used to decay pooling
        activation.

    @param maxUnionActivity: Maximum sparsity of the union SDR

    @param decayTimeConst: Time constant for the decay function

    @param decayLinearConst: Linear decay rate for decay function

    @param minHistory don't perform union (output all zeros) until buffer
    length >= minHistory
    """

    super(UnionTemporalPooler, self).__init__(**kwargs)

    self._activeOverlapWeight = activeOverlapWeight
    self._predictedActiveOverlapWeight = predictedActiveOverlapWeight
    self._maxUnionActivity = maxUnionActivity

    self._synPermPredActiveInc = synPermPredActiveInc
    self._synPermPreviousPredActiveInc = synPermPreviousPredActiveInc

    self._historyLength = historyLength
    self._minHistory = minHistory

    self._exciteFunctionType = exciteFunctionType
    self._decayFunctionType = decayFunctionType
    self._decayLinearConst = decayLinearConst
    self._decayTimeConst = decayTimeConst
    self._initFunctions();

    # The maximum number of cells allowed in a single union SDR
    self._maxUnionCells = int(self.getNumColumns() * self._maxUnionActivity)

    # Scalar activation of potential union SDR cells; most active cells become
    # the union SDR
    self._poolingActivation = numpy.zeros(self.getNumColumns(), dtype=REAL_DTYPE)

    # include a small amount of tie-breaker when sorting pooling activation
    numpy.random.seed(1)
    self._poolingActivation_tieBreaker = numpy.random.randn(self.getNumColumns()) * _TIE_BREAKER_FACTOR

    # time since last pooling activation increment
    # initialized to be a large number
    self._poolingTimer = numpy.ones(self.getNumColumns(), dtype=REAL_DTYPE) * 1000

    # pooling activation level after the latest update, used for sigmoid decay function
    self._poolingActivationInitLevel = numpy.zeros(self.getNumColumns(), dtype=REAL_DTYPE)

    # Current union SDR; the output of the union pooler algorithm
    self._unionSDR = numpy.array([], dtype=UINT_DTYPE)

    # Indices of active cells from spatial pooler
    self._activeCells = numpy.array([], dtype=UINT_DTYPE)

    # lowest possible pooling activation level
    self._poolingActivationlowerBound = 0.1

    self._preActiveInput = numpy.zeros(self.getNumInputs(), dtype=REAL_DTYPE)
    # predicted inputs from the last n steps
    self._prePredictedActiveInput = numpy.zeros((self.getNumInputs(), self._historyLength), dtype=REAL_DTYPE)

  def _initFunctions(self):
      # initialize excite/decay functions
      if self._exciteFunctionType == 'Fixed':
        self._exciteFunction = FixedExciteFunction()
      elif self._exciteFunctionType == 'Logistic':
        self._exciteFunction = LogisticExciteFunction()
      elif self._exciteFunctionType == 'Linear':
        self._exciteFunction = LinearExciteFunction()
      else:
        raise NotImplementedError('unknown excite function type'+exciteFunctionType)

      if self._decayFunctionType == 'NoDecay':
        self._decayFunction = NoDecayFunction()
      elif self._decayFunctionType == 'Exponential':
        self._decayFunction = ExponentialDecayFunction(self._decayTimeConst)
      elif self._decayFunctionType == 'Linear':
        self._decayFunction = LinearDecayFunction(self._decayLinearConst)
      else:
        raise NotImplementedError('unknown decay function type'+decayFunctionType)


  def reset(self):
    """
    Reset the state of the Union Temporal Pooler.
    """

    # Reset Union Temporal Pooler fields
    self._poolingActivation = numpy.zeros(self.getNumColumns(), dtype=REAL_DTYPE)
    self._unionSDR = numpy.array([], dtype=UINT_DTYPE)
    self._poolingTimer = numpy.ones(self.getNumColumns(), dtype=REAL_DTYPE) * 1000
    self._poolingActivationInitLevel = numpy.zeros(self.getNumColumns(), dtype=REAL_DTYPE)
    self._preActiveInput = numpy.zeros(self.getNumInputs(), dtype=REAL_DTYPE)
    self._prePredictedActiveInput = numpy.zeros((self.getNumInputs(), self._historyLength), dtype=REAL_DTYPE)

    # Reset Spatial Pooler fields
    self.setOverlapDutyCycles(numpy.zeros(self.getNumColumns(), dtype=REAL_DTYPE))
    self.setActiveDutyCycles(numpy.zeros(self.getNumColumns(), dtype=REAL_DTYPE))
    self.setMinOverlapDutyCycles(numpy.zeros(self.getNumColumns(), dtype=REAL_DTYPE))
    self.setBoostFactors(numpy.ones(self.getNumColumns(), dtype=REAL_DTYPE))


  def compute(self, activeInput, predictedActiveInput, learn):
    """
    Computes one cycle of the Union Temporal Pooler algorithm.
    @param activeInput            (numpy array) A numpy array of 0's and 1's that comprises the input to the union pooler
    @param predictedActiveInput   (numpy array) A numpy array of 0's and 1's that comprises the correctly predicted input to the union pooler
    @param learn                  (boolen)      A boolen value indicating whether learning should be performed
    """
    assert numpy.size(activeInput) == self.getNumInputs()
    assert numpy.size(predictedActiveInput) == self.getNumInputs()
    self._updateBookeepingVars(learn)

    # Compute proximal dendrite overlaps with active and active-predicted inputs
    overlapsActive = self._calculateOverlap(activeInput)
    overlapsPredictedActive = self._calculateOverlap(predictedActiveInput)
    totalOverlap = (overlapsActive * self._activeOverlapWeight +
                    overlapsPredictedActive *
                    self._predictedActiveOverlapWeight).astype(REAL_DTYPE)

    if learn:
      boostFactors = numpy.zeros(self.getNumColumns(), dtype=REAL_DTYPE)
      self.getBoostFactors(boostFactors)
      boostedOverlaps = boostFactors * totalOverlap
    else:
      boostedOverlaps = totalOverlap

    activeCells = self._inhibitColumns(boostedOverlaps)
    self._activeCells = activeCells

    # Decrement pooling activation of all cells
    self._decayPoolingActivation()

    # Update the poolingActivation of current active Union Temporal Pooler cells
    self._addToPoolingActivation(activeCells, totalOverlap)

    # update union SDR
    self._getMostActiveCells()

    if learn:
      # adapt permanence of connections from predicted active inputs to newly active cell
      # This step is the spatial pooler learning rule, applied only to the predictedActiveInput
      # Todo: should we also include unpredicted active input in this step?
      self._adaptSynapses(predictedActiveInput, activeCells, self.getSynPermActiveInc(), self.getSynPermInactiveDec())

      # Increase permanence of connections from predicted active inputs to cells in the union SDR
      # This is Hebbian learning applied to the current time step
      self._adaptSynapses(predictedActiveInput, self._unionSDR, self._synPermPredActiveInc, 0.0)

      # adapt permenence of connections from previously predicted inputs to newly active cells
      # This is a reinforcement learning rule that considers previous input to the current cell
      for i in xrange(self._historyLength):
        self._adaptSynapses(self._prePredictedActiveInput[:,i], activeCells, self._synPermPreviousPredActiveInc, 0.0)

      # Homeostasis learning inherited from the spatial pooler
      self._updateDutyCycles(totalOverlap.astype(UINT_DTYPE), activeCells)
      self._bumpUpWeakColumns()
      self._updateBoostFactors()
      if self._isUpdateRound():
        self._updateInhibitionRadius()
        self._updateMinDutyCycles()

    # save inputs from the previous time step
    self._preActiveInput = copy.copy(activeInput)
    self._prePredictedActiveInput = numpy.roll(self._prePredictedActiveInput,1,1)
    if self._historyLength > 0:
      self._prePredictedActiveInput[:, 0] = predictedActiveInput

    return self._unionSDR


  def _decayPoolingActivation(self):
    """
    Decrements pooling activation of all cells
    """
    if self._decayFunctionType == 'NoDecay':
      self._poolingActivation = self._decayFunction.decay(self._poolingActivation)
    elif self._decayFunctionType == 'Exponential':
      self._poolingActivation = self._decayFunction.decay(\
                                self._poolingActivationInitLevel, self._poolingTimer)

    return self._poolingActivation


  def _addToPoolingActivation(self, activeCells, overlaps):
    """
    Adds overlaps from specified active cells to cells' pooling
    activation.
    @param activeCells: Indices of those cells winning the inhibition step
    @param overlaps: A current set of overlap values for each cell
    @return current pooling activation
    """
    self._poolingActivation[activeCells] = self._exciteFunction.excite(
      self._poolingActivation[activeCells], overlaps[activeCells])

    # increase pooling timers for all cells
    self._poolingTimer[self._poolingTimer >= 0] += 1

    # reset pooling timer for active cells
    self._poolingTimer[activeCells] = 0
    self._poolingActivationInitLevel[activeCells] = self._poolingActivation[activeCells]

    return self._poolingActivation


  def _getMostActiveCells(self):
    """
    Gets the most active cells in the Union SDR having at least non-zero
    activation in sorted order.
    @return: a list of cell indices
    """
    poolingActivation = self._poolingActivation
    nonZeroCells = numpy.argwhere(poolingActivation > 0)[:,0]

    # include a tie-breaker before sorting
    poolingActivationSubset = poolingActivation[nonZeroCells] + \
                              self._poolingActivation_tieBreaker[nonZeroCells]
    potentialUnionSDR = nonZeroCells[numpy.argsort(poolingActivationSubset)[::-1]]

    topCells = potentialUnionSDR[0: self._maxUnionCells]

    if max(self._poolingTimer) > self._minHistory:
      self._unionSDR = numpy.sort(topCells).astype(UINT_DTYPE)
    else:
      self._unionSDR = []

    return self._unionSDR


  # overide
  def _adaptSynapses(self, inputVector, activeColumns, synPermActiveInc, synPermInactiveDec):
    """
    The primary method in charge of learning. Adapts the permanence values of
    the synapses based on the input vector, and the chosen columns after
    inhibition round. Permanence values are increased for synapses connected to
    input bits that are turned on, and decreased for synapses connected to
    inputs bits that are turned off.

    Parameters:
    ----------------------------
    @param inputVector:
                    A numpy array of 0's and 1's that comprises the input to
                    the spatial pooler. There exists an entry in the array
                    for every input bit.
    @param activeColumns:
                    An array containing the indices of the columns that
                    survived inhibition.

    @param synPermActiveInc:
                    Permanence increment for active inputs
    @param synPermInactiveDec:
                    Permanence decrement for inactive inputs
    """
    inputIndices = numpy.where(inputVector > 0)[0]
    permChanges = numpy.zeros(self.getNumInputs(), dtype=REAL_DTYPE)
    permChanges.fill(-1 * synPermInactiveDec)
    permChanges[inputIndices] = synPermActiveInc
    perm = numpy.zeros(self.getNumInputs(), dtype=REAL_DTYPE)
    potential = numpy.zeros(self.getNumInputs(), dtype=REAL_DTYPE)
    for i in activeColumns:
      self.getPermanence(i, perm)
      self.getPotential(i, potential)
      maskPotential = numpy.where(potential > 0)[0]
      perm[maskPotential] += permChanges[maskPotential]
      self._updatePermanencesForColumn(perm, i, raisePerm=False)


  def getUnionSDR(self):
    return self._unionSDR

  def write(self, proto):
    """
    Override the Spatial pooler serialization for union pooler.
    """
    # SpatialPooler
    super(UnionTemporalPooler, self).write(proto);

    # UnionTemporalPooler
    proto.activeOverlapWeight = float(self._activeOverlapWeight)
    proto.predictedActiveOverlapWeight = float(self._predictedActiveOverlapWeight)
    proto.maxUnionActivity = float(self._maxUnionActivity)
    proto.synPermPredActiveInc = float(self._synPermPredActiveInc)
    proto.synPermPreviousPredActiveInc = float(self._synPermPreviousPredActiveInc)

    proto.historyLength = int(self._historyLength)
    proto.minHistory = int(self._minHistory)

    proto.exciteFunctionType = self._exciteFunctionType
    proto.decayFunctionType = self._decayFunctionType
    proto.decayTimeConst = self._decayTimeConst
    proto.decayLinearConst = self._decayLinearConst

    proto.maxUnionCells = int(self._maxUnionCells)

    poolingActivationProto = proto.init("poolingActivation", len(self._poolingActivation))
    for i, cell in enumerate(self._poolingActivation):
        poolingActivationProto[i] = float(cell)

    tieBreakerProto = proto.init("poolingActivationTieBreaker", len(self._poolingActivation_tieBreaker))
    for i, cell in enumerate(self._poolingActivation_tieBreaker):
        tieBreakerProto[i] = float(cell)

    poolingTimerProto = proto.init("poolingTimer", len(self._poolingTimer))
    for i, cell in enumerate(self._poolingTimer):
        poolingTimerProto[i] = float(cell)

    initProto = proto.init("poolingActivationInitLevel", len(self._poolingActivationInitLevel))
    for i, cell in enumerate(self._poolingActivationInitLevel):
        initProto[i] = float(cell)

    proto.poolingActivationLowerBound = float(self._poolingActivationlowerBound)

    unionProto = proto.init("unionSDR", len(self._unionSDR))
    for i, cell in enumerate(self._unionSDR):
        unionProto[i] = int(cell)

    activeCellsProto = proto.init("activeCells", len(self._activeCells))
    for i, cell in enumerate(self._activeCells):
        activeCellsProto[i] = int(cell)

    preActiveProto = proto.init("preActiveInput", len(self._preActiveInput))
    for i, cell in enumerate(self._preActiveInput):
        preActiveProto[i] = float(cell)

    prePredActiveProto = proto.init("prePredictedActiveInput", len(self._prePredictedActiveInput))
    for i, row in enumerate(self._prePredictedActiveInput):
        predictedActiveInput = prePredActiveProto.init(i, len(row))
        for j, cell in enumerate(row):
            predictedActiveInput[j] = float(cell)

# ----------------------------------------------
# Deserialization
# ----------------------------------------------

class _SparseMatrixCorticalColumnAdapter(object):
  """ Many functions in SpatialPooler operate on a columnIndex but use an
  underlying storage implementation based on a Sparse Matrix in which cortical
  columns are represented as rows.  This can be confusing to someone trying to
  follow the algorithm, confusing terminology between matrix math and cortical
  columns.  This class is provided to abstract away some of the details of the
  underlying implementation, providing a cleaner API that isn't specific to
  sparse matrices.
  """

  def __getitem__(self, columnIndex):
    """ Wraps getRow() such that instances may be indexed by columnIndex."""
    return super(_SparseMatrixCorticalColumnAdapter, self).getRow(columnIndex)


  def replace(self, columnIndex, bitmap):
    """ Wraps replaceSparseRow()"""
    return super(_SparseMatrixCorticalColumnAdapter, self).replaceSparseRow(
      columnIndex, bitmap
    )


  def update(self, columnIndex, vector):
    """ Wraps setRowFromDense()"""
    return super(_SparseMatrixCorticalColumnAdapter, self).setRowFromDense(
      columnIndex, vector
    )

class CorticalColumns(_SparseMatrixCorticalColumnAdapter, SparseMatrix):
  """ SparseMatrix variant of _SparseMatrixCorticalColumnAdapter.  Use in cases
  where column connections are represented as float values, such as permanence
  values
  """
  pass


class BinaryCorticalColumns(_SparseMatrixCorticalColumnAdapter,
                            SparseBinaryMatrix):
  """ SparseBinaryMatrix variant of _SparseMatrixCorticalColumnAdapter.  Use in
  cases where column connections are represented as bitmaps.
  """
  pass

  @classmethod
  def read(cls, proto):
    """
    Override the Spatial pooler deserialization for union pooler.
    """

    instance = cls.__new__(cls)
    # Spatial Pooler (copied)
    VERSION = 3
    EPSILON_ROUND = 5

    instance._random = NupicRandom()
    instance._random.read(proto.random)
    instance._numInputs = proto.numInputs
    instance._numColumns = proto.numColumns
    instance._columnDimensions = numpy.array(proto.columnDimensions)
    instance._inputDimensions = numpy.array(proto.inputDimensions)
    instance._potentialRadius = proto.potentialRadius
    instance._potentialPct = round(proto.potentialPct,
                                   EPSILON_ROUND)
    instance._inhibitionRadius = proto.inhibitionRadius
    instance._globalInhibition = proto.globalInhibition
    instance._numActiveColumnsPerInhArea = proto.numActiveColumnsPerInhArea
    instance._localAreaDensity = proto.localAreaDensity
    instance._stimulusThreshold = proto.stimulusThreshold
    instance._synPermInactiveDec = round(proto.synPermInactiveDec,
                                         EPSILON_ROUND)
    instance._synPermActiveInc = round(proto.synPermActiveInc, EPSILON_ROUND)
    instance._synPermBelowStimulusInc = round(proto.synPermBelowStimulusInc,
                                              EPSILON_ROUND)
    instance._synPermConnected = round(proto.synPermConnected,
                                       EPSILON_ROUND)
    instance._minPctOverlapDutyCycles = round(proto.minPctOverlapDutyCycles,
                                              EPSILON_ROUND)
    instance._dutyCyclePeriod = proto.dutyCyclePeriod
    instance._boostStrength = proto.boostStrength
    instance._wrapAround = proto.wrapAround
    instance._spVerbosity = proto.spVerbosity

    instance._synPermMin = proto.synPermMin
    instance._synPermMax = proto.synPermMax
    instance._synPermTrimThreshold = round(proto.synPermTrimThreshold,
                                           EPSILON_ROUND)

    # TODO: These two overlaps attributes aren't currently saved.
    instance._overlaps = numpy.zeros(numColumns, dtype=realDType)
    instance._boostedOverlaps = numpy.zeros(numColumns, dtype=realDType)

    instance._updatePeriod = proto.updatePeriod

    instance._version = VERSION
    instance._iterationNum = proto.iterationNum
    instance._iterationLearnNum = proto.iterationLearnNum

    instance._potentialPools = BinaryCorticalColumns(numInputs)
    instance._potentialPools.resize(numColumns, numInputs)
    instance._potentialPools.read(proto.potentialPools)

    instance._permanences = CorticalColumns(numColumns, numInputs)
    instance._permanences.read(proto.permanences)
    # Initialize ephemerals and make sure they get updated
    instance._connectedCounts = numpy.zeros(numColumns, dtype=realDType)
    instance._connectedSynapses = BinaryCorticalColumns(numInputs)
    instance._connectedSynapses.resize(numColumns, numInputs)
    for columnIndex in xrange(proto.numColumns):
      instance._updatePermanencesForColumn(
        instance._permanences[columnIndex], columnIndex, False
      )

    instance._tieBreaker = numpy.array(proto.tieBreaker, dtype=realDType)

    instance._overlapDutyCycles = numpy.array(proto.overlapDutyCycles,
                                          dtype=realDType)
    instance._activeDutyCycles = numpy.array(proto.activeDutyCycles,
                                         dtype=realDType)
    instance._minOverlapDutyCycles = numpy.array(proto.minOverlapDutyCycles,
                                             dtype=realDType)
    instance._boostFactors = numpy.array(proto.boostFactors, dtype=realDType)


    # Union Temporal Pooler
    instance._activeOverlapWeight = proto.activeOverlapWeight
    instance._predictedActiveOverlapWeight = proto.predictedActiveOverlapWeight
    instance._maxUnionActivity = proto.maxUnionActivity
    instance._synPermPredActiveInc = proto.synPermPredActiveInc
    instance._synPermPreviousPredActiveInc = proto.synPermPreviousPredActiveInc

    instance._historyLength = proto.historyLength
    instance._minHistory = proto.minHistory

    instance._decayFunctionType = proto.exciteFunctionType
    instance._decayFunctionType = proto.decayFunctionType
    instance._decayTimeConst = proto.decayTimeConst
    instance._decayLinearConst = proto.decayLinearConst
    instance._initFunctions();

    instance._maxUnionCells = proto.maxUnionCells

    instance._poolingActivation = numpy.array(proto.poolingActivation, dtype=REAL_DTYPE)
    instance._poolingActivation_tieBreaker = numpy.array(proto.poolingActivationTieBreaker, dtype=REAL_DTYPE)

    instance._poolingTimer = numpy.array(proto.poolingTimer, dtype=REAL_DTYPE)
    instance._poolingActivationInitLevel = numpy.array(proto.poolingActivationInitLevel, dtype=REAL_DTYPE)
    instance._poolingActivationlowerBound = proto.poolingActivationLowerBound

    instance._unionSDR = numpy.array(proto.unionSDR, dtype='uint32')
    instance._activeCells = numpy.array(proto.activeCells, dtype='uint32')

    instance._preActiveInput = numpy.array(proto.preActiveInput, dtype=REAL_DTYPE)

    # load matrix
    instance._prePredictedActiveInput = [[] for _ in xrange(len(proto.prePredictedActiveInput))]
    for row, rowProto in zip(instance._prePredictedActiveInput, proto.prePredictedActiveInput):
        row = numpy.array(rowProto, dtype=REAL_DTYPE)

    return instance
