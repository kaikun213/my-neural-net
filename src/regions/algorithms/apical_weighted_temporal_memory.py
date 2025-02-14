# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

"""An implementation of ApicalWeightedTemporalMemory"""

import numpy as np

from htmresearch.support import numpy_helpers as np2
from nupic.bindings.math import Random, SparseMatrixConnections



class ApicalWeightedTemporalMemory(object):
  """
  An alternate approach to apical dendrites. Every cell SDR is specific to both
  the basal the apical input. However they have different weighting as they have different thresholds and permanence increments.

  For sequence memory, the result is that every sequence happens within a
  "world" which is specified by the apical input. Sequences are not shared
  between worlds.
  """

  def __init__(self,
               columnCount=2048,
               basalInputSize=0,
               apicalInputSize=0,
               cellsPerColumn=32,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               sampleSize=20,
               permanenceIncrement=0.1,
               permanenceDecrement=0.1,
               activationThresholdBasal=13,
               minThresholdBasal=10,
               activationThresholdApical=2,
               minThresholdApical=1,
               basalPredictedSegmentDecrement=0.0,
               apicalPredictedSegmentDecrement=0.0,
               maxSynapsesPerSegment=-1,
               seed=42):
    """
    @param columnCount (int)
    The number of minicolumns

    @param basalInputSize (sequence)
    The number of bits in the basal input

    @param apicalInputSize (int)
    The number of bits in the apical input

    @param cellsPerColumn (int)
    Number of cells per column

    @param initialPermanence (float)
    Initial permanence of a new synapse

    @param connectedPermanence (float)
    If the permanence value for a synapse is greater than this value, it is said
    to be connected.

    @param activationThresholdBasal (int)
    If the number of active connected basal synapses on a segment is at least this
    threshold, the segment is said to be active.

    @param activationThresholdApical (int)
    If the number of active connected apical synapses on a segment is at least this
    threshold, the segment is said to be active.

    @param minThresholdBasal (int)
    If the number of potential basal synapses active on a segment is at least this
    threshold, it is said to be "matching" and is eligible for learning.

    @param minThresholdApical (int)
    If the number of potential apical synapses active on a segment is at least this
    threshold, it is said to be "matching" and is eligible for learning.

    @param sampleSize (int)
    How much of the active SDR to sample with synapses.

    @param permanenceIncrement (float)
    Amount by which permanences of synapses are incremented during learning.

    @param permanenceDecrement (float)
    Amount by which permanences of synapses are decremented during learning.

    @param basalPredictedSegmentDecrement (float)
    Amount by which segments are punished for incorrect predictions.

    @param apicalPredictedSegmentDecrement (float)
    Amount by which segments are punished for incorrect predictions.

    @param maxSynapsesPerSegment
    The maximum number of synapses per segment.

    @param seed (int)
    Seed for the random number generator.
    """

    self.columnCount = columnCount
    self.cellsPerColumn = cellsPerColumn
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.sampleSize = sampleSize
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.maxSynapsesPerSegment = maxSynapsesPerSegment

    # weighted basal/apical params
    self.basalPredictedSegmentDecrement = basalPredictedSegmentDecrement
    self.apicalPredictedSegmentDecrement = apicalPredictedSegmentDecrement
    self.minThresholdBasal = minThresholdBasal
    self.minThresholdApical = minThresholdApical
    self.activationThresholdBasal = activationThresholdBasal
    self.activationThresholdApical = activationThresholdApical

    self.basalConnections = SparseMatrixConnections(columnCount*cellsPerColumn,
                                                    basalInputSize)
    self.apicalConnections = SparseMatrixConnections(columnCount*cellsPerColumn,
                                                     apicalInputSize)
    self.rng = Random(seed)
    self.activeCells = ()
    self.winnerCells = ()
    self.predictedCells = ()
    self.activeBasalSegments = ()
    self.activeApicalSegments = ()


  def reset(self):
    """
    Clear all cell and segment activity. This has no effect on the subsequent
    predictions or activity.
    """

    self.activeCells = ()
    self.winnerCells = ()
    self.predictedCells = ()
    self.activeBasalSegments = ()
    self.activeApicalSegments = ()


  def compute(self,
              activeColumns,
              basalInput,
              apicalInput,
              basalGrowthCandidates=None,
              apicalGrowthCandidates=None,
              learn=True):
    """
    Perform one timestep. Use the basal and apical input to form a set of
    predictions, then activate the specified columns.

    @param activeColumns (numpy array)
    List of active columns

    @param basalInput (numpy array)
    List of active input bits for the basal dendrite segments

    @param apicalInput (numpy array)
    List of active input bits for the apical dendrite segments

    @param basalGrowthCandidates (numpy array or None)
    List of bits that the active cells may grow new basal synapses to.
    If None, the basalInput is assumed to be growth candidates.

    @param apicalGrowthCandidates (numpy array or None)
    List of bits that the active cells may grow new apical synapses to
    If None, the apicalInput is assumed to be growth candidates.

    @param learn (bool)
    Whether to grow / reinforce / punish synapses
    """

    if basalGrowthCandidates is None:
      basalGrowthCandidates = basalInput

    if apicalGrowthCandidates is None:
      apicalGrowthCandidates = apicalInput

    # Calculate predictions for this timestep
    (activeBasalSegments,
     matchingBasalSegments,
     basalPotentialOverlaps) = self._calculateSegmentActivity(
       self.basalConnections, basalInput, self.connectedPermanence,
       self.activationThresholdBasal, self.minThresholdBasal)

    (activeApicalSegments,
     matchingApicalSegments,
     apicalPotentialOverlaps) = self._calculateSegmentActivity(
       self.apicalConnections, apicalInput, self.connectedPermanence,
       self.activationThresholdApical, self.minThresholdApical)

    # Take union of predictions
    predictedCells = np.union1d(
      self.basalConnections.mapSegmentsToCells(activeBasalSegments),
      self.apicalConnections.mapSegmentsToCells(activeApicalSegments))

    # Calculate active cells
    (correctPredictedCells,
     burstingColumns) = np2.setCompare(predictedCells, activeColumns,
                                       predictedCells / self.cellsPerColumn,
                                       rightMinusLeft=True)
    newActiveCells = np.concatenate((correctPredictedCells,
                                     np2.getAllCellsInColumns(
                                       burstingColumns, self.cellsPerColumn)))

    # Calculate learning
    (learningActiveBasalSegments,
     learningActiveApicalSegments,
     learningMatchingBasalSegments,
     learningMatchingApicalSegments,
     basalSegmentsToPunish,
     apicalSegmentsToPunish,
     newSegmentCells,
     learningCells) = self._calculateLearning(activeColumns,
                                              burstingColumns,
                                              correctPredictedCells,
                                              activeBasalSegments,
                                              activeApicalSegments,
                                              matchingBasalSegments,
                                              matchingApicalSegments,
                                              basalPotentialOverlaps,
                                              apicalPotentialOverlaps)

    if learn:
      # Learn on existing segments
      for learningSegments in (learningActiveBasalSegments,
                               learningMatchingBasalSegments):
        self._learn(self.basalConnections, self.rng, learningSegments,
                    basalInput, basalGrowthCandidates, basalPotentialOverlaps,
                    self.initialPermanence, self.sampleSize,
                    self.permanenceIncrement, self.permanenceDecrement,
                    self.maxSynapsesPerSegment)

      for learningSegments in (learningActiveApicalSegments,
                               learningMatchingApicalSegments):
        self._learn(self.apicalConnections, self.rng, learningSegments,
                    apicalInput, apicalGrowthCandidates,
                    apicalPotentialOverlaps, self.initialPermanence,
                    self.sampleSize, self.permanenceIncrement,
                    self.permanenceDecrement, self.maxSynapsesPerSegment)

      # Punish incorrect predictions
      if self.basalPredictedSegmentDecrement != 0.0:
        self.basalConnections.adjustActiveSynapses(
          basalSegmentsToPunish, basalInput,
          -self.basalPredictedSegmentDecrement)

      if self.apicalPredictedSegmentDecrement != 0.0:
        self.apicalConnections.adjustActiveSynapses(
          apicalSegmentsToPunish, apicalInput,
          -self.apicalPredictedSegmentDecrement)

      # Only grow segments if there is basal *and* apical input.
      if len(basalGrowthCandidates) > 0 and len(apicalGrowthCandidates) > 0:
        self._learnOnNewSegments(self.basalConnections, self.rng,
                                 newSegmentCells, basalGrowthCandidates,
                                 self.initialPermanence, self.sampleSize,
                                 self.maxSynapsesPerSegment)
        self._learnOnNewSegments(self.apicalConnections, self.rng,
                                 newSegmentCells, apicalGrowthCandidates,
                                 self.initialPermanence, self.sampleSize,
                                 self.maxSynapsesPerSegment)


    # Save the results
    newActiveCells.sort()
    learningCells.sort()
    self.activeCells = newActiveCells
    self.winnerCells = learningCells
    self.predictedCells = predictedCells
    self.activeBasalSegments = activeBasalSegments
    self.activeApicalSegments = activeApicalSegments


  def _calculateLearning(self,
                         activeColumns,
                         burstingColumns,
                         correctPredictedCells,
                         activeBasalSegments,
                         activeApicalSegments,
                         matchingBasalSegments,
                         matchingApicalSegments,
                         basalPotentialOverlaps,
                         apicalPotentialOverlaps):
    """
    Learning occurs on pairs of segments. Correctly predicted cells always have
    active basal and apical segments, and we learn on these segments. In
    bursting columns, we either learn on an existing segment pair, or we grow a
    new pair of segments.

    @param activeColumns (numpy array)
    @param burstingColumns (numpy array)
    @param correctPredictedCells (numpy array)
    @param activeBasalSegments (numpy array)
    @param activeApicalSegments (numpy array)
    @param matchingBasalSegments (numpy array)
    @param matchingApicalSegments (numpy array)
    @param basalPotentialOverlaps (numpy array)
    @param apicalPotentialOverlaps (numpy array)

    @return (tuple)
    - learningActiveBasalSegments (numpy array)
      Active basal segments on correct predicted cells

    - learningActiveApicalSegments (numpy array)
      Active apical segments on correct predicted cells

    - learningMatchingBasalSegments (numpy array)
      Matching basal segments selected for learning in bursting columns

    - learningMatchingApicalSegments (numpy array)
      Matching apical segments selected for learning in bursting columns

    - basalSegmentsToPunish (numpy array)
      Basal segments that should be punished for predicting an inactive column

    - apicalSegmentsToPunish (numpy array)
      Apical segments that should be punished for predicting an inactive column

    - newSegmentCells (numpy array)
      Cells in bursting columns that were selected to grow new segments

    - learningCells (numpy array)
      Every cell that has a learning segment or was selected to grow a segment
    """

    # Correctly predicted columns
    learningActiveBasalSegments = self.basalConnections.filterSegmentsByCell(
      activeBasalSegments, correctPredictedCells)
    learningActiveApicalSegments = self.apicalConnections.filterSegmentsByCell(
      activeApicalSegments, correctPredictedCells)

    # Bursting columns
    cellsForMatchingBasal = self.basalConnections.mapSegmentsToCells(
      matchingBasalSegments)
    cellsForMatchingApical = self.apicalConnections.mapSegmentsToCells(
      matchingApicalSegments)
    matchingCells = np.intersect1d(
      cellsForMatchingBasal, cellsForMatchingApical)

    (matchingCellsInBurstingColumns,
     burstingColumnsWithNoMatch) = np2.setCompare(
       matchingCells, burstingColumns, matchingCells / self.cellsPerColumn,
       rightMinusLeft=True)

    (learningMatchingBasalSegments,
     learningMatchingApicalSegments) = self._chooseBestSegmentPairPerColumn(
       matchingCellsInBurstingColumns, matchingBasalSegments,
       matchingApicalSegments, basalPotentialOverlaps, apicalPotentialOverlaps)
    newSegmentCells = self._getCellsWithFewestSegments(
      burstingColumnsWithNoMatch)

    # Incorrectly predicted columns
    if self.basalPredictedSegmentDecrement > 0.0:
      correctMatchingBasalMask = np.in1d(
        cellsForMatchingBasal / self.cellsPerColumn, activeColumns)
      basalSegmentsToPunish = matchingBasalSegments[~correctMatchingBasalMask]
    else:
      basalSegmentsToPunish = ()

    if self.apicalPredictedSegmentDecrement > 0.0:
      correctMatchingApicalMask = np.in1d(
        cellsForMatchingApical / self.cellsPerColumn, activeColumns)
      apicalSegmentsToPunish = matchingApicalSegments[~correctMatchingApicalMask]
    else:
      apicalSegmentsToPunish = ()

    # Make a list of every cell that is learning
    learningCells =  np.concatenate(
      (correctPredictedCells,
       self.basalConnections.mapSegmentsToCells(learningMatchingBasalSegments),
       newSegmentCells))

    return (learningActiveBasalSegments,
            learningActiveApicalSegments,
            learningMatchingBasalSegments,
            learningMatchingApicalSegments,
            basalSegmentsToPunish,
            apicalSegmentsToPunish,
            newSegmentCells,
            learningCells)


  @staticmethod
  def _calculateSegmentActivity(connections, activeInput, connectedPermanence,
                                activationThreshold, minThreshold):
    """
    Calculate the active and matching segments for this timestep.

    @param connections (SparseMatrixConnections)
    @param activeInput (numpy array)

    @return (tuple)
    - activeSegments (numpy array)
      Dendrite segments with enough active connected synapses to cause a
      dendritic spike

    - matchingSegments (numpy array)
      Dendrite segments with enough active potential synapses to be selected for
      learning in a bursting column

    - potentialOverlaps (numpy array)
      The number of active potential synapses for each segment.
      Includes counts for active, matching, and nonmatching segments.
    """

    # Active
    overlaps = connections.computeActivity(activeInput, connectedPermanence)
    activeSegments = np.flatnonzero(overlaps >= activationThreshold)

    # Matching
    potentialOverlaps = connections.computeActivity(activeInput)
    matchingSegments = np.flatnonzero(potentialOverlaps >= minThreshold)

    return (activeSegments,
            matchingSegments,
            potentialOverlaps)


  @staticmethod
  def _learn(connections, rng, learningSegments, activeInput, growthCandidates,
             potentialOverlaps, initialPermanence, sampleSize,
             permanenceIncrement, permanenceDecrement, maxSynapsesPerSegment):
    """
    Adjust synapse permanences, and grow new synapses.

    @param learningActiveSegments (numpy array)
    @param learningMatchingSegments (numpy array)
    @param activeInput (numpy array)
    @param growthCandidates (numpy array)
    @param potentialOverlaps (numpy array)
    """

    # Learn on existing segments
    connections.adjustSynapses(learningSegments, activeInput,
                               permanenceIncrement, -permanenceDecrement)

    # Grow new synapses. Calculate "maxNew", the maximum number of synapses to
    # grow per segment. "maxNew" might be a number or it might be a list of
    # numbers.
    if sampleSize == -1:
      maxNew = len(growthCandidates)
    else:
      maxNew = sampleSize - potentialOverlaps[learningSegments]

    if maxSynapsesPerSegment != -1:
      synapseCounts = connections.mapSegmentsToSynapseCounts(
        learningSegments)
      numSynapsesToReachMax = maxSynapsesPerSegment - synapseCounts
      maxNew = np.where(maxNew <= numSynapsesToReachMax,
                        maxNew, numSynapsesToReachMax)

    connections.growSynapsesToSample(learningSegments, growthCandidates,
                                     maxNew, initialPermanence, rng)


  @staticmethod
  def _learnOnNewSegments(connections, rng, newSegmentCells, growthCandidates,
                          initialPermanence, sampleSize, maxSynapsesPerSegment):
    """
    Create new segments, and grow synapses on them.

    @param connections (SparseMatrixConnections)
    @param rng (Random)
    @param newSegmentCells (numpy array)
    @param growthCandidates (numpy array)
    """

    numNewSynapses = len(growthCandidates)

    if sampleSize != -1:
      numNewSynapses = min(numNewSynapses, sampleSize)

    if maxSynapsesPerSegment != -1:
      numNewSynapses = min(numNewSynapses, maxSynapsesPerSegment)

    newSegments = connections.createSegments(newSegmentCells)
    connections.growSynapsesToSample(newSegments, growthCandidates,
                                     numNewSynapses, initialPermanence,
                                     rng)


  def _chooseBestSegmentPairPerColumn(self,
                                      matchingCellsInBurstingColumns,
                                      matchingBasalSegments,
                                      matchingApicalSegments,
                                      basalPotentialOverlaps,
                                      apicalPotentialOverlaps):
    """
    Choose the best pair of matching segments - one basal and one apical - for
    each column. Pairs are ranked by the sum of their potential overlaps.
    When there's a tie, the first pair wins.

    @param matchingCellsInBurstingColumns (numpy array)
    Cells in bursting columns that have at least one matching basal segment and
    at least one matching apical segment

    @param matchingBasalSegments (numpy array)
    @param matchingApicalSegments (numpy array)
    @param basalPotentialOverlaps (numpy array)
    @param apicalPotentialOverlaps (numpy array)

    @return (tuple)
    - learningBasalSegments (numpy array)
      The selected basal segments

    - learningApicalSegments (numpy array)
      The selected apical segments
    """

    basalCandidateSegments = self.basalConnections.filterSegmentsByCell(
      matchingBasalSegments, matchingCellsInBurstingColumns)
    apicalCandidateSegments = self.apicalConnections.filterSegmentsByCell(
      matchingApicalSegments, matchingCellsInBurstingColumns)

    # Sort everything once rather than inside of each call to argmaxMulti.
    self.basalConnections.sortSegmentsByCell(basalCandidateSegments)
    self.apicalConnections.sortSegmentsByCell(apicalCandidateSegments)

    # Narrow it down to one pair per cell.
    oneBasalPerCellFilter = np2.argmaxMulti(
      basalPotentialOverlaps[basalCandidateSegments],
      self.basalConnections.mapSegmentsToCells(basalCandidateSegments),
      assumeSorted=True)
    basalCandidateSegments = basalCandidateSegments[oneBasalPerCellFilter]
    oneApicalPerCellFilter = np2.argmaxMulti(
      apicalPotentialOverlaps[apicalCandidateSegments],
      self.apicalConnections.mapSegmentsToCells(apicalCandidateSegments),
      assumeSorted=True)
    apicalCandidateSegments = apicalCandidateSegments[oneApicalPerCellFilter]

    # Narrow it down to one pair per column.
    cellScores = (basalPotentialOverlaps[basalCandidateSegments] +
                  apicalPotentialOverlaps[apicalCandidateSegments])
    columnsForCandidates = (
      self.basalConnections.mapSegmentsToCells(basalCandidateSegments) /
      self.cellsPerColumn)
    onePerColumnFilter = np2.argmaxMulti(cellScores, columnsForCandidates,
                                         assumeSorted=True)

    learningBasalSegments = basalCandidateSegments[onePerColumnFilter]
    learningApicalSegments = apicalCandidateSegments[onePerColumnFilter]

    return (learningBasalSegments,
            learningApicalSegments)


  def _getCellsWithFewestSegments(self, columns):
    """
    For each column, get the cell that has the fewest total segments (basal or
    apical). Break ties randomly.

    @param columns (numpy array)
    Columns to check

    @return (numpy array)
    One cell for each of the provided columns
    """
    candidateCells = np2.getAllCellsInColumns(columns, self.cellsPerColumn)

    # Arrange the segment counts into one row per minicolumn.
    segmentCounts = np.reshape(
      self.basalConnections.getSegmentCounts(candidateCells) +
      self.apicalConnections.getSegmentCounts(candidateCells),
      newshape=(len(columns),
                self.cellsPerColumn))

    # Filter to just the cells that are tied for fewest in their minicolumn.
    minSegmentCounts = np.amin(segmentCounts, axis=1, keepdims=True)
    candidateCells = candidateCells[np.flatnonzero(segmentCounts ==
                                                   minSegmentCounts)]

    # Filter to one cell per column, choosing randomly from the minimums.
    # To do the random choice, add a random offset to each index in-place, using
    # casting to floor the result.
    (_,
     onePerColumnFilter,
     numCandidatesInColumns) = np.unique(candidateCells / self.cellsPerColumn,
                                         return_index=True, return_counts=True)

    offsetPercents = np.empty(len(columns), dtype="float32")
    self.rng.initializeReal32Array(offsetPercents)

    np.add(onePerColumnFilter,
           offsetPercents*numCandidatesInColumns,
           out=onePerColumnFilter,
           casting="unsafe")

    return candidateCells[onePerColumnFilter]


  def getActiveCells(self):
    """
    @return (numpy array)
    Active cells
    """
    return self.activeCells


  def getPredictedActiveCells(self):
    """
    @return (numpy array)
    Active cells that were correctly predicted
    """
    return np.intersect1d(self.activeCells, self.predictedCells)


  def getWinnerCells(self):
    """
    @return (numpy array)
    Cells that were selected for learning
    """
    return self.winnerCells


  def getPredictiveCells(self):
    """
    @return (numpy array)
    Cells that were predicted for this timestep
    """
    return self.predictedCells


  def getActiveBasalSegments(self):
    """
    @return (numpy array)
    Active basal segments for this timestep
    """
    return self.activeBasalSegments


  def getActiveApicalSegments(self):
    """
    @return (numpy array)
    Matching basal segments for this timestep
    """
    return self.activeApicalSegments


  def numberOfColumns(self):
    """ Returns the number of columns in this layer.

    @return (int) Number of columns
    """
    return self.columnCount


  def numberOfCells(self):
    """
    Returns the number of cells in this layer.

    @return (int) Number of cells
    """
    return self.numberOfColumns() * self.cellsPerColumn


  def getCellsPerColumn(self):
    """
    Returns the number of cells per column.

    @return (int) The number of cells per column.
    """
    return self.cellsPerColumn


  def getActivationThreshold(self):
    """
    Returns the activation threshold.
    @return (int) The activation threshold.
    """
    return self.activationThreshold


  def setActivationThreshold(self, activationThreshold):
    """
    Sets the activation threshold.
    @param activationThreshold (int) activation threshold.
    """
    self.activationThreshold = activationThreshold

  def getActivationThresholdApical(self):
    """
    Returns the activation threshold.
    @return (int) The activation threshold.
    """
    return self.activationThresholdApical


  def setActivationThresholdApical(self, activationThresholdApical):
    """
    Sets the activation threshold.
    @param activationThresholdApical (int) activation threshold.
    """
    self.activationThresholdApical = activationThresholdApical

  def getInitialPermanence(self):
    """
    Get the initial permanence.
    @return (float) The initial permanence.
    """
    return self.initialPermanence


  def setInitialPermanence(self, initialPermanence):
    """
    Sets the initial permanence.
    @param initialPermanence (float) The initial permanence.
    """
    self.initialPermanence = initialPermanence


  def getMinThresholdBasal(self):
    """
    Returns the min threshold.
    @return (int) The min threshold.
    """
    return self.minThresholdBasal


  def setMinThresholdBasal(self, minThresholdBasal):
    """
    Sets the min threshold.
    @param minThresholdBasal (int) min threshold.
    """
    self.minThresholdBasal = minThresholdBasal

  def getMinThresholdApical(self):
    """
    Returns the min threshold.
    @return (int) The min threshold.
    """
    return self.minThresholdApical


  def setMinThresholdAasal(self, minThresholdApical):
    """
    Sets the min threshold.
    @param minThresholdApical (int) min threshold.
    """
    self.minThresholdApical = minThresholdApical


  def getSampleSize(self):
    """
    Gets the sampleSize.
    @return (int)
    """
    return self.sampleSize


  def setSampleSize(self, sampleSize):
    """
    Sets the sampleSize.
    @param sampleSize (int)
    """
    self.sampleSize = sampleSize


  def getPermanenceIncrement(self):
    """
    Get the permanence increment.
    @return (float) The permanence increment.
    """
    return self.permanenceIncrement


  def setPermanenceIncrement(self, permanenceIncrement):
    """
    Sets the permanence increment.
    @param permanenceIncrement (float) The permanence increment.
    """
    self.permanenceIncrement = permanenceIncrement


  def getPermanenceDecrement(self):
    """
    Get the permanence decrement.
    @return (float) The permanence decrement.
    """
    return self.permanenceDecrement


  def setPermanenceDecrement(self, permanenceDecrement):
    """
    Sets the permanence decrement.
    @param permanenceDecrement (float) The permanence decrement.
    """
    self.permanenceDecrement = permanenceDecrement


  def getBasalPredictedSegmentDecrement(self):
    """
    Get the predicted segment decrement.
    @return (float) The predicted segment decrement.
    """
    return self.basalPredictedSegmentDecrement


  def setBasalPredictedSegmentDecrement(self, predictedSegmentDecrement):
    """
    Sets the predicted segment decrement.
    @param predictedSegmentDecrement (float) The predicted segment decrement.
    """
    self.basalPredictedSegmentDecrement = basalPredictedSegmentDecrement


  def getApicalPredictedSegmentDecrement(self):
    """
    Get the predicted segment decrement.
    @return (float) The predicted segment decrement.
    """
    return self.apicalPredictedSegmentDecrement


  def setApicalPredictedSegmentDecrement(self, predictedSegmentDecrement):
    """
    Sets the predicted segment decrement.
    @param predictedSegmentDecrement (float) The predicted segment decrement.
    """
    self.apicalPredictedSegmentDecrement = apicalPredictedSegmentDecrement


  def getConnectedPermanence(self):
    """
    Get the connected permanence.
    @return (float) The connected permanence.
    """
    return self.connectedPermanence


  def setConnectedPermanence(self, connectedPermanence):
    """
    Sets the connected permanence.
    @param connectedPermanence (float) The connected permanence.
    """
    self.connectedPermanence = connectedPermanence
