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

"""An implementation of MotorTM"""

import numpy as np

from htmresearch.support import numpy_helpers as np2
from nupic.bindings.math import Random, SparseMatrixConnections



class MotorTM(object):
  """
  An alternate approach to apical dendrites. Every cell SDR is specific to both
  the basal the apical input. Prediction requires both basal and apical support.

  For sequence memory, the result is that every sequence happens within a
  "world" which is specified by the apical input. Sequences are not shared
  between worlds.
  """

  def __init__(self,
               columnCount=2048,
               cellsPerColumn=32,
               TDLearningRate=0.5,
               basalInputSize=0,
               apicalInputSize=0,
               activationThreshold=13,
               synPermActiveIncMotor=0.02,
               synPermInactiveDecMotor=0.04,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               punishPredDec=0.004,
               motorCount=32,
               minThreshold=10,
               sampleSize=40,
               winnerSize=4,
               apicalGlobalDecay=0.000001,
               maxSynapsesPerSegment=-1,
               seed=42):
    """

    @param TDLearningRate (float)
    The learning rate for temporal difference learning.
    Used for calculation of the increment/decrement of apical synapses.

    @param motorCount (int)
    The number of motor neurons to represent the action space.

    @param columnCount (int)
    The number of minicolumns. (same as L5 as we overtake distal and active columns)

    @param cellsPerColumn (int)
    The number of cells per mini-column. Used to determine column structure
    for input of depolarizedBasalCells.

    @param basalInputSize (int)
    The number of bits in the basal input. (Depolarized cells from L5)

    @param apicalInputSize (int)
    The number of bits in the apical input from D1 or D2.

    @param activationThreshold (int)
    If the number of active connected synapses on a segment is at least this
    threshold, the segment is said to be active.

    @param synPermActiveIncMotor (float)
    Amount by which permanences of L5-Motor synapses are incremented during learning.

    @param synPermInactiveDecMotor (float)
    Amount by which permanences of L5-Motor synapses are incremented during learning.

    @param initialPermanence (float)
    Initial permanence of a new synapse

    @param connectedPermanence (float)
    If the permanence value for a synapse is greater than this value, it is said
    to be connected.

    @param punishPredDec (float)
    Not correctly predicted cells will be decremented by this amount.

    @param minThreshold (int)
    If the number of potential synapses active on a segment is at least this
    threshold, it is said to be "matching" and is eligible for learning.

    @param sampleSize (int)
    How much of the active SDR to sample with synapses.

    @param winnerSize (int)
    Determines how many motor cells ultimatively become active.
    The k-cells with highest excitation are the winner cells.

    @param apicalGlobalDecay (float)
    Global decay applied to apical synapses at each iteration.
    Needed to remove unused segments and have better memory performance.

    @param maxSynapsesPerSegment
    The maximum number of synapses per segment.

    @param seed (int)
    Seed for the random number generator.
    """

    # Cell structure and learning params
    self.columnCount = columnCount
    self.cellsPerColumn = cellsPerColumn
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.minThreshold = minThreshold
    self.sampleSize = sampleSize
    self.activationThreshold = activationThreshold
    self.maxSynapsesPerSegment = maxSynapsesPerSegment
    self.rng = Random(seed)

    self.apicalConnectionsD1 = SparseMatrixConnections(basalInputSize,
                                                       apicalInputSize)
    self.apicalConnectionsD2 = SparseMatrixConnections(basalInputSize,
                                                       apicalInputSize)

    # TD Learning params
    self.apicalGlobalDecay = apicalGlobalDecay
    self.TDLearningRate = TDLearningRate
    self.winnerSize = winnerSize
    self.punishPredDec = punishPredDec
    self.motorCount = motorCount

    # OUTPUT
    self.voluntaryActiveCells = ()
    self.winnerCells = ()
    self.activeMotorCells = ()
    self.depolarizedApicalCells = ()
    self.depolarizedBasalCells = ()
    self.motorCells = ()

    #  save prev. input for learning - L5(t) & D1/D2(t-1)
    self.prevApicalInputD1 = ()
    self.prevApicalInputD2 = ()

    # used to learn connections to L5
    # -> State associates itself with action that produced it
    self.motorMapping = SparseMatrixConnections(motorCount,
                                                basalInputSize)
    self.synPermActiveIncMotor = synPermActiveIncMotor
    self.synPermInactiveDecMotor = synPermInactiveDecMotor


  def reset(self):
    """
    Clear all cell and segment activity. This has no effect on the subsequent
    predictions or activity.
    """
    #  INPUT RESET
    self.prevApicalInputD1 = ()
    self.prevApicalInputD2 = ()
    # OUTPUT Reset
    # No reset of active cells, as they are used for input in L4
    # self.depolarizedApicalCells = ()
    # self.depolarizedBasalCells = ()
    # self.voluntaryActiveCells = ()
    # self.winnerCells = ()
    # self.activeMotorCells = ()

  def compute(self,
              activeColumns,
              depolarizedBasalCells,
              activeCells,
              apicalInputD1,
              apicalInputD2,
              TDError,
              learn=True):
    """
    Perform one timestep. Use the basal and apical input to form a set of
    cells to excite/inhibit.

    @param TDError (float)
    Temporal Difference error calculated in D1/D2 from the state values
    at time t and t+1.

    @param activeColumns (numpy array)
    List of active column indicies needed for learning on apical dendrite segments on layer5-D1/D2.

    @param depolarizedBasalCells (numpy array)
    List of cell indices depolarized by basal input in layer 5.

    @param activeCells (numpy array)
    List of active cells from layer 5 needed to learn apical connections from Motor-Layer5.

    @param apicalInputD1 (numpy array)
    List of active input bits (indicies) for the apical dendrite segments from D1

    @param apicalInputD2 (numpy array)
    List of active input bits (indicies) for the apical dendrite segments from D2

    @param learn (bool)
    Whether to grow / reinforce / punish synapses
    """

    # SECTION 1: Calculate voluntary activation (ALL L5 layer input D1)
    # Use L5(t), TD-Error(t) and D1/D2(t-1) input to calculate depolarized cells and learn connections
    # Calculate apical predictions for this timestep

    (depolarizedApicalCellsD1,
     depolarizedApicalCellsD2,
     voluntaryActiveCellsD1,
     voluntaryActiveCellsD2) = self._apicalDepolarization(learn,
                                                          self.prevApicalInputD1,
                                                          self.prevApicalInputD2,
                                                          activeColumns,
                                                          depolarizedBasalCells,
                                                          TDError)

    # SECTION 2: Calculate motor activity (MOTOR LAYER)
    # calculate the winner cells (Motor commands)
    # -> base level excitation (causing random behavior) + excitation/inhibition

    (depolarizedMotorCellsD1,
     depolarizedMotorCellsD2) = self._calculateMotorExcitation(voluntaryActiveCellsD1,
                                                               voluntaryActiveCellsD2)

    # SECTION 3: Learn mapping connections
    #            Action(t-1) -> lead to L5(t) state/activeCells
    #            This are connections to the L5 active state produced from the action
    #            but they are utilized by section 2 from the voluntary active neurons

    if learn:
        self._learnMotorApicalConnections(activeCells)

    # SECTION 4: Save the results
    # save current apical input for next timestep
    self.prevApicalInputD1 = apicalInputD1
    self.prevApicalInputD2 = apicalInputD2

    # Output: get from MotorRegion
    self.depolarizedApicalCells = np.union1d(depolarizedApicalCellsD1, depolarizedApicalCellsD2)
    self.depolarizedBasalCells = depolarizedBasalCells
        # Active through depolarization from apical *and* distal connections
    self.voluntaryActiveCells = np.union1d(voluntaryActiveCellsD1, voluntaryActiveCellsD2)
        # Motor cell indices that got excited/inhibited (mapped from voluntaryActiveCells)
    self.activeMotorCells = np.union1d(depolarizedMotorCellsD1, depolarizedMotorCellsD2)
        # Winner cells: highest excitation -> determines motor command (action)
    self.winnerCells = np.argpartition(self.motorCells, -self.winnerSize)[-self.winnerSize:]

    print '\033[93m'
    print '[Motor Neurons] ~~~~~~~ Excite/Inhibit ~~~~~~~'
    print '[Motor Neurons] Voluntary active cells (input apical activation)', self.voluntaryActiveCells
    print '[Motor Neurons] Depolarized motor cells D1 (from voluntary active)', depolarizedMotorCellsD1
    print '[Motor Neurons] Depolarized motor cells D2 (from voluntary active)', depolarizedMotorCellsD2
    print '[Motor Neurons] Motor Values', self.motorCells
    print '[Motor Neurons] Winner cells', self.winnerCells
    print '\033[0m'

  def _calculateMotorExcitation(self,
                                voluntaryActiveCellsD1,
                                voluntaryActiveCellsD2):
      """ Calculate the excitation level of motor neurons
          excited/inhibited by voluntary active L5 neurons.

          @param voluntaryActiveCellsD1 (numpy array)
          Indices of voluntary activation (through D1 origin) from L5 neurons calculated in SECTION 2.

          @param voluntaryActiveCellsD2 (numpy array)
          Indices of voluntary activation (through D2 origin) from L5 neurons calculated in SECTION 2.

          @return (tuple)
            - depolarizedMotorCellsD1
            - depolarizedMotorCellsD2
      """

      # params
      mean = 1
      variance = 0.25
      inhibitionLevel = 1
      excitementLevel = 1
      motorExciteThreshold = 1

      # Initialize with random excitation
      self.motorCells = np.random.normal(mean, variance, self.motorCount)

      # NOTE: Map motor neurons CONSTANT (-1 because indices)
      # mappedActiveCellsD1 = np.interp(activeCellsD1, [0, self.columnCount*self.cellsPerColumn-1], [0, self.motorCount-1]).astype(int)
      # mappedActiveCellsD2 = np.interp(activeCellsD2, [0, self.columnCount*self.cellsPerColumn-1], [0, self.motorCount-1]).astype(int)

      # Map motor cells: Calculate which neurons to excite/inhibit
      # from voluntary activation through existing connections
      (activeApicalSegmentsMotorD1,
       matchingApicalSegmentsMotorD1,
       apicalPotentialOverlapsMotorD1) = self._calculateSegmentActivity(
         self.motorMapping, voluntaryActiveCellsD1, self.connectedPermanence,
         motorExciteThreshold, motorExciteThreshold)

      (activeApicalSegmentsMotorD2,
       matchingApicalSegmentsMotorD2,
       apicalPotentialOverlapsMotorD2) = self._calculateSegmentActivity(
         self.motorMapping, voluntaryActiveCellsD2, self.connectedPermanence,
         motorExciteThreshold, motorExciteThreshold)

      print 'activeApicalSegments D1', activeApicalSegmentsMotorD1
      print 'Overlaps D1', apicalPotentialOverlapsMotorD1
      print 'activeApicalSegments D2', activeApicalSegmentsMotorD2
      print 'Overlaps D2', apicalPotentialOverlapsMotorD2


      # Map segments to depolarized cells
      depolarizedMotorCellsD1 = self.motorMapping.mapSegmentsToCells(activeApicalSegmentsMotorD1)
      depolarizedMotorCellsD2 = self.motorMapping.mapSegmentsToCells(activeApicalSegmentsMotorD2)

    # # excite/inhibit

      # - NOTE: Should multiple voluntary neurons excite a single motor neuron multiple or unique?
      # Get the motor cells that had active segments (for excite/inhibit) & sort (possibly excited by multiple segments)
    #   Unique option
        #   depolarizedMotorCellsD1 = np.sort(np.unique(depolarizedMotorCellsD1))
        #   depolarizedMotorCellsD2 = np.sort(np.unique(depolarizedMotorCellsD2))
        #   self.motorCells[depolarizedMotorCellsD1] += excitementLevel
        #   self.motorCells[depolarizedMotorCellsD2] -= inhibitionLevel

    #  Multiple option
      for i in depolarizedMotorCellsD1:
          self.motorCells[i] += excitementLevel
      for i in depolarizedMotorCellsD2:
          self.motorCells[i] -= inhibitionLevel

      return (depolarizedMotorCellsD1,
              depolarizedMotorCellsD2)

  def _apicalDepolarization(self,
                            learn,
                            prevApicalInputD1,
                            prevApicalInputD2,
                            activeColumns,
                            depolarizedBasalCells,
                            TDError):
      """ Calculate the apical depolarization of layer 5 neurons

      @param learn (bool)
      Whether to grow / reinforce / punish synapses

      @param prevApicalInputD1 (numpy array)
      Indices of D1 activation at time t-1.

      @param prevApicalInputD2 (numpy array)
      Indices of D2 activation at time t-1.

      @param activeColumns (numpy array)
      Indices of active column activation from layer 5 at time t.

      @param depolarizedBasalCells (numpy array)
      List of cell indices depolarized by basal input in layer 5.

      @param TDError (float)
      Temporal Difference error calculated in D1/D2 from the state values
      at time t-1 and t. It determines how the apical connections to t-1 are formed.

      If positive: prevState=Action resulted in unexpected positive change
                    -> Strengthen D1 connections to excite activation
                    -> Weaken D2 connections to not inhibit
      If negative: prevState=Action resulted in unexpected negative change
                    -> Strengthen D2 connections to inhibit acivation
                    -> Weaken D1 connections to not excite

      Formula: TDError = reward(t) + self.TDDiscount * (stateValue(t) - prevStateValue(t-1))

      @return (tuple)
        - depolarizedApicalCellsD1 (Motor cells)
        - depolarizedApicalCellsD2 (Motor cells)
        - voluntaryActiveCellsD1   (Layer 5 cells)
        - voluntaryActiveCellsD2   (Layer 5 cells)
      """

      # Calculate layer 5 neurons apical segment activity
      (activeApicalSegmentsD1,
       matchingApicalSegmentsD1,
       apicalPotentialOverlapsD1) = self._calculateSegmentActivity(
         self.apicalConnectionsD1, prevApicalInputD1, self.connectedPermanence,
         self.activationThreshold, self.minThreshold)

      (activeApicalSegmentsD2,
       matchingApicalSegmentsD2,
       apicalPotentialOverlapsD2) = self._calculateSegmentActivity(
         self.apicalConnectionsD2, prevApicalInputD2, self.connectedPermanence,
         self.activationThreshold, self.minThreshold)

      # Calculate apically depolarized cells
      depolarizedApicalCellsD1 = self.apicalConnectionsD1.mapSegmentsToCells(activeApicalSegmentsD1)
      depolarizedApicalCellsD2 = self.apicalConnectionsD2.mapSegmentsToCells(activeApicalSegmentsD2)

      # Calculate voluntary active cells through apical *and* distal depolarization
      voluntaryActiveCellsD1 = np.intersect1d(
        depolarizedBasalCells,
        depolarizedApicalCellsD1)
      voluntaryActiveCellsD2 = np.intersect1d(
        depolarizedBasalCells,
        depolarizedApicalCellsD2)

      # Calculate correctly predicted Cells using layer 5 columnar activation
      depolarizedApicalCells = np.union1d(depolarizedApicalCellsD1, depolarizedApicalCellsD2)
      (correctPredictedCells,
       burstingColumns) = np2.setCompare(depolarizedApicalCells, activeColumns,
                                         depolarizedApicalCells / self.cellsPerColumn,
                                         rightMinusLeft=True)

      print '[Motor TM] D1 Apical Input(t-1)', prevApicalInputD1
      print '[Motor TM] D1 Active apical segments', activeApicalSegmentsD1
      print '[Motor TM] D1 Matching apical segments', matchingApicalSegmentsD1
      print '[Motor TM] D1 Potential apical overlap', apicalPotentialOverlapsD1
      print '[Motor TM] Bursting columns (will grow new segments)', burstingColumns

      (learningActiveApicalSegmentsD1,
       learningActiveApicalSegmentsD2,
       learningMatchingApicalSegmentsD1,
       learningMatchingApicalSegmentsD2,
       apicalSegmentsToPunishD1,
       apicalSegmentsToPunishD2,
       newSegmentCells,
       learningCells) = self._calculateLearning(activeColumns,
                                                burstingColumns,
                                                correctPredictedCells,
                                                activeApicalSegmentsD1,
                                                activeApicalSegmentsD2,
                                                matchingApicalSegmentsD1,
                                                matchingApicalSegmentsD2,
                                                apicalPotentialOverlapsD1,
                                                apicalPotentialOverlapsD2)

      # Calculate permanence dec/inc from TD error
      permanenceDec = np.abs(TDError) * self.TDLearningRate;
      permanenceIncD1 = TDError * self.TDLearningRate;
      permanenceIncD2 = - TDError * self.TDLearningRate;
      permanencePredictedDec = self.punishPredDec;

      if learn:
        # Gloabal Decay
        if self.apicalGlobalDecay != 0.0:
            # Get ALL segments (TODO: should be a better way) -> call without permanence threshold
            allSegmentsD1 = np.arange(len(self.apicalConnectionsD1.computeActivity(prevApicalInputD1)))
            allSegmentsD2 = np.arange(len(self.apicalConnectionsD2.computeActivity(prevApicalInputD2)))

            # Decay all decremented
            self.apicalConnectionsD1.adjustSynapses(allSegmentsD1,
                      [],0.0, -self.apicalGlobalDecay)
            self.apicalConnectionsD2.adjustSynapses(allSegmentsD2,
                      [],0.0, -self.apicalGlobalDecay)
            # TODO:Destroy segments with no synapses
            #self.apicalConnectionsD1.destroySegments(allSegmentsD1 == 0)
            #self.apicalConnectionsD2.destroySegments(allSegmentsD2 == 0)

        # Declared for clearer code
        apicalGrowthCandidatesD1 = prevApicalInputD1
        apicalGrowthCandidatesD2 = prevApicalInputD2

        print '[Motor TM] Learning active segments D1', learningActiveApicalSegmentsD1
        print '[Motor TM] Learning matching segments D1', learningMatchingApicalSegmentsD1
        print '[Motor TM] Permanence Increment D1 (TD-Error)', permanenceIncD1
        print '[Motor TM] Permanence Increment D2 (TD-Error)', permanenceIncD2
        print '[Motor TM] Permanence Decrement (TD-Error)', permanenceDec
        print '[Motor TM] Learning cells', learningCells

        # Learn on existing segments
        for learningSegments in (learningActiveApicalSegmentsD1,
                                 learningMatchingApicalSegmentsD1):
          self._learn(self.apicalConnectionsD1, self.rng, learningSegments,
                      prevApicalInputD1, apicalGrowthCandidatesD1, apicalPotentialOverlapsD1,
                      self.initialPermanence, self.sampleSize,
                      permanenceIncD1, permanenceDec,
                      self.maxSynapsesPerSegment)

        for learningSegments in (learningActiveApicalSegmentsD2,
                                 learningMatchingApicalSegmentsD2):
          self._learn(self.apicalConnectionsD2, self.rng, learningSegments,
                      prevApicalInputD2, apicalGrowthCandidatesD2,
                      apicalPotentialOverlapsD2, self.initialPermanence,
                      self.sampleSize, permanenceIncD2,
                      permanenceDec, self.maxSynapsesPerSegment)

        # Punish incorrect predictions
        if permanencePredictedDec != 0.0:
          self.apicalConnectionsD1.adjustActiveSynapses(
            apicalSegmentsToPunishD1, prevApicalInputD1,
            -permanencePredictedDec)

        if permanencePredictedDec != 0.0:
          self.apicalConnectionsD2.adjustActiveSynapses(
            apicalSegmentsToPunishD2, prevApicalInputD2,
            -permanencePredictedDec)

        # If there is apical input build new Segments to learn
        if len(prevApicalInputD1) > 0 and len(prevApicalInputD2) > 0:
          self._learnOnNewSegments(self.apicalConnectionsD1, self.rng,
                                   newSegmentCells, apicalGrowthCandidatesD1,
                                   self.initialPermanence, self.sampleSize,
                                   self.maxSynapsesPerSegment)
          self._learnOnNewSegments(self.apicalConnectionsD2, self.rng,
                                   newSegmentCells, apicalGrowthCandidatesD2,
                                   self.initialPermanence, self.sampleSize,
                                   self.maxSynapsesPerSegment)

      return (depolarizedApicalCellsD1,
              depolarizedApicalCellsD2,
              voluntaryActiveCellsD1,
              voluntaryActiveCellsD2)


  def _learnMotorApicalConnections(self,
                                   activeCells):
      """ Learn apical connections from the previous motor activation (t-1)
          to layer 5(t) activations.
          Is associates the action with the state it produced.

          @param activeCells (numpy array)
          Indices of layer 5(t) neural activation (Agent State).
      """

      # Calculate motor segment activity for layer 5 neural activation.
      (activeApicalSegmentsMotor,
       matchingApicalSegmentsMotor,
       apicalPotentialOverlapsMotor) = self._calculateSegmentActivity(
         self.motorMapping, activeCells, self.connectedPermanence,
         self.activationThreshold, self.minThreshold)

      # calculate learning segments
      (learningActiveMotorSegments,
       learningMatchingMotorSegments,
       motorSegmentsToPunish,
       newSegmentMotorCells) = self._calculateMotorLearning(activeApicalSegmentsMotor,
                                                matchingApicalSegmentsMotor,
                                                apicalPotentialOverlapsMotor)

      activeMotorCellsDEBUG = self.motorMapping.mapSegmentsToCells(learningActiveMotorSegments)
      matchingMotorCellsDEBUG = self.motorMapping.mapSegmentsToCells(learningMatchingMotorSegments)
      punishMotorCellsDEBUG = self.motorMapping.mapSegmentsToCells(motorSegmentsToPunish)

      print '\033[91m'
      print 'active motorCells which produced state', activeMotorCellsDEBUG
      print 'matching motorCells which produced state', matchingMotorCellsDEBUG
      print 'punish wrong mapping which didnt produced state', punishMotorCellsDEBUG
      print '\033[0m'

      # Learn on existing segments (activeCells = motorInput and motorGrowthCandidates)
      for learningSegments in (learningActiveMotorSegments,
                               learningMatchingMotorSegments):
          self._learn(self.motorMapping, self.rng, learningSegments,
                      activeCells, activeCells, apicalPotentialOverlapsMotor,
                      self.initialPermanence, self.sampleSize,
                      self.synPermActiveIncMotor, self.synPermInactiveDecMotor,
                      self.maxSynapsesPerSegment)

      # Punish incorrect predictions
      if self.punishPredDec != 0.0:
          self.motorMapping.adjustActiveSynapses(
            motorSegmentsToPunish, activeCells,
            -self.punishPredDec)

      # If there is apical input build new Segments to learn
      if len(activeCells) > 0:
          self._learnOnNewSegments(self.motorMapping, self.rng,
                                   newSegmentMotorCells, activeCells,
                                   self.initialPermanence, self.sampleSize,
                                   self.maxSynapsesPerSegment)


  def _calculateMotorLearning(self,
                           activeApicalSegmentsMotor,
                           matchingApicalSegmentsMotor,
                           apicalPotentialOverlapsMotor):
    """ Calculate the motor cells that learn dependent
        on segment activity caused by layer 5 activation.

        @param activeApicalSegmentsMotor (numpy array)
        Active apical segments through layer 5 neural activation.

        @param matchingApicalSegmentsMotor (numpy array)
        Matching apical segments through layer 5 neural activation.

        @param apicalPotentialOverlapsMotor (numpy array)
        Overlap scores of segments with layer 5 neural activation.

        @return (tuple)
            - learningActiveMotorSegments
            - learningMatchingMotorSegments
            - motorSegmentsToPunish
            - newSegmentMotorCells

    """

    # calculate correct predictions => connections that associated action-leadsTo-state correctly
        # winnerCells represent the previous motorActivation(t-1)
    depolarizedMotorCells = self.motorMapping.mapSegmentsToCells(activeApicalSegmentsMotor)
        # produced L5 state and predicted
    correctPredictedCells = np.intersect1d(depolarizedMotorCells, self.winnerCells)
        # produced L5 state not predicted
    nonPredictedCells = np.setdiff1d(self.winnerCells, depolarizedMotorCells)

    print '\033[91m'
    print "Prev. winner cells", self.winnerCells
    print
    print 'Depolarized cells (most overlapping segments to L5 State)', depolarizedMotorCells
    print 'The ones of them which were correctly predicted (match prev. motor cells)', correctPredictedCells
    print 'The ones that were not correctly predicted (did not match prevWinnerCells)', nonPredictedCells
    print '\033[0m'

    # Correctly predicted cells
    learningActiveMotorSegments = self.motorMapping.filterSegmentsByCell(
            activeApicalSegmentsMotor, correctPredictedCells)

    # Matching cells from matching segments (segments with enough activity
    # to learn, but did not become active)
    allMatchingCells = self.motorMapping.mapSegmentsToCells(matchingApicalSegmentsMotor)

    # Cells that matched with state activation and cells without any match
    matchingCells = np.intersect1d(allMatchingCells, nonPredictedCells)
    cellsWithNoMatch = np.setdiff1d(nonPredictedCells, allMatchingCells)

    # new segment cells for cells without any match
    # (cellsWithFewestSegments- with one cell per column)
    newSegmentMotorCells = cellsWithNoMatch

    # Matching cells filter one segment per cell
    motorCandidateSegments = self.motorMapping.filterSegmentsByCell(
       matchingApicalSegmentsMotor, matchingCells)
    # sort for argmaxMulti
    self.motorMapping.sortSegmentsByCell(motorCandidateSegments)

    # Narrow it down to one segment per cell. (Use Segment with max. activity)
    oneMotorPerCellFilter = np2.argmaxMulti(
      apicalPotentialOverlapsMotor[motorCandidateSegments],
      self.motorMapping.mapSegmentsToCells(motorCandidateSegments),
      assumeSorted=True)
    learningMatchingMotorSegments = motorCandidateSegments[oneMotorPerCellFilter]

    # Calculate punish segments
    correctMatchingMask = np.in1d(allMatchingCells, self.winnerCells)
    matchingSegmentsToPunish = matchingApicalSegmentsMotor[~correctMatchingMask]


    correctActiveMask = np.in1d(depolarizedMotorCells, self.winnerCells)
    activeSegmentsToPunish = activeApicalSegmentsMotor[~correctActiveMask]

    print 'Punish matching', matchingSegmentsToPunish
    print 'Punish active', activeSegmentsToPunish

    motorSegmentsToPunish = np.union1d(matchingSegmentsToPunish,activeSegmentsToPunish)

    return (learningActiveMotorSegments,
            learningMatchingMotorSegments,
            motorSegmentsToPunish,
            newSegmentMotorCells)



  def _calculateLearning(self,
                         activeColumns,
                         burstingColumns,
                         correctPredictedCells,
                         activeApicalSegmentsD1,
                         activeApicalSegmentsD2,
                         matchingApicalSegmentsD1,
                         matchingApicalSegmentsD2,
                         apicalPotentialOverlapsD1,
                         apicalPotentialOverlapsD2):
    """
    Learning occurs on pairs of segments. Correctly predicted cells always have
    active basal and apical segments, and we learn on these segments. In
    bursting columns, we either learn on an existing segment pair, or we grow a
    new pair of segments.

    @param activeColumns (numpy array)
    @param burstingColumns (numpy array)
    @param correctPredictedCells (numpy array)
    @param activeApicalSegmentsD1 (numpy array)
    @param activeApicalSegmentsD2 (numpy array)
    @param matchingApicalSegmentsD1 (numpy array)
    @param matchingApicalSegmentsD2 (numpy array)
    @param apicalPotentialOverlapsD1 (numpy array)
    @param apicalPotentialOverlapsD2 (numpy array)

    @return (tuple)
    - learningActiveApicalSegmentsD1 (numpy array)
      Active apical segments on correct predicted cells for D1

    - learningActiveApicalSegmentsD2 (numpy array)
      Active apical segments on correct predicted cells for D2

    - learningMatchingApicalSegmentsD1 (numpy array)
      Matching apical segments selected for learning in bursting columns

    - learningMatchingApicalSegmentsD2 (numpy array)
      Matching apical segments selected for learning in bursting columns

    - apicalSegmentsToPunishD1 (numpy array)
      Apical segments that should be punished for predicting an inactive column

    - apicalSegmentsToPunishD2 (numpy array)
      Apical segments that should be punished for predicting an inactive column

    - newSegmentCells (numpy array)
      Cells in bursting columns that were selected to grow new segments

    - learningCells (numpy array)
      Every cell that has a learning segment or was selected to grow a segment
    """

    # Correctly predicted columns
    learningActiveApicalSegmentsD1 = self.apicalConnectionsD1.filterSegmentsByCell(
      activeApicalSegmentsD1, correctPredictedCells)
    learningActiveApicalSegmentsD2 = self.apicalConnectionsD2.filterSegmentsByCell(
      activeApicalSegmentsD2, correctPredictedCells)

    # Matching cells from matching segments (segments with enough activity
    # to learn, but did not become active)
    cellsForMatchingApicalD1 = self.apicalConnectionsD1.mapSegmentsToCells(
      matchingApicalSegmentsD1)
    cellsForMatchingApicalD2 = self.apicalConnectionsD2.mapSegmentsToCells(
      matchingApicalSegmentsD2)
    matchingCells = np.intersect1d(
      cellsForMatchingApicalD1, cellsForMatchingApicalD2)

    # Matching Cells in bursting columns and bursting columns without match
    (matchingCellsInBurstingColumns,
     burstingColumnsWithNoMatch) = np2.setCompare(
       matchingCells, burstingColumns, matchingCells / self.cellsPerColumn,
       rightMinusLeft=True)

     # Bursting columns with no matching or predicted cells -> grow new segment cells
    (learningMatchingApicalSegmentsD1,
     learningMatchingApicalSegmentsD2) = self._chooseBestSegmentPairPerColumn(
       matchingCellsInBurstingColumns, matchingApicalSegmentsD1,
       matchingApicalSegmentsD2, apicalPotentialOverlapsD1, apicalPotentialOverlapsD2)
    newSegmentCells = self._getCellsWithFewestSegments(
      burstingColumnsWithNoMatch)

    # Incorrectly predicted columns (always punish) TODO: Make nicer (outer function calculates depolarizedApicalCellsD1 already)
    correctMatchingApicalMaskD1 = np.in1d(
        cellsForMatchingApicalD1 / self.cellsPerColumn, activeColumns)
    matchingSegmentsToPunishD1 = matchingApicalSegmentsD1[~correctMatchingApicalMaskD1]

    depolarizedApicalCellsD1 = self.apicalConnectionsD1.mapSegmentsToCells(activeApicalSegmentsD1)
    correctActiveApicalMaskD1 = np.in1d(
        depolarizedApicalCellsD1 / self.cellsPerColumn, activeColumns)
    activeSegmentsToPunishD1 = activeApicalSegmentsD1[~correctActiveApicalMaskD1]

    correctMatchingApicalMaskD2 = np.in1d(
        cellsForMatchingApicalD2 / self.cellsPerColumn, activeColumns)
    matchingSegmentsToPunishD2 = matchingApicalSegmentsD2[~correctMatchingApicalMaskD2]

    depolarizedApicalCellsD2 = self.apicalConnectionsD2.mapSegmentsToCells(activeApicalSegmentsD2)
    correctActiveApicalMaskD2 = np.in1d(
        depolarizedApicalCellsD2 / self.cellsPerColumn, activeColumns)
    activeSegmentsToPunishD2 = activeApicalSegmentsD2[~correctActiveApicalMaskD2]

    apicalSegmentsToPunishD1 = np.union1d(matchingSegmentsToPunishD1, activeSegmentsToPunishD1)
    apicalSegmentsToPunishD2 = np.union1d(matchingSegmentsToPunishD2, activeSegmentsToPunishD2)

    # Make a list of every cell that is learning
    learningCells =  np.concatenate(
      (correctPredictedCells,
       np.union1d(self.apicalConnectionsD1.mapSegmentsToCells(learningMatchingApicalSegmentsD1),
                  self.apicalConnectionsD2.mapSegmentsToCells(learningMatchingApicalSegmentsD2)),
       newSegmentCells))

    return (learningActiveApicalSegmentsD1,
            learningActiveApicalSegmentsD2,
            learningMatchingApicalSegmentsD1,
            learningMatchingApicalSegmentsD2,
            apicalSegmentsToPunishD1,
            apicalSegmentsToPunishD2,
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

    # TODO:DEBUG -> it takes the maxNew from the 3rd if statement,
    # maybe it fills up since old synapses do not get removed
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

    basalCandidateSegments = self.apicalConnectionsD1.filterSegmentsByCell(
      matchingBasalSegments, matchingCellsInBurstingColumns)
    apicalCandidateSegments = self.apicalConnectionsD2.filterSegmentsByCell(
      matchingApicalSegments, matchingCellsInBurstingColumns)

    # Sort everything once rather than inside of each call to argmaxMulti.
    self.apicalConnectionsD1.sortSegmentsByCell(basalCandidateSegments)
    self.apicalConnectionsD2.sortSegmentsByCell(apicalCandidateSegments)

    # Narrow it down to one pair per cell.
    oneBasalPerCellFilter = np2.argmaxMulti(
      basalPotentialOverlaps[basalCandidateSegments],
      self.apicalConnectionsD1.mapSegmentsToCells(basalCandidateSegments),
      assumeSorted=True)
    basalCandidateSegments = basalCandidateSegments[oneBasalPerCellFilter]
    oneApicalPerCellFilter = np2.argmaxMulti(
      apicalPotentialOverlaps[apicalCandidateSegments],
      self.apicalConnectionsD2.mapSegmentsToCells(apicalCandidateSegments),
      assumeSorted=True)
    apicalCandidateSegments = apicalCandidateSegments[oneApicalPerCellFilter]

    # Narrow it down to one pair per column.
    cellScores = (basalPotentialOverlaps[basalCandidateSegments] +
                  apicalPotentialOverlaps[apicalCandidateSegments])
    columnsForCandidates = (
      self.apicalConnectionsD1.mapSegmentsToCells(basalCandidateSegments) /
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
      self.apicalConnectionsD1.getSegmentCounts(candidateCells) +
      self.apicalConnectionsD2.getSegmentCounts(candidateCells),
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


  def getVoluntaryActiveCells(self):
    """
    @return (numpy array)
    Active through depolarization from apical *and* distal connections
    """
    return self.voluntaryActiveCells

  def getMotorCells(self):
    """
    @return (numpy array)
    Motor cells (randomly generated values influenced by excitement/inhibition)
    """
    return self.motorCells

  def getActiveMotorCells(self):
    """
    @return (numpy array)
    Motor cell indices that got excited/inhibited (mapped from voluntaryActiveCells)
    """
    return self.activeMotorCells

  def getWinnerCells(self):
    """
    @return (numpy array)
    Motor cells that won the competition for k-winners through highest excitation values
    """
    return self.winnerCells


  def getDepolarizedApicalCells(self):
    """
    @return (numpy array)
    Cells that were depolarized through apical connections for this timestep
    """
    return self.depolarizedApicalCells

  def getDepolarizedBasalCells(self):
    """
    @return (numpy array)
    Cells that were depolarized through basal connections for this timestep
    """
    return self.depolarizedBasalCells


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


  def getMinThreshold(self):
    """
    Returns the min threshold.
    @return (int) The min threshold.
    """
    return self.minThreshold


  def setMinThreshold(self, minThreshold):
    """
    Sets the min threshold.
    @param minThreshold (int) min threshold.
    """
    self.minThreshold = minThreshold


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

  def write(self, proto):
    """Populate serialization proto instance.

    :param proto: (MotorTMProto) the proto instance to populate
    """
    # Cell structure and learning params
    proto.columnCount = int(self.columnCount)
    proto.cellsPerColumn = int(self.cellsPerColumn)
    proto.activationThreshold = int(self.activationThreshold)
    proto.initialPermanence = float(self.initialPermanence)
    proto.connectedPermanence = float(self.connectedPermanence)
    proto.minThreshold = int(self.minThreshold)
    proto.sampleSize = int(self.sampleSize)
    proto.maxSynapsesPerSegment = int(self.maxSynapsesPerSegment)

    self.rng.write(proto.random)

    voluntaryActiveCellsProto = proto.init("voluntaryActiveCells", len(self.voluntaryActiveCells))
    for i, cell in enumerate(self.voluntaryActiveCells):
        voluntaryActiveCellsProto[i] = int(cell)

    winnerCellsProto = proto.init("winnerCells", len(self.winnerCells))
    for i, cell in enumerate(self.winnerCells):
        winnerCellsProto[i] = int(cell)

    # TD Learning
    proto.learningRate = float(self.TDLearningRate)
    proto.apicalGlobalDecay = float(self.apicalGlobalDecay)
    proto.winnerSize = int(self.winnerSize)
    proto.punishPredDec = float(self.punishPredDec)
    proto.motorCount = int(self.motorCount)

    motorCellsProto = proto.init("motorCells", len(self.motorCells))
    for i, cell in enumerate(self.motorCells):
        motorCellsProto[i] = int(cell)

    activeMotorCellsProto = proto.init("activeMotorCells", len(self.activeMotorCells))
    for i, cell in enumerate(self.activeMotorCells):
        activeMotorCellsProto[i] = int(cell)

    depolarizedApicalCellsProto = proto.init("depolarizedApicalCells", len(self.depolarizedApicalCells))
    for i, cell in enumerate(self.depolarizedApicalCells):
        depolarizedApicalCellsProto[i] = int(cell)

    depolarizedBasalCellsProto = proto.init("depolarizedBasalCells", len(self.depolarizedBasalCells))
    for i, cell in enumerate(self.depolarizedBasalCells):
        depolarizedBasalCellsProto[i] = int(cell)

    #  save prev. input for learning - L5(t) & D1/D2(t-1)
    prevApicalInputD1Proto = proto.init("prevApicalInputD1", len(self.prevApicalInputD1))
    for i, cell in enumerate(self.prevApicalInputD1):
        prevApicalInputD1Proto[i] = int(cell)

    prevApicalInputD2Proto = proto.init("prevApicalInputD2", len(self.prevApicalInputD2))
    for i, cell in enumerate(self.prevApicalInputD2):
        prevApicalInputD2Proto[i] = int(cell)

    # self.apicalConnectionsD1.write(proto.apicalConnectionsD1)
    # self.apicalConnectionsD2.write(proto.apicalConnectionsD2)


  @classmethod
  def read(cls, proto):
    """Deserialize from proto instance.

    :param proto: (MotorTMProto) the proto instance to read from
    """
    instance = object.__new__(cls)

    # Cell structure and learning params
    instance.columnCount = proto.columnCount
    instance.cellsPerColumn = proto.cellsPerColumn
    instance.activationThreshold = proto.activationThreshold
    instance.initialPermanence = proto.initialPermanence
    instance.connectedPermanence = proto.connectedPermanence
    instance.minThreshold = proto.minThreshold
    instance.sampleSize = proto.sampleSize
    instance.maxSynapsesPerSegment = proto.maxSynapsesPerSegment

    instance.rng = Random(0).read(proto.random)

    instance.voluntaryActiveCells = np.array(proto.voluntaryActiveCells, dtype='uint32');
    instance.winnerCells = np.array(proto.winnerCells, dtype='uint32');

    instance.activeBasalSegments = np.array(proto.activeBasalSegments, dtype='uint32');
    instance.activeApicalSegments = np.array(proto.activeApicalSegments, dtype='uint32');

    # TD Learning
    instance.TDLearningRate = proto.learningRate
    instance.apicalGlobalDecay = proto.apicalGlobalDecay
    instance.winnerSize = proto.winnerSize
    instance.punishPredDec = proto.punishPredDec
    instance.motorCount = proto.motorCount

    instance.motorCells = np.array(proto.motorCells, dtype='uint32');
    instance.activeMotorCells = np.array(proto.activeMotorCells, dtype='uint32');
    instance.depolarizedApicalCells = np.array(proto.depolarizedApicalCells, dtype='uint32');
    instance.depolarizedBasalCells = np.array(proto.depolarizedBasalCells, dtype='uint32');
    instance.prevApicalInputD1 = np.array(proto.prevApicalInputD1, dtype='uint32');
    instance.prevApicalInputD2 = np.array(proto.prevApicalInputD2, dtype='uint32');

    # instance.apicalConnectionsD1 = SparseMatrixConnections.read(proto.apicalConnectionsD1)
    # instance.apicalConnectionsD2 = SparseMatrixConnections.read(proto.apicalConnectionsD2)
