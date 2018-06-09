_VERBOSITY = 0;

# L4
_L4_columnCount = 512;
_L4_potentialRadius = int(0.25 * _L4_columnCount);

# L3/L2
_L3_L2_columnCount = 512;
_L3_L2_potentialRadius = int(0.5 * _L3_L2_columnCount);

# L5
_L5_columnCount = 512; # also for D1 & D2
_L5_potentialRadius = int(0.5 * _L5_columnCount);

# Motor (TODO: Do mapping in motor layer, currently has to be L5 size)
_MOTOR_NEURON = 32;
_MOTOR_WINNER_CELLS = 6;

# TD(Lambda)
_TDTraceDecay = 0.1 # Lambda
_TDDiscount = 0.1
_TDLearningRate = 0.1
_TDValueDecay = 0.0001


SENSOR_PARAMS = {
    "verbosity": _VERBOSITY,
    "colorOption":False,
    "name":'encoder',
    "radius":40 # - limit agents field to 30px radius
}

MOTOR_PARAMS = {
    # Modified TM params
    "basalInputWidth":0,
    "apicalInputWidth":0,
    "apicalGlobalDecay":0.00001,
    "TDLearningRate":_TDLearningRate,
    "winnerSize":_MOTOR_WINNER_CELLS,
    "motorCount":_MOTOR_NEURON,
    "synPermActiveIncMotor":0.04,
    "synPermInactiveDecMotor":0.08,
    "punishPredDec":0.004,  # 0.004 on apical


    # TM params
    "columnCount":_L5_columnCount,
    "cellsPerColumn":32,
    "activationThreshold":9,
    "initialPermanence":0.21,
    "connectedPermanence":0.25,
    "minThreshold":6,
    # "maxNewSynapseCount":20,
    "maxSynapsesPerSegment": -1, #255, => TODO: Not sure if they get removed, so old ones could fill up 255
    "seed":42,

    # Region params
    "learn":True
}

TP_PARAMS = {
    # ~~~~~~ Temporal Pooler Region ~~~~~~
    "columnCount":_L3_L2_columnCount,
    #"inputWidth":
    "historyLength":1,
    "minHistory":0,
    "poolerType":"union",
    "learningMode":1,
    "inferenceMode":0,   # no need to output inference (speed up)

    # ~~~~~~ Union Temporal Pooler ~~~~~~
    "activeOverlapWeight":0.4,
    "predictedActiveOverlapWeight":0.6,
    #"maxUnionActivity":0.20
    "exciteFunctionType":'Linear',
    "decayFunctionType":'Linear',
    #"decayTimeConst":20.0
    "decayLinearConst":0.9,
    #"synPermPredActiveInc":0.0
    #"synPermPreviousPredActiveInc":0.0
    #"historyLength":0
    #"minHistory":0

    # ~~~~~~ Underlying Spatial Pooler ~~~~~~
    "spVerbosity": _VERBOSITY,
    "seed": 2001,
    # spike_type: activation
    # synapse_max_perm: 1
    "synPermConnected": 0.2,
    "synPermActiveInc": 0.04,
    "synPermInactiveDec": 0.004,
    # base_perm_inc: 0.0004
    "globalInhibition": 1, # global potential
    "localAreaDensity":0.04,
    "numActiveColumnsPerInhArea":-1,
    "potentialRadius":_L3_L2_potentialRadius, # receptive field size "0.5" percent of all input
    "potentialPct": 0.15, # connection percentage = 0.15 (Percentage of initially randomly connections in perceptive field)
    "boostStrength": 0.0,    # active boost = false ? (boost strength = 4)
}

L4_SP_PARAMS = {
    "columnCount": _L4_columnCount,
    "spatialImp": "cpp",
    "spVerbosity": _VERBOSITY,
    "seed": 1956,

    # spike_type: activation
    # synapse_max_perm: 1
    "synPermConnected": 0.2,
    "synPermActiveInc": 0.02,
    "synPermInactiveDec": 0.02,
    # base_perm_inc: 0.0004
    "globalInhibition": 1, # global potential
    "localAreaDensity":0.04,
    "numActiveColumnsPerInhArea":-1,
    "potentialRadius":_L4_potentialRadius, # potential percentage - receptive field size "0.5" percent of all input
    "potentialPct": 0.15, # connection percentage = 0.15 (Percentage of initially randomly connections in perceptive field)
    "boostStrength": 0.0,    # active boost = false ? (boost strength = 4)
}

L4_ETM_PARAMS = {
    # spike_type: activation
    # random distal connections = False
    # single cell learning = True
    # "globalDecay": 0.000001,
    # apical inactive decrement: 0.0008
    "learn": True,
    "columnCount": _L4_columnCount,
    #"columnDimensions":
    "basalInputWidth": _MOTOR_NEURON,
    #"apicalInputWidth":
    "cellsPerColumn":32,
    "activationThreshold":2,
    "initialPermanence":0.2,
    "connectedPermanence":0.2,
    "minThreshold":1,
    # "maxNewSynapseCount":3,
    "maxSegmentsPerCell":128,
    "maxSynapsesPerSegment":128,
    "permanenceIncrement":0.04,
    "permanenceDecrement":0.08,
    "predictedSegmentDecrement":0.0008, # inactive decrement (basal)
    "seed":1960,
    "formInternalBasalConnections": False, # Only Motor Basal Input
    #"learnOnOneCell":
    #"implementation":
    #"checkInputs":
}

L4_WEIGHTED_PARAMS = {
    # spike_type: activation
    # random distal connections = False
    # single cell learning = True
    # "globalDecay": 0.000001,
    # apical inactive decrement: 0.0008
    "learn": True,
    "columnCount": _L4_columnCount,
    "basalInputWidth": _L4_columnCount * 32, # input (t-1) Layer 4
    "apicalInputWidth": _MOTOR_NEURON,
    "cellsPerColumn":32,
    "initialPermanence":0.2,
    "connectedPermanence":0.2,
    # "maxNewSynapseCount":3,
    "maxSynapsesPerSegment":-1,
    "permanenceIncrement":0.04,
    "permanenceDecrement":0.08,
    "seed":1960,
    
   # apical, basal weighting
    "activationThresholdBasal":13,
    "activationThresholdApical":2,
    "minThresholdBasal":10,
    "minThresholdApical":1,
    "basalPredictedSegmentDecrement":0.0008, # inactive decrement (basal)
    "apicalPredictedSegmentDecrement":0.0001
}

DEFAULT_SP_PARAMS = {
    # ~~~~~~ Spatial Pooler ~~~~~~
    "columnCount": 512,
    "spatialImp": "cpp",
    "inputWidth": 512*32,
    "spVerbosity": _VERBOSITY,
    "seed": 2001,
    # spike_type: activation
    # synapse_max_perm: 1
    "synPermConnected": 0.2,
    "synPermActiveInc": 0.04,
    "synPermInactiveDec": 0.004,
    # base_perm_inc: 0.0004
    "globalInhibition": 1, # global potential
    "localAreaDensity":0.04,
    "numActiveColumnsPerInhArea":-1,
    "potentialRadius":_L5_potentialRadius, # receptive field size "0.5" percent of all input
    "potentialPct": 0.15, # connection percentage = 0.15 (Percentage of initially randomly connections in perceptive field)
    "boostStrength": 0.0    # active boost = false ? (boost strength = 4)
}


DEFAULT_TM_PARAMS = {
    "columnCount": 512,
    "cellsPerColumn": 32,
    "inputWidth": 512*32,
    "verbosity": _VERBOSITY,
    "temporalImp": "cpp",
    "seed": 1933,
    "computePredictedActiveCellIndices":True,
    "learningMode":True,

    # spike_type: activation
    "permanenceMax": 10, # synapse max. permanence = 10
    # synapse connection permanence = 0.2
    "initialPerm": 0.2,
    "connectedPerm":0.2,
    "permanenceInc": 0.04,
    "permanenceDec": 0.08,
    # inactive decrement = 0.0008 VS 0.004 on APICAL (TM Paper "predicted_decrement")
    "globalDecay": 0.000001,
    # "maxSegmentsPerCell": 128,
    "newSynapseCount": 12,
    "minThreshold": 6,  # matching threshold
    "activationThreshold": 9,
    # random distal connections = True
    # single cell learning = True
    "outputType": "normal",
    "pamLength": 3,
}

DEFAULT_ETM_PARAMS = {
    # spike_type: activation
    # random distal connections = False
    # single cell learning = True
    # "globalDecay": 0.000001, -> Motor Layer for apical (basal missing)
    # apical inactive decrement: 0.004 -> In Motor Layer
    "learn": True,
    "columnCount": _L5_columnCount,
    #"columnDimensions":
    "basalInputWidth": _L4_columnCount*32 + _L3_L2_columnCount*32,
    #"apicalInputWidth":
    "cellsPerColumn":32,
    "activationThreshold":4,
    "initialPermanence":0.2,
    "connectedPermanence":0.2,
    "minThreshold":2,
    "maxNewSynapseCount":3,
    "maxSegmentsPerCell":128,
    "maxSynapsesPerSegment":128,
    "permanenceIncrement":0.04,
    "permanenceDecrement":0.08,
    "predictedSegmentDecrement":0.0008, # inactive decrement (basal)
    "seed":1960,
    "formInternalBasalConnections": False, # Use L4
    #"learnOnOneCell":
    #"implementation":
    #"checkInputs":
}

TD_ETM_PARAMS = {
    # spike_type: activation
    # random distal connections = False
    # single cell learning = True
    # "globalDecay": 0.000001, -> Motor Layer for apical (basal missing)
    # apical inactive decrement: 0.004 -> In Motor Layer
    "learn": True,
    "columnCount": _L5_columnCount,
    #"columnDimensions":
    "basalInputWidth": _L4_columnCount*32 + _L3_L2_columnCount*32,
    #"apicalInputWidth":
    "cellsPerColumn":32,
    "activationThreshold":4,
    "initialPermanence":0.2,
    "connectedPermanence":0.2,
    "minThreshold":2,
    "maxNewSynapseCount":3,
    "maxSegmentsPerCell":128,
    "maxSynapsesPerSegment":128,
    "permanenceIncrement":0.04,
    "permanenceDecrement":0.08,
    "predictedSegmentDecrement":0.0008, # inactive decrement (basal)
    "seed":1960,
    "formInternalBasalConnections": False, # Only L4/L2
    #"learnOnOneCell":
    #"implementation":
    #"checkInputs":
    "TDTraceDecay":_TDTraceDecay,
    "TDDiscount":_TDDiscount,
    "TDLearningRate":_TDLearningRate,
    "globalValueDecay":_TDValueDecay
}
