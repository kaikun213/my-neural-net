import json
import os
import datetime
import numpy as np
from nupic.engine import Network

# Regions for network
from regions.PluggableUniverseSensor import PluggableUniverseSensor
from encoders.UniverseEncoder import UniverseEncoder
from regions.SensoryIntegrationRegion import SensoryIntegrationRegion
from regions.AgentStateRegion import AgentStateRegion
from regions.ReinforcementRegion import ReinforcementRegion
from regions.MyTemporalPoolerRegion import MyTemporalPoolerRegion
from regions.MotorRegion import MRegion

# copies of TMRegion and SPRegion for simple customization
from regions.MyTMRegion import MyTMRegion
from regions.MySPRegion import MySPRegion

from htmresearch.regions.ExtendedTMRegion import ExtendedTMRegion

# Model params
from config.modelParams import SENSOR_PARAMS, MOTOR_PARAMS, TP_PARAMS, \
                        L4_SP_PARAMS, L4_ETM_PARAMS, L4_WEIGHTED_PARAMS, \
                        DEFAULT_SP_PARAMS, DEFAULT_TM_PARAMS, \
                        DEFAULT_ETM_PARAMS, TD_ETM_PARAMS

# Schemas (Proto) for serialization
from nupic.proto.SpatialPoolerProto_capnp import SpatialPoolerProto
from htmresearch_core.proto.ExtendedTemporalMemoryProto_capnp import ExtendedTemporalMemoryProto

from proto.ReinforcementRegionProto_capnp import ReinforcementRegionProto
from proto.AgentStateRegionProto_capnp import AgentStateRegionProto
from proto.MyTemporalPoolerRegionProto_capnp import MyTemporalPoolerRegionProto
from proto.MotorRegionProto_capnp import MotorRegionProto

# Algorithms for deserialization
from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.algorithms.backtracking_tm_cpp import BacktrackingTMCPP
from htmresearch_core.experimental import ExtendedTemporalMemory

def createNetwork():
    net = Network();

    # Register Regions
    Network.registerRegion(PluggableUniverseSensor);
    Network.registerRegion(AgentStateRegion);
    Network.registerRegion(ReinforcementRegion);
    Network.registerRegion(MRegion);
    Network.registerRegion(SensoryIntegrationRegion);

    # HTMResearch regions
    Network.registerRegion(ExtendedTMRegion);
    Network.registerRegion(MyTMRegion);
    Network.registerRegion(MySPRegion);
    Network.registerRegion(MyTemporalPoolerRegion);

    sensor = net.addRegion("sensor", 'py.PluggableUniverseSensor', "").getSelf();
    sensor.encoder = UniverseEncoder(**SENSOR_PARAMS)

    # ~~~~~~~~~~~~~~~ Create Regions ~~~~~~~~~~~~~~~
    # Make sure the SP input width matches the sensor output width.
    L4_SP_PARAMS["inputWidth"] = sensor.encoder.getWidth()
    L4_SP = net.addRegion("L4_SP", "py.MySPRegion", json.dumps(L4_SP_PARAMS)).getSelf();
    #L4_TM = net.addRegion("L4_TM", "py.ExtendedTMRegion", json.dumps(L4_TM_PARAMS)).getSelf();
    L4_TM = net.addRegion("L4_TM", "py.SensoryIntegrationRegion", json.dumps(L4_WEIGHTED_PARAMS)).getSelf();

    TP_PARAMS["inputWidth"] = L4_TM.getOutputElementCount("activeCells")
    L3_TP = net.addRegion("L3_TP","py.MyTemporalPoolerRegion", json.dumps(TP_PARAMS)).getSelf();
    L3_TM = net.addRegion("L3_TM", "py.MyTMRegion", json.dumps(DEFAULT_TM_PARAMS)).getSelf();
    #net.regions["L3_TM"].setParameter("inferenceMode", True)

    TP_PARAMS["inputWidth"] = L3_TM.getOutputElementCount("activeCells")
    L2_TP = net.addRegion("L2_TP","py.MyTemporalPoolerRegion", json.dumps(TP_PARAMS)).getSelf();
    L2_TM = net.addRegion("L2_TM", "py.MyTMRegion", json.dumps(DEFAULT_TM_PARAMS)).getSelf();
    net.regions["L2_TM"].setParameter("inferenceMode", True)
    #  DEFAULT_ETM_PARAMS["basalInputWidth"] = L4_TM.getOutputElementCount("activeCells")

    DEFAULT_SP_PARAMS["inputWidth"] = L4_TM.getOutputElementCount("activeCells")
    L5_SP = net.addRegion("L5_SP","py.MySPRegion", json.dumps(DEFAULT_SP_PARAMS)).getSelf();
    DEFAULT_ETM_PARAMS["basalInputWidth"] = L4_TM.getOutputElementCount("activeCells") + L2_TM.getOutputElementCount("activeCells")
    L5_TM = net.addRegion("L5_TM", "py.AgentStateRegion", json.dumps(DEFAULT_ETM_PARAMS)).getSelf();


    DEFAULT_SP_PARAMS["inputWidth"] = L5_SP.getOutputElementCount("bottomUpOut")
    D1_SP = net.addRegion("D1_SP","py.MySPRegion", json.dumps(DEFAULT_SP_PARAMS)).getSelf();
    TD_ETM_PARAMS["basalInputWidth"] = L5_TM.getOutputElementCount("activeCells")
    D1_TM = net.addRegion("D1_TM", "py.ReinforcementRegion", json.dumps(TD_ETM_PARAMS)).getSelf();

    D2_SP = net.addRegion("D2_SP","py.MySPRegion", json.dumps(DEFAULT_SP_PARAMS)).getSelf();
    D2_TM = net.addRegion("D2_TM", "py.ReinforcementRegion", json.dumps(TD_ETM_PARAMS)).getSelf();

    MOTOR_PARAMS["basalInputWidth"] = L5_TM.getOutputElementCount("activeCells")
    MOTOR_PARAMS["apicalInputWidth"] = D1_TM.getOutputElementCount("activeCells")
    Motor = net.addRegion("Motor", "py.MRegion", json.dumps(MOTOR_PARAMS)).getSelf();

    # ~~~~~~~~~~~~~~~ Link Regions ~~~~~~~~~~~~~~~
    # L4: Proximal: Sensor -> L4
    #     Distal:   Motor -> L4 (alt. Motor & Sensor for distal input)
    net.link("sensor", "L4_SP", "UniformLink", "",
             srcOutput="encoded", destInput="bottomUpIn")
    net.link("L4_SP", "L4_TM", "UniformLink", "",
             srcOutput="bottomUpOut", destInput="activeColumns")
    net.link("Motor", "L4_TM", "UniformLink", "",
             srcOutput="winnerCells", destInput="basalInput")

    # Temporal Pooling --> Overlap_i = (PrevOverlap_i + Overlap_i) * Overlap_decay (0.9 in this study)
    # L3: Proximal: L4 -> L3
    #     Distal:   L3 -> L3
    net.link("L4_TM", "L3_TP", "UniformLink", "",
             srcOutput="activeCells", destInput="activeCells")
    net.link("L4_TM", "L3_TP", "UniformLink", "",
             srcOutput="predictedActiveCells", destInput="predictedActiveCells")
    net.link("L3_TP", "L3_TM", "UniformLink", "",
             srcOutput="mostActiveCells", destInput="bottomUpIn")

    # L2: Proximal: L3 -> L2
    #     Distal:   L2 -> L2
    net.link("L3_TM", "L2_TP", "UniformLink", "",
             srcOutput="activeCells", destInput="activeCells")
    net.link("L3_TM", "L2_TP", "UniformLink", "",
             srcOutput="predictedActiveCells", destInput="predictedActiveCells")
    net.link("L2_TP", "L2_TM", "UniformLink", "",
             srcOutput="mostActiveCells", destInput="bottomUpIn")

    # L5: Proximal: L4 -> L5
    #     Distal:   L4(t-1), L2(t-1) -> L5
    net.link("L4_TM", "L5_SP", "UniformLink", "",
             srcOutput="activeCells", destInput="bottomUpIn")
    net.link("L5_SP", "L5_TM", "UniformLink", "",
             srcOutput="bottomUpOut", destInput="activeColumns")
    # net.link("L2_TM", "L5_TM", "UniformLink", "",
    #          srcOutput="activeCells", destInput="basalInput")
    net.link("L4_TM", "L5_TM", "UniformLink", "",
             srcOutput="activeCells", destInput="basalInput")

    # D1: Proximal: L5 -> D1
    #     Distal:   L5 -> D1
    net.link("L5_SP", "D1_SP", "UniformLink", "",
             srcOutput="bottomUpOut", destInput="bottomUpIn")
    net.link("D1_SP", "D1_TM", "UniformLink", "",
             srcOutput="bottomUpOut", destInput="activeColumns")
    net.link("L5_TM", "D1_TM", "UniformLink", "",
             srcOutput="activeCells", destInput="basalInput")
    net.link("sensor", "D1_TM", "UniformLink", "",
             srcOutput="reward", destInput="reward")

    # D2: Proximal: L5 -> D2
    #     Distal:   L5 -> D2
    net.link("L5_SP", "D2_SP", "UniformLink", "",
             srcOutput="bottomUpOut", destInput="bottomUpIn")
    net.link("D2_SP", "D2_TM", "UniformLink", "",
             srcOutput="bottomUpOut", destInput="activeColumns")
    net.link("L5_TM", "D2_TM", "UniformLink", "",
             srcOutput="activeCells", destInput="basalInput")
    net.link("sensor", "D2_TM", "UniformLink", "",
             srcOutput="reward", destInput="reward")

    # Motor: Apical: D1, D2 -> Motor (TD Learning)
    #        State-copy: L5 SP and Neural, Basal activation
    #        Extra: (TDError) D1,D2 -> Motor
    net.link("D1_TM", "Motor", "UniformLink", "",
             srcOutput="activeCells", destInput="apicalInputD1")
    net.link("D2_TM", "Motor", "UniformLink", "",
             srcOutput="activeCells", destInput="apicalInputD2")
    net.link("D1_TM", "Motor", "UniformLink", "",
             srcOutput="TDError", destInput="TDErrorD1")
    net.link("D2_TM", "Motor", "UniformLink", "",
             srcOutput="TDError", destInput="TDErrorD2")
    # # Copy of L5 state (active columns and depolarizedBasalCells)
    net.link("L5_SP", "Motor", "UniformLink", "",
             srcOutput="bottomUpOut", destInput="activeColumns")
    net.link("L5_TM", "Motor", "UniformLink", "",
             srcOutput="predictedCells", destInput="depolarizedBasalCells")
    net.link("L5_TM", "Motor", "UniformLink", "",
             srcOutput="activeCells", destInput="activeCells")

    # Link Sensor.resetOut -> resetIn (All)
    net.link("sensor", "L4_SP", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    net.link("sensor", "L4_TM", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    net.link("sensor", "L3_TP", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    net.link("sensor", "L3_TM", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    net.link("sensor", "L2_TP", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    net.link("sensor", "L2_TM", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    net.link("sensor", "L5_SP", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    net.link("sensor", "L5_TM", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    net.link("sensor", "D1_SP", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    net.link("sensor", "D1_TM", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    net.link("sensor", "D2_SP", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    net.link("sensor", "D2_TM", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    net.link("sensor", "Motor", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")

    layers = extractRegions(net)
    return (net, layers)


def extractRegions(net):
    """ Extract Network Regions for internal access """
    sensor = net.regions["sensor"].getSelf()
    L4_SP = net.regions["L4_SP"].getSelf()
    L4_TM = net.regions["L4_TM"].getSelf()
    L3_TP = net.regions["L3_TP"].getSelf()
    L3_TM = net.regions["L3_TM"].getSelf()
    L2_TP = net.regions["L2_TP"].getSelf()
    L2_TM = net.regions["L2_TM"].getSelf()
    D1_SP = net.regions["D1_SP"].getSelf()
    D1_TM = net.regions["D1_TM"].getSelf()
    D2_SP = net.regions["D2_SP"].getSelf()
    D2_TM = net.regions["D2_TM"].getSelf()
    L5_SP = net.regions["L5_SP"].getSelf()
    L5_TM = net.regions["L5_TM"].getSelf()
    Motor = net.regions["Motor"].getSelf()

    layers = {
        'sensor':sensor,
        'L4_SP':L4_SP,
        'L4_TM':L4_TM,
        'L3_TP':L3_TP,
        'L3_TM':L3_TM,
        'L2_TP':L2_TP,
        'L2_TM':L2_TM,
        'D1_SP':D1_SP,
        'D1_TM':D1_TM,
        'D2_SP':D2_SP,
        'D2_TM':D2_TM,
        'L5_SP':L5_SP,
        'L5_TM':L5_TM,
        'Motor':Motor}

    return layers


def saveNetwork(layers, networkDirName='networks'):
    """ Serialize the network
        TODO: Change to cap'n proto
    """
    # network name to save
    datetimestr = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    netName = "%s/%s_agent_net" % (networkDirName, datetimestr)

    # If necessary create networks dir
    if not os.path.exists(networkDirName):
        os.makedirs(networkDirName)
    os.makedirs(netName)

    # Go through the regions, extract the algorithm instances and save them
    for key, layer in layers.iteritems():
        filename = "%s/%s.tmp" % (netName, key)
        with open(filename, "wb") as f:
            # AgentStateRegion (ETM instance)
            if 'L5_TM' in key:
                print 'save AgentStateRegion', layer.__class__
                proto = AgentStateRegionProto.new_message()
                layer.writeToProto(proto)
                proto.write_packed(f)
            # ReinforcementRegion (ETM instance)
            elif any(name in key for name in ['D1_TM', 'D2_TM']):
                print 'save ReinforcementRegion', layer.__class__
                proto = ReinforcementRegionProto.new_message() # load schema
                layer.writeToProto(proto)
                proto.write_packed(f)
            # SPRegion - cpp instance (L4,L5,D1,D2)
            elif 'SP' in key:
                proto = SpatialPoolerProto.new_message()
                instance = layer.getAlgorithmInstance()
                instance.write(proto)
                print 'save instance', key,  instance.__class__
                proto.write_packed(f)
            # MyTMRegion - cpp backtrackingTM instance (L2,L3)
            elif any(name in key for name in ['L2_TM', 'L3_TM']):
                instance = layer.getAlgorithmInstance()
                print 'save MyTMRegion', key, instance.__class__
                instance.writeToFile(f)
            # ExtendedTemporalMemory - cpp instance (L4,L5,D1,D2)
            elif 'TM' in key:
                proto = ExtendedTemporalMemoryProto.new_message() # load schema
                instance = layer.getAlgorithmInstance()
                instance.write(proto) # populate message
                proto.write_packed(f) # write to file
                print 'save ETM', key, instance.__class__
            # UnionPooler based on cpp spatial pooler (L2,L3)
            elif 'TP' in key:
                print 'save instance', key,  layer.__class__
                proto = MyTemporalPoolerRegionProto.new_message() # load schema
                layer.writeToProto(proto)
                proto.write_packed(f)
            elif 'sensor' in key:
                continue
            # MotorRegion - MotorTM instance (ApicalDependentTemporalMemory)
            elif 'Motor' in key:
                print 'save Motor', layer.__class__
                proto = MotorRegionProto.new_message() # load schema
                layer.writeToProto(proto)
                proto.write_packed(f)

    return netName

def loadNetwork(path):
    """ Deserialize the network from the given path
    """
    # Create the network structure
    (net, layers) = createNetwork();

    # Replace algorithm components with loaded (connections etc.)
    for key, layer in layers.iteritems():
        filename = "%s/%s.tmp" % (path, key)
        print 'load', key, filename
        with open(filename, "rb") as f:
            # AgentStateRegion with ETM instance
            if 'L5_TM' in key:
                proto = AgentStateRegionProto.read_packed(f);
                layers[key] = AgentStateRegion.readFromProto(proto);
            # ReinforcementRegion with ETM instance
            elif any(name in key for name in ['D1_TM', 'D2_TM']):
                proto = ReinforcementRegionProto.read_packed(f);
                layers[key] = ReinforcementRegion.readFromProto(proto);
            # TemporalPoolerRegion using UnionTemporalPooler (extended Spatial Pooler)
            elif 'TP' in key:
                proto = MyTemporalPoolerRegionProto.read_packed(f);
                layers[key] = MyTemporalPoolerRegion.readFromProto(proto);
            elif 'Motor' in key:
                proto = MotorRegionProto.read_packed(f);
                layers[key] = MRegion.readFromProto(proto);
            else:
            # MySPRegion - sp cpp instance (L4,L5,D1,D2)
                if 'SP' in key:
                    proto = SpatialPoolerProto.read_packed(f);
                    instance = SpatialPooler.read(proto);
                # MyTMRegion - cpp backtrackingTM instance (L2,L3)
                elif any(name in key for name in ['L2_TM', 'L3_TM']):
                    instance = BacktrackingTMCPP.readFromFile(f);
                # ExtendedTemporalMemory - cpp instance (L4,L5,D1,D2)
                elif 'TM' in key:
                    # Read from file - schema.read(file)
                    proto = ExtendedTemporalMemoryProto.read_packed(f)
                    # Generate class instance from the schema
                    instance = ExtendedTemporalMemory.read(proto)
                elif 'sensor' in key:
                    continue
                layer.setAlgorithmInstance(instance)

    return (net,layers)
