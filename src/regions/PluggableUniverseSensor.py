# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

from nupic.bindings.regions.PyRegion import PyRegion
from nupic.encoders.coordinate import CoordinateEncoder
import numpy as np


class PluggableUniverseSensor(PyRegion):
  """
  Slightly modified version of the PluggableEncoderSensor.

  Holds an observation value and a reward and encodes it into network output.

  It requires you to reach in and insert an encoder:

  .. code-block:: python

    timestampSensor = network.addRegion("timestampSensor",
                                      'py.PluggableUniverseSensor', "")
    timestampSensor.getSelf().encoder = DateEncoder(timeOfDay=(21, 9.5),
                                                  name="timestamp_timeOfDay")

  """

  @classmethod
  def getSpec(cls):
    spec = dict(
      description=PluggableUniverseSensor.__doc__,
      singleNodeOnly=True,
      outputs=dict(
        encoded=dict(
          description="The encoded observational data ",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=True),
        resetOut=dict(
          description="Reset signal",
          dataType="Real32",
          count=1,
          regionLevel=True,
          isDefaultOutput=False),
        reward=dict(
          description="The reward from the current iteration. Dictionary with value.",
          dataType='Real32',
          count=0,
          required=False,
          regionLevel=True,
          isDefaultOutput=True)
      ),
      parameters = dict(),
    )
    return spec

  def __init__(self, **kwargs):
    # We don't know the sensed value's type, so it's not a spec parameter.
    self._sensedValue = None
    # coordinate encoder also taking 6400 output SDR with 4% on
    self.coordinateEncoder = CoordinateEncoder(n=80*80, w=257)

  def initialize(self):
    pass

  def compute(self, inputs, outputs):
    if self.encoder is None:
      raise Exception('Please insert an encoder.')

    viualfield = []
    self.encoder.encodeIntoArray(self._sensedValue["observation"],
                                 viualfield,
                                 self._sensedValue['mouse'])
    # append coordinate encoded with radius 3
    npCoords = np.array([self._sensedValue['coordinates']['x'], self._sensedValue['coordinates']['y']]);
    coords = self.coordinateEncoder.encode((npCoords, 3))
    outputs['encoded'][:] = np.append(viualfield,coords);
    outputs['reward'][:] = self._sensedValue["reward"]
    outputs['resetOut'][:] = self._sensedValue["done"]

    # Debug
    print '~~~~~ Sensor Summary ~~~~~'
    print "[Sensor] Inputs:Reward", self._sensedValue['reward']
    print "[Sensor] Outputs:Reward", outputs['reward']
    print "[Sensor] Outputs:Done", outputs['resetOut'][0]
    print "[Sensor] Observation", outputs['encoded'].nonzero()[0]
    print "[Sensor] Observation on bits length", len(outputs['encoded'].nonzero()[0]), 'from total', len(outputs['encoded'])


  def getOutputElementCount(self, name):
    if name == 'encoded':
      return self.encoder.getWidth() + self.coordinateEncoder.getWidth()
    elif name == 'reward':
      return 1
    else:
      raise Exception('Unrecognized output %s' % name)

  def getState(self):
    """
    Returns the current state saved in the encoder (1d numpy array)
    """
    return self.encoder.lastRecord;

  def getSensedValue(self):
    """
    :return: sensed value
    """
    return self._sensedValue

  def setSensedValue(self, value):
    """
    :param value: will be encoded when this region does a compute.
    """
    self._sensedValue = value

  def getParameter(self, parameterName, index=-1):
    if parameter == 'sensedValue':
      raise Exception('For the PluggableUniverseSensor, get the sensedValue via the getSensedValue method')
    else:
      raise Exception('Unrecognized parameter %s' % parameterName)

  def setParameter(self, parameterName, index, parameterValue):
    if parameter == 'sensedValue':
      raise Exception('For the PluggableUniverseSensor, set the sensedValue via the setSensedValue method')
    else:
      raise Exception('Unrecognized parameter %s' % parameterName)
