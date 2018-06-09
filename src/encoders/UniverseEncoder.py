import numpy as np
from nupic.data.field_meta import FieldMetaType
from nupic.encoders.base import Encoder

try:
  import capnp
except ImportError:
  capnp = None
if capnp:
  from nupic.encoders.pass_through_capnp import PassThroughEncoderProto


class UniverseEncoder(Encoder):
  """
  Pass an encoded SDR straight to the model.

  Takes raw (768,1024,3) uint8 screen and returns a numpy array with binary pixels of the important parts(field of tasks) of the image preprocessed for the SP.
  The browser window indents the origin of MiniWob by 75 pixels from top and
  10 pixels from the left. The first 50 pixels along height are the query.

  It will crop the image and then return a binary representation of them.
  If the color_option is enabled it will convert the RGB 0-255 colors to an intesity from 1-5 and represent them as a binary array.

  The output without color scheme is (160,160,1) and binary black/white.
  The output with color option is (160,160,3,5) where each pixel is represented by the three color channels and a binary intensity from 1-5.

  :input sensedValue: Observation from the universe environment.

  :param outWidth: the total #bits in output
  :param sparsity: used to normalize the sparsity of the output, exactly w bits ON,
         if None (default) - do not alter the input, just pass it further.
  :param forced: if forced, encode will accept any data, and just return it back
  :param colorOption: defines if it produces a discrete color scheme for each color channel or just binary black/white
  """

  def __init__(self, verbosity=0, sparsity=None, name="universe_encoder", forced=False, colorOption=False, radius=0):
    self.outWidth = 160*160*3*5 if colorOption else 160*160  # array size of output depends if color channels are modeled
    # add more focus to perceptive field around mouse
    if radius>0:
        self.outWidth = self.outWidth + (radius*2) * (radius*2)
    self.radius = radius
    self.sparsity = sparsity
    self.verbosity = verbosity
    self.description = [(name, 0)]
    self.name = name
    self.encoders = None
    self.forced = forced
    self.colorOption = colorOption

    # Save current state for debug
    self.lastRecord = []


  def getDecoderOutputFieldTypes(self):
    """ [Encoder class virtual method override]
    """
    return (FieldMetaType.string,)


  def getWidth(self):
    return self.outWidth

  def getState(self):
    return self.lastRecord

  def getDescription(self):
    return self.description


  def getScalars(self, input):
    """ See method description in base.py """
    return np.array([0])


  def getBucketIndices(self, input):
    """ See method description in base.py """
    return [0]


  def encodeIntoArray(self, inputVal, outputVal, mouseCoords):
    """See method description in base.py"""
    # if len(inputVal) != len(outputVal):
    #   raise ValueError("Different input (%i) and output (%i) sizes." % (
    #       len(inputVal), len(outputVal)))
	#
    # if self.sparsity is not None and sum(inputVal) != self.sparsity:
    #   raise ValueError("Input has %i bits but sparsity was set to %i." % (
    #       sum(inputVal), self.sparsity))
    self.lastRecord = self.applyFilters(inputVal, mouseCoords)
    outputVal[:] = self.lastRecord[:]

    if self.verbosity >= 2:
      print ' ********** ENCODER VERBOSE OUTPUT ********** '
      print "input:", inputVal, "output:", outputVal
      print "decoded:", self.decodedToStr(self.decode(outputVal))


  # Apply filter to data - cut out agents visual field & transform to binary
  def applyFilters(self, data, mouseCoords):
      intensity = 5;
      query_pixels = 50;

      # miniwob task coordinates crop without query = (160,160,3)
      crop = np.array(data[75+query_pixels:75+query_pixels+160, 10:10+160, :])

      if self.colorOption:
          result = np.zeros((160,160,3,intensity));
          crop = crop / (255/intensity+1);    # convert scale 0-255 -> 0-4 intensity for each color channel
          for row in range(160):
              for col in range(160):
                  for color in range(3):
                      result[row,col,color,crop[row,col,color]] = 1;
      else:
           result = self.rgb2gray(crop); 	# greyscale
           # INVERSE, so that white = 0, black = 1 (less on bits, more detailed picture)
           result[result < 128] = 1    # greyscale to binary black/white
           result[result >= 128] = 0

      # If defined append focus - Agent visual field moves with mouse
      if self.radius:
          x1 = max(mouseCoords['x']-self.radius, 10)
          x2 = min(mouseCoords['x']+self.radius, 10+160)
          y1 = max(mouseCoords['y']-self.radius, 75+query_pixels)
          y2 = min(mouseCoords['y']+self.radius, 75+query_pixels+160)
          crop = np.array(data[y1:y2, x1:x2, :])
          focus = self.rgb2gray(crop); 	# greyscale
          focus[focus < 128] = 1    # greyscale INVERSE to binary black/white
          focus[focus >= 128] = 0
          return np.append(result.flatten(),focus.flatten())

      return result.flatten()   # TODO: Topology, do not flatten arrays


  # Helper function
  def rgb2gray(self, rgb):
      return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


  def decode(self, encoded, parentFieldName=""):
    """See the function description in base.py"""

    if parentFieldName != "":
      fieldName = "%s.%s" % (parentFieldName, self.name)
    else:
      fieldName = self.name

    return ({fieldName: ([[0, 0]], "input")}, [fieldName])


  def getBucketInfo(self, buckets):
    """See the function description in base.py"""
    return [EncoderResult(value=0, scalar=0, encoding=np.zeros(self.outWidth))]


  def topDownCompute(self, encoded):
    """See the function description in base.py"""
    return EncoderResult(value=0, scalar=0,
                         encoding=np.zeros(self.outWidth))


  def closenessScores(self, expValues, actValues, **kwargs):
    """
    Does a bitwise compare of the two bitmaps and returns a fractonal
    value between 0 and 1 of how similar they are.

    - ``1`` => identical
    - ``0`` => no overlaping bits

    ``kwargs`` will have the keyword "fractional", which is assumed by this
    encoder.
    """
    ratio = 1.0
    esum = int(expValues.sum())
    asum = int(actValues.sum())
    if asum > esum:
      diff = asum - esum
      if diff < esum:
        ratio = 1 - diff/float(esum)
      else:
        ratio = 1/float(diff)

    olap = expValues & actValues
    osum = int(olap.sum())
    if esum == 0:
      r = 0.0
    else:
      r = osum/float(esum)
    r = r * ratio

    return np.array([r])

  @classmethod
  def getSchema(cls):
    return PassThroughEncoderProto


  @classmethod
  def read(cls, proto):
    encoder = object.__new__(cls)
    encoder.outWidth = proto.outWidth
    encoder.sparsity = proto.sparsity if proto.sparsity else None
    encoder.verbosity = proto.verbosity
    encoder.name = proto.name
    encoder.description = [(encoder.name, 0)]
    encoder.encoders = None
    encoder.forced = proto.forced
    encoder.colorOption = proto.colorOption
    return encoder


  def write(self, proto):
    proto.outWidth = self.outWidth
    if self.sparsity is not None:
      proto.sparsity = self.sparsity
    proto.verbosity = self.verbosity
    proto.name = self.name
    proto.forced = self.forced
    proto.colorOption = self.colorOption
