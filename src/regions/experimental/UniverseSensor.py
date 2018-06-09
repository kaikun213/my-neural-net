import numpy as np
import matplotlib.pyplot as plt

from nupic.bindings.regions.PyRegion import PyRegion

class UniverseSensor(PyRegion):
    """
    UniverseSensor is designed to implement the preprocessing of the observation data (sensory)
    from the universe environment to output a binary array for the SP.

    It's basic functionality is to preprocess the observations into a binary format either grayscale or color option.
    """

    @classmethod
    def getSpec(cls):
        """Return the Spec for PreprocessRegion.
        """
        spec = {
            "description":UniverseSensor.__doc__,
            "singleNodeOnly":True,
            "inputs":{
              "in":{
                "description":"The input vector.",
                "dataType":"Real32",
                "count":0,
                "required":True,
                "regionLevel":False,
                "isDefaultInput":True,
                "requireSplitterMap":False},
            },
            "outputs":{
              "out":{
                "description":"The output without color scheme is (160,160,1) and binary black/white. The output with color option is (160,160,3,intensity).",
                "dataType":"Real32",
                "count":0,
                "regionLevel":True,
                "isDefaultOutput":True},
            },

            "parameters":{
              "verbosity":{
                "description":"Detail of debug print out.",
                "accessMode":"Read",
                "dataType":"UInt32",
                "count":1,
                "constraints":""},
              "colorOption":{
                "description":"Defines if it produces a discrete color scheme for each color channel or just binary black/white.",
                "accessMode":"Read",
                "dataType":"Bool",
                "count":1,
                "constraints":""},
              "colorDiscreteIntensity":{
                "description":"Convert scale of color from 1-255 to binary representation e.g. 1-5 intensity per color channel [0 0 1 0 0] (medium)",
                "accessMode":"Read",
                "dataType":"UInt32",
                "count":1,
                "constraints":""},
            },
        }

        return spec

    def __init__(self, verbosity=0, colorOption=False, colorDiscreteIntensity=5, **kwargs):
        self._colorOption = colorOption;
        self._colorDiscreteIntensity = colorDiscreteIntensity;
        self.dataSource = None;
        self.verbosity = verbosity;
        self._inputWidth = 768*1024*3;    # fixed universe input
        self._dataWidth = 160*160*3*5 if colorOption else 160*160;  # output size dependent on color option

        # lastRecord is the last record returned. Used for debugging only
        self.lastRecord = None

        PyRegion.__init__(self, **kwargs)



    def initialize(self):
        pass

    def getNextRecord(self):
        """
        Get the next observation from the `dataSource` and applying filters.

        TODO: Add a possibility for n-observations to work continously. (n-remotes)
        """

        # Get the data from the dataSource
        data = self.dataSource
        if not data:
            raise StopIteration("Datasource has no more data")

        data = self.applyFilters(data)

        self.lastRecord = data

        return data


    def compute(self, inputs, outputs):
        """
        Run one iteration of UniverseSensor's compute
        """
        data = self.getNextRecord()

        outputs["dataOut"] = data;

    def getOutputElementCount(self, name):
        if name == "dataOut" or name == "out":
            return self._dataWidth;
        elif name == "inputOut":
            return self._inputWidth;
        else:
            raise Exception("Unrecognized output: " + name)

    # Apply filter to data - cut out agents visual field & transform to binary
    def applyFilters(self, data):
        """
        Takes raw (768,1024,3) uint8 screen and returns a numpy array with binary pixels of the important parts(field of tasks) of the image preprocessed for the SP.
        The browser window indents the origin of MiniWob by 75 pixels from top and
        10 pixels from the left. The first 50 pixels along height are the query.

        It will crop the image and then return a binary representation of them.
        If the color_option is enabled it will convert the RGB 0-255 colors to an intesity from 1-5 and represent them as a binary array.

        The output without color scheme is (160,160,1) and binary black/white.
        The output with color option is (160,160,3,5) where each pixel is represented by the three color channels and a binary intensity from 1-5.

        :param dict data: Observation from the universe environment.
        :param bool color_option: Defines if it produces a discrete color scheme for each color channel or just binary black/white.
        :param int color_discrete_intensity: Convert scale of color from 1-255 to binary representation e.g. 1-5 intensity per color channel [0 0 1 0 0] (medium)
        """

        query_pixels = 50;
        crop = np.array(data[75+query_pixels:75+query_pixels+160, 10:10+160, :])  # miniwob coordinates crop without query = (160,160,3)

        if self._colorOption:
            result = np.zeros((160,160,3,self._colorDiscreteIntensity));
            crop = crop / (255/self._colorDiscreteIntensity+1);    # convert scale 0-255 -> 0-4 intensity for each color channel
            for row in range(160):
                for col in range(160):
                    for color in range(3):
                        result[row,col,color,crop[row,col,color]] = 1;
        else:
            result = rgb2gray(crop); # greyscale
            result[result < 128] = 0    # greyscale to binary black/white
            result[result >= 128] = 1

        return result


    # Helper function
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
