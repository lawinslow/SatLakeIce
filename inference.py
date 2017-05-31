
# Run this script with an argument pointing to a specific file
# or folder of GDAL images containing a lake to classify it as frozen or not

import sys
import os
import joblib
import numpy as np
from keras.models import load_model
from geotiff_convert import convert_geotifs

def convert(path):

    if os.path.isfile(path):
        # user sent a path to a file
        data = convert_geotifs([path])
    elif os.path.isdir(path):
        # user sent directory of geotifs
        paths = [path + "/" + p for p in os.listdir(path)]
        data = convert_geotifs(paths)
    
    return data

def run(path = None):
    """
    Converts gdal files at path and runs inference on them.
    """

    # find the trained model path and load it
    trained_model_path = "./trained/" + os.listdir("./trained")[0]

    # load the X_max value so we can normalize properly
    X_max = joblib.load("./data/X_max.joblib")

    model = load_model(trained_model_path)

    if path is not None:
        data = convert(path)

        data /= X_max

        predicted = model.predict(data)
        frozen = np.zeros_like(predicted, dtype = np.bool)

        frozen[predicted > 0.5] = True
        print(frozen)

    else:
        test_summer_data = joblib.load("./data/test_summer_data.joblib")
        test_winter_data = joblib.load("./data/test_winter_data.joblib")

        test_summer_data /= X_max
        test_winter_data /= X_max

        # labels:
        # winter = 1, summer = 0
        summer_acc = model.evaluate(test_summer_data, [0 for _ in range(len(test_summer_data))], verbose = 0)
        winter_acc = model.evaluate(test_winter_data, [1 for _ in range(len(test_winter_data))], verbose = 0)

        # first value is loss, second is accuracy, which we actually want
        print("accuracy on summer images: {:.2f}".format(summer_acc[1]))
        print("accuracy on winter images: {:.2f}".format(winter_acc[1]))
        


if __name__ == "__main__":

    # if a path to image(s) is provided,
    # assume user wants a classification result on them,
    # otherwise evaluate the test data
    if len(sys.argv) == 1:
        run()
    else:
        run(sys.argv[1])

