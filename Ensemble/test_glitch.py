import cv2
import numpy as np
import sys
import pickle
import importlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np

def print_npy(data):
    if data.dtype.fields is not None:  # It's a structured array
        print("Fields and data:")
        for field in data.dtype.names:
            print(f"Field: {field}, Data: {data[field]}")
    else:  # It's not a structured array
        print("Data:")
        print(data)

test_directory = "test.npy"
model_directory = ""



images = np.load(test_directory)
print_npy(images)
print(images.shape, " ... images are loaded.")
pred = np.zeros(len(images))

# glitches = [ "discoloration","morsecode", "screen_tearing", "random_patch", "parallel_lines", 
#             "shader", "square_patch", "texture_popin",
#             "radial_dotted_line",
#             "shape", "stuttering", "triangulation"]
glitches = [ "screen_tearing"]

gbl = globals()
for glitch in glitches:
    filename = model_directory + glitch + "_test." +  glitch + "_test"
    gbl[filename] = importlib.import_module(filename)
    #import str(filename)
    # print('gbl ', gbl)
    labels = gbl[filename].test(images)

    pred = np.logical_or(pred, labels)
    print('tested ', glitch)

np.save("test_result.npy", pred)
print("saved the result")
