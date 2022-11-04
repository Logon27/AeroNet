import numpy as np
from scipy.ndimage import rotate, zoom, shift
import random
# For image debugging
from PIL import Image as im

# Helper Functions To Randomize Training Inputs
# These are all currently optimized for mnist

# Rotation method
def randomRotateArray(x):
    x = rotate(x, angle=random.randint(-20, 20), reshape=False)
    return x

# Translation method
def randomShiftArray(x):
    x = shift(x, shift=(random.uniform(-3, 3),random.uniform(-3, 3)))
    return x

# https://stackoverflow.com/questions/54633038/how-to-add-masking-noise-to-numpy-2-d-matrix-in-a-vectorized-manner
# Noise method
def randomNoiseArray(x):
    frac = 0.005
    for i in range(5):
        randomInt = random.randint(50, 255)
        x[np.random.sample(size=x.shape) < frac] = randomInt
    return x

# Method from stackoverflow
# https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
# Scaling method
# Was 0.75, 1.25
def randomClippedZoomArray(img, zoom_factor=random.uniform(0.75, 1.4), **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

# For debug only. Saves numpy array to file.
# You might have to multiple the numpy array by 255 if it was normalized
def saveImage(npArray, fileName):
    # Create an image from the array
    data = im.fromarray(npArray)
    data = data.convert("L")
    
    # Saving the final output to file
    data.save(fileName)
    print("Saved Image... {}".format(fileName))