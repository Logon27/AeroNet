import numpy as np
from scipy.ndimage import rotate, zoom, shift
import random
# For image debugging
from PIL import Image as im

# Helper Functions To Randomize Training Inputs
# These are all currently optimized for mnist

# Rotation method
def randomRotateArray(x, low_rotate_degrees = -20, high_rotate_degrees = 20):
    x = rotate(x, angle=random.randint(low_rotate_degrees, high_rotate_degrees), reshape=False)
    return x

# Translation method
def randomShiftArray(x, vertical_shift_low = -3, vertical_shift_high = 3, horizontal_shift_low = -3, horizontal_shift_high = 3):
    x = shift(x, shift=(random.uniform(vertical_shift_low, vertical_shift_high),
                        random.uniform(horizontal_shift_low, horizontal_shift_high)))
    return x

# https://stackoverflow.com/questions/54633038/how-to-add-masking-noise-to-numpy-2-d-matrix-in-a-vectorized-manner
# Noise method
def randomNoiseArray(x, percentage_noise = 0.5, num_noise_iterations = 5, noise_value_low = 50, noise_value_high = 255):
    frac = percentage_noise / 100
    # Each iteration only applies a single noise value multiple times.
    # So typically you want a really low noise percentage and a higher number of iterations.
    # That way you get a bunch of unique noise values applied a small number of times.
    for i in range(num_noise_iterations):
        randomInt = random.randint(noise_value_low, noise_value_high)
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

# Used to scale up numpy images for testing
def scaleImage(np_array, scale_factor: int):
    return np.repeat(np.repeat(np_array, scale_factor, axis=0), scale_factor, axis=1)

# For debug only. Saves numpy array to file.
# You might have to multiple the numpy array by 255 if it was normalized
def saveImage(np_array, file_name: str):
    # Create an image from the array
    data = im.fromarray(np_array)
    data = data.convert("L")
    
    # Saving the final output to file
    data.save(file_name)
    print("Saved Image... {}".format(file_name))