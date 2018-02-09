import numpy as np
import math

def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    channels = generated_images.shape[1]
    dims = generated_images.shape[2:4]
    image = np.zeros((channels, width*dims[0], height*dims[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/height)
        j = index % height
        image[:, i*dims[0]:(i+1)*dims[0], j*dims[1]:(j+1)*dims[1]] = img[:, :, :]
    image = np.transpose(image, (2, 1, 0))
    return image