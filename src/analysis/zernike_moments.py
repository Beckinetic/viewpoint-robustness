import numpy as np
import cv2
import mahotas
from skimage import io, color
from skimage.measure import regionprops, label
from skimage.feature import shape_index
from skimage.metrics import structural_similarity as ssim


def compute_zernike_moments(image_path, radius=100):
    """
    Computes the Zernike moments of an object in a given image.

    Parameters:
        image_path (str): The path to the input image (.png format).
        radius (int): The radius for Zernike moments calculation. Default is 100.

    Returns:
        zernike_moments (list): A list of Zernike moments.
    """
    # Read the image
    image = io.imread(image_path, as_gray=True)

    # Threshold the image to create a binary mask
    _, binary_image = cv2.threshold((image * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

    # Label the connected components in the binary image
    labeled_image, num_features = label(binary_image, return_num=True, connectivity=2)

    # Extract the largest component assuming it is the object of interest
    if num_features > 1:
        regions = regionprops(labeled_image)
        largest_region = max(regions, key=lambda region: region.area)
        binary_image = (labeled_image == largest_region.label).astype(np.uint8)

    # Ensure the binary image is padded to be square for Zernike computation
    padded_image = np.pad(binary_image, ((radius, radius), (radius, radius)), mode='constant', constant_values=0)

    # Compute the Zernike moments
    zernike_moments = mahotas.features.zernike_moments(padded_image, radius)

    return zernike_moments
