import argparse
import glob
import os

import yaml


import numpy as np
import skimage as sk

from skimage.filters import gaussian
from io import BytesIO

from tqdm import tqdm

import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings

warnings.simplefilter("ignore", UserWarning)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


#=================== Distortion Helper Functions ===================

def auc(errs):  # area under the alteration error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


#=================== Distortion Functions ===================


def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


# def fgsm(x, source_net, severity=1):
#     c = [8, 16, 32, 64, 128][severity - 1]
#
#     x = V(x, requires_grad=True)
#     logits = source_net(x)
#     source_net.zero_grad()
#     loss = F.cross_entropy(logits, V(logits.data.max(1)[1].squeeze_()), size_average=False)
#     loss.backward()
#
#     return standardize(torch.clamp(unstandardize(x.data) + c / 255. * unstandardize(torch.sign(x.grad.data)), 0, 1))


def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=-1) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(224 - c[1], c[1], -1):
            for w in range(224 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1):
    # Define severity levels for radius (kernel size) and angle range for motion blur
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    # Create a random angle for the motion blur effect
    angle = np.random.uniform(-45, 45)

    # Convert image to a NumPy array if it's not already
    if isinstance(x, PILImage.Image):
        x = np.array(x)

    # Set up the motion blur kernel
    kernel_size = c[0]
    kernel = np.zeros((kernel_size, kernel_size))

    # Create a linear motion blur kernel
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size

    # Rotate the kernel to apply motion blur in different directions
    center = (kernel_size // 2, kernel_size // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))

    # Apply the motion blur kernel to the image
    output = cv2.filter2D(x, -1, kernel)

    # If the image is grayscale, convert it to RGB
    if len(output.shape) == 2:
        output = np.stack([output, output, output], axis=-1)

    # Clip values to ensure valid pixel range and convert back to uint8
    return np.clip(output, 0, 255).astype(np.uint8)


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:256, :256][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    filename = ['frost1.png', 'frost2.png', 'frost3.png', 'frost4.jpg', 'frost5.jpg', 'frost6.jpg'][idx]
    frost = cv2.imread(filename)

    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 256), np.random.randint(0, frost.shape[1] - 256)
    frost = frost[x_start:x_start + 256, y_start:y_start + 256][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def clipped_zoom(img, zoom_factor):
    # Clipping zoom function to simulate zooming in and out
    h, w = img.shape[:2]
    # Crop the image according to the zoom factor
    ch = int(np.round(h / zoom_factor))
    cw = int(np.round(w / zoom_factor))
    top = (h - ch) // 2
    left = (w - cw) // 2

    img_cropped = img[top:top + ch, left:left + cw]

    # Resize it back to original dimensions
    return cv2.resize(img_cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def snow(x, severity=1):
    # Parameters based on severity level
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    # Normalize the input image
    x = np.array(x, dtype=np.float32) / 255.

    # Create a snow layer using random noise
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    # Apply zoom effect to the snow layer
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])

    # Apply threshold to simulate snowflakes
    snow_layer[snow_layer < c[3]] = 0

    # Convert snow_layer to uint8 for OpenCV processing
    snow_layer = (np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8)

    # Apply motion blur to the snow layer using OpenCV
    kernel_size = int(c[4])
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    snow_layer = cv2.filter2D(snow_layer, -1, kernel_motion_blur)

    # Normalize the snow layer for blending
    snow_layer = snow_layer / 255.0
    snow_layer = snow_layer[..., np.newaxis]  # Add channel back

    # Blend the snow layer with the original image
    grayscale_x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(x.shape[0], x.shape[1], 1)
    blended_image = c[6] * x + (1 - c[6]) * np.maximum(x, grayscale_x * 1.5 + 0.5)

    # Add the snow layer to the blended image
    result = np.clip(blended_image + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255

    return result.astype(np.uint8)


def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x = x.resize((int(224 * c), int(224 * c)), PILImage.BOX)
    x = x.resize((224, 224), PILImage.BOX)

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),  # 244 should have been 224, but ultimately nothing is incorrect
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


def parse_args():
    parser = argparse.ArgumentParser(description='Run picture distortion')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    return parser.parse_args()


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def load_image(image_path):
    with open(image_path, 'rb') as f:
        img = PILImage.open(f)
        return img.convert('RGB')


def apply_distortion(image_paths, output_folder, distortion_function, severity):
    for image_path in image_paths:
        image = load_image(image_path)
        output = distortion_function(image, severity)
        # Check if the output is in PIL format (as with jpeg_compression and pixelate)
        if isinstance(output, PILImage.Image):
            output_img = output
        else:
            output_img = PILImage.fromarray(output.astype(np.uint8))

        output_img.save(os.path.join(output_folder, image_path.split('/')[-1]))


def main():
    args = parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    root = config['data']['root']
    distortion_types = config['distortion_types']

    original_image_folder = "/".join([root, 'content'])
    image_paths = glob.glob(original_image_folder + '/*')

    # verify all the image paths
    valid_image_paths = []
    for image_path in image_paths:
        image_filename = os.path.basename(image_path)
        if is_image_file(image_filename):
            valid_image_paths.append(image_path)
    image_paths = valid_image_paths

    # distorted images are created in 5 levels of severities by default
    severities = range(1,6)

    for distortion_type in tqdm(distortion_types):
        for severity in severities:
            output_folder = os.path.join(root, distortion_type, str(severity))
            os.makedirs(output_folder, exist_ok=True)
            match distortion_type:
                case 'gaussian_noise':
                    apply_distortion(image_paths, output_folder, gaussian_noise, severity)
                case 'shot_noise':
                    apply_distortion(image_paths, output_folder, shot_noise, severity)
                case 'impulse_noise':
                    apply_distortion(image_paths, output_folder, impulse_noise, severity)
                case 'speckle_noise':
                    apply_distortion(image_paths, output_folder, speckle_noise, severity)
                case 'gaussian_blur':
                    apply_distortion(image_paths, output_folder, gaussian_blur, severity)
                case 'glass_blur':
                    apply_distortion(image_paths, output_folder, glass_blur, severity)
                case 'defocus_blur':
                    apply_distortion(image_paths, output_folder, defocus_blur, severity)
                case 'motion_blur':
                    apply_distortion(image_paths, output_folder, motion_blur, severity)
                case 'zoom_blur':
                    apply_distortion(image_paths, output_folder, zoom_blur, severity)
                case 'fog':
                    apply_distortion(image_paths, output_folder, fog, severity)
                case 'frost':
                    apply_distortion(image_paths, output_folder, frost, severity)
                case 'snow':
                    apply_distortion(image_paths, output_folder, snow, severity)
                case 'spatter':
                    apply_distortion(image_paths, output_folder, spatter, severity)
                case 'contrast':
                    apply_distortion(image_paths, output_folder, contrast, severity)
                case 'brightness':
                    apply_distortion(image_paths, output_folder, brightness, severity)
                case 'saturate':
                    apply_distortion(image_paths, output_folder, saturate, severity)
                case 'jpeg_compression':
                    apply_distortion(image_paths, output_folder, jpeg_compression, severity)
                case 'pixelate':
                    apply_distortion(image_paths, output_folder, pixelate, severity)
                case 'elastic_transform':
                    apply_distortion(image_paths, output_folder, elastic_transform, severity)


if __name__ == '__main__':
    main()