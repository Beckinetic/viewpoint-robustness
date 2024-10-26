import glob
import logging

import yaml
from tqdm import tqdm

from src.robustness.distortion import *

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('config', type=str, help='Path to the configuration file')
parser.add_argument('--data-dir', type=str, default='data/', help='Directory to fetch the data')
parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
parser.add_argument('--model-dir', type=str, default='models/', help='Directory to save models')
args = parser.parse_args()

data_dir = args.data_dir
log_dir = args.log_dir
model_dir = args.model_dir

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)


def create_distorted():
    roots = config['data']['root']
    distortion_types = config['distortion_types']
    for root in roots:
        logging.info(f'Creating distorted images in {root}')
        content_folder = os.path.join(data_dir, root, 'content')
        content_image_paths = glob.glob(content_folder + '/*')

        # verify all the image paths
        valid_image_paths = []
        for image_path in content_image_paths:
            image_filename = os.path.basename(image_path)
            if is_image_file(image_filename):
                valid_image_paths.append(image_path)
        content_image_paths = valid_image_paths

        # distorted images are created in 5 levels of severities by default
        severities = range(1, 6)

        for distortion_type in tqdm(distortion_types):
            for severity in severities:
                output_folder = os.path.join(data_dir, root, distortion_type, str(severity))
                os.makedirs(output_folder, exist_ok=True)
                match distortion_type:
                    case 'gaussian_noise':
                        apply_distortion(content_image_paths, output_folder, gaussian_noise, severity)
                    case 'shot_noise':
                        apply_distortion(content_image_paths, output_folder, shot_noise, severity)
                    case 'impulse_noise':
                        apply_distortion(content_image_paths, output_folder, impulse_noise, severity)
                    case 'speckle_noise':
                        apply_distortion(content_image_paths, output_folder, speckle_noise, severity)
                    case 'gaussian_blur':
                        apply_distortion(content_image_paths, output_folder, gaussian_blur, severity)
                    case 'glass_blur':
                        apply_distortion(content_image_paths, output_folder, glass_blur, severity)
                    case 'defocus_blur':
                        apply_distortion(content_image_paths, output_folder, defocus_blur, severity)
                    case 'motion_blur':
                        apply_distortion(content_image_paths, output_folder, motion_blur, severity)
                    case 'zoom_blur':
                        apply_distortion(content_image_paths, output_folder, zoom_blur, severity)
                    case 'fog':
                        apply_distortion(content_image_paths, output_folder, fog, severity)
                    case 'frost':
                        apply_distortion(content_image_paths, output_folder, frost, severity)
                    case 'snow':
                        apply_distortion(content_image_paths, output_folder, snow, severity)
                    case 'spatter':
                        apply_distortion(content_image_paths, output_folder, spatter, severity)
                    case 'contrast':
                        apply_distortion(content_image_paths, output_folder, contrast, severity)
                    case 'brightness':
                        apply_distortion(content_image_paths, output_folder, brightness, severity)
                    case 'saturate':
                        apply_distortion(content_image_paths, output_folder, saturate, severity)
                    case 'jpeg_compression':
                        apply_distortion(content_image_paths, output_folder, jpeg_compression, severity)
                    case 'pixelate':
                        apply_distortion(content_image_paths, output_folder, pixelate, severity)
                    case 'elastic_transform':
                        apply_distortion(content_image_paths, output_folder, elastic_transform, severity)


if __name__ == '__main__':
    create_distorted()
