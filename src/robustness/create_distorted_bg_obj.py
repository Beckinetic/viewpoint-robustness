import argparse
import glob
import os

import yaml
from PIL import Image, ImageChops
from tqdm import tqdm

import create_distorted as cd


def parse_args():
    parser = argparse.ArgumentParser(description='Run picture distortion')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    root = config['data']['root']
    bases = config['data']['bases']
    for base in bases:
        distortion_types = config['distortion_types']
        severities = range(1, 6)

        original_image_folder = os.path.join(root, base, 'content')
        original_image_paths = glob.glob(original_image_folder + '/*.png')
        gt_image_folder = os.path.join(root, base, 'gt')
        gt_image_paths = glob.glob(gt_image_folder + '/*.png')
        gt_image_filenames = [os.path.basename(img_path) for img_path in gt_image_paths]

        bg_base_dir = os.path.join(root, base + '_background')
        obj_base_dir = os.path.join(root, base + '_object')
        os.makedirs(bg_base_dir, exist_ok=True)
        os.makedirs(obj_base_dir, exist_ok=True)

        # make background-distorted and object distorted pictures
        for distortion_type in tqdm(distortion_types):
            for severity in severities:
                distorted_image_folder = os.path.join(root, base, distortion_type, str(severity))
                distorted_image_paths = glob.glob(distorted_image_folder + '/*.png')
                for distorted_image_path in distorted_image_paths:
                    distorted_image_filename = os.path.basename(distorted_image_path)
                    distorted_image = Image.open(distorted_image_path).convert('RGBA')
                    content_image_path = os.path.join(original_image_folder, distorted_image_filename)
                    content_image = Image.open(content_image_path).convert('RGBA')
                    if distorted_image_filename in gt_image_filenames:
                        gt_image_path = os.path.join(gt_image_folder, distorted_image_filename)
                        gt_image = Image.open(gt_image_path)
                        alpha_channel = gt_image.split()[-1].convert('L')

                        # -- Background-distorted images --
                        # extract the masked picture
                        masked_image = Image.composite(content_image, Image.new("RGBA", gt_image.size), alpha_channel)

                        # paste the masked picture back to the distorted picture
                        bg_distorted_image = Image.alpha_composite(distorted_image, masked_image)

                        # save the background-distorted image
                        bg_output_folder = os.path.join(bg_base_dir, distortion_type, str(severity))
                        os.makedirs(bg_output_folder, exist_ok=True)
                        bg_output_path = os.path.join(bg_base_dir, distortion_type, str(severity), distorted_image_filename)
                        bg_distorted_image.save(bg_output_path, 'PNG')

                        # -- Object-distorted images --
                        # create the inverse mask (background mask)
                        inverted_alpha = ImageChops.invert(alpha_channel)

                        # composite the background with no distortion
                        bg_only_image = Image.composite(content_image, Image.new("RGBA", content_image.size), inverted_alpha)

                        # composite the distorted object with the original background
                        obj_distorted_image = Image.composite(distorted_image, bg_only_image, alpha_channel)

                        # save object-distorted image
                        obj_output_folder = os.path.join(obj_base_dir, distortion_type, str(severity))
                        os.makedirs(obj_output_folder, exist_ok=True)
                        obj_output_path = os.path.join(obj_base_dir, distortion_type, str(severity), distorted_image_filename)
                        obj_distorted_image.save(obj_output_path, 'PNG')


if __name__ == '__main__':
    main()