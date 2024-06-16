import os
import warnings

from PIL import Image
from tqdm import tqdm


def picture_pasting(root, transparent_path, content_path, output):
    """
    Pasting style-transferred object fraction to the original content picture
    :param root: Root (style-transferred pictures)
    :param transparent_path: Path to the transparent picture
    :param content_path: Path to the content picture
    :param output: Path to the output picture
    :return:
    """
    if not os.path.exists(transparent_path):
        warnings.warn('The transparent picture does not exist')
        return 0
    elif not os.path.exists(content_path):
        warnings.warn('The content picture does not exist')
        return 0

    if not os.path.exists(output):
        os.makedirs(output)

    tf_names = [pic for pic in os.listdir(root) if pic.endswith('.jpg')]
    tp_names = [pic for pic in os.listdir(transparent_path) if pic.endswith('.png')]
    ct_names = [pic for pic in os.listdir(content_path) if pic.endswith('.png')]

    for tf_name in tqdm(tf_names):
        # open the transferred pic
        tf_path = os.path.join(root, tf_name)
        tf_pic = Image.open(tf_path).convert('RGB')

        # find the corresponding transparent and content picture
        original_name = "_".join(tf_name.split('_')[:-2])
        content_name = original_name + '.png'
        transparent_name = content_name

        # obtain the alpha channel
        if transparent_name in tp_names:
            tp_path = os.path.join(transparent_path, transparent_name)
            tp_pic = Image.open(tp_path)
            alpha_channel = tp_pic.split()[-1].convert('L')
        else:
            warnings.warn(f"{transparent_name} is not found in the transparent pictures", UserWarning)
            continue

        # obtain the content image for being pasted on
        if content_name in ct_names:
            ct_path = os.path.join(content_path, content_name)
            ct_pic = Image.open(ct_path).convert('RGBA')
        else:
            warnings.warn(f"{content_name} is not found in the content pictures", UserWarning)
            continue

        # extract the masked transferred picture
        masked_image = Image.composite(tf_pic, Image.new("RGBA", tf_pic.size), alpha_channel)

        # paste the masked picture back to the content picture
        combined_image = Image.alpha_composite(ct_pic, masked_image)

        # save the pasted picture to the output folder with the same name of the tf_name
        output_path = os.path.join(output, tf_name.replace('.jpg', '.png'))
        combined_image.save(output_path, 'PNG')
