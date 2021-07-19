from pathlib import Path
from random import randint, choice, randrange, uniform
from typing import Tuple
import augly.image as imaugs
import augly.text as txtaugs

import PIL

from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms as T
import torchvision
from torchvision.transforms.functional import center_crop, scale

# TODO broken
def overlay_font_of_caption():
    pil_image = PIL.Image.open(image_file)
    aug_opacity = uniform(0.0, 1.0)
    random_substr = get_random_str(description, 16)
    aug_text = textaugs.ReplaceSimilarUnicodeChars(random_substr, p=0.6),
    aug_text = textaugs.replace_fun_fonts(aug_text, vary_fonts=True, granularity="word"),
    aug_image = imaugs.meme_format(pil_image, text=aug_text, caption_height=randint(64,128), opacity=aug_opacity)
    image_tensor = self.image_transform(aug_image)
    return image_tensor

def get_random_str(main_str, substr_len):
    idx = randrange(0, len(main_str) - substr_len + 1)    # Randomly select an "idx" such that "idx + substr_len <= len(main_str)".
    return main_str[idx : (idx+substr_len)]

def four_line_caption(text, row_length=12, words_per_row=4): 
    """
    Generate small two-line captions from a larger string.

    description: Full length caption.
    row_length: Max characters per row. Higher than 16 will shrink text.
    words_per_row: Number of words in a row.
    """
    split_by_spaces = text.split(" ")
    if len(split_by_spaces) > words_per_row:
        row_one = " ".join(split_by_spaces[:words_per_row])
        row_two = " ".join(split_by_spaces[words_per_row:words_per_row*2])
        row_three = " ".join(split_by_spaces[words_per_row*2:words_per_row*3])
        row_four = " ".join(split_by_spaces[words_per_row*3:words_per_row*4])
        return row_one + "\n" + row_two + "\n" + row_three + "\n" + row_four
    return text[:row_length]

def two_panel_style_transfer(image, image_size=256, resize_ratio=0.8, img_transform=None):
    """ image: PIL.Image to be cropped.  image_size: Size of the cropped image.  resize_ratio: Resize ratio of the cropped image.  """
    half_height_center_crop = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.RandomResizedCrop((image_size // 2, image_size), scale=(resize_ratio, 1.0)),
    ])
    top_panel = half_height_center_crop(image)
    bottom_panel = img_transform(image)
    return torch.stack([top_panel, bottom_panel], dim=0)

transform_lookup = {
    "solarized": torchvision.transforms.RandomSolarize(threshold=16, p=1),
    "sharpen": torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2, p=1),
    "blur": torchvision.transforms.RandomAdjustSharpness(sharpness_factor=0, p=1),
    "color_jitter": torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    "grayscale": torchvision.transforms.RandomGrayscale(p=1),
    "horizontal_flip": torchvision.transforms.RandomHorizontalFlip(p=1),
    "vertical_flip": torchvision.transforms.RandomVerticalFlip(p=1),
}
transform_keys = ["solarized", "sharpen", "blur", "color_jitter", "grayscale", "horizontal_flip", "vertical_flip"]

def zero_shot_transform(image, caption="", p=0.5, transform_to_apply="solarized"):
    """
    Output PIL.Image of a two-panel style transfer image with an english word describing the transform included in the caption.
    See OpenAI DALL-E blog post for more details:
    The top half of the image is the original image, and the bottom half is a stylized image.
    """
    assert len(caption) > 0, "Zero-shot transform must have a caption."
    image_transform = torchvision.transforms.RandomApply(transform_lookup[transform_to_apply], p=p)
    friendly_transform = transform_to_apply.replace("_", " ")
    pil_image = two_panel_style_transfer(image, img_transform=image_transform, resize_ratio=0.6)
    return pil_image, f"Two panel image of the exact same picture." + \
            "On the top {caption} and on the bottom the same image but with {friendly_transform} applied." + \
            "The original image is on the top and the {friendly_transform} image on the bottom. The caption is {caption}."
