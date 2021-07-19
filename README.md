# ZeroShotTransform

Work In Progress.

Transforms for captioned images.

For now, will include two useful examples I use for training DALLE-pytorch.
I'm interested in kickstarting the learning for zero shot style transfer 
by training with transforms that encourage the zero shot style transfer 
text-image abilities shown off in the DALL-E paper/blog post.

## Two Panel Style Transfer
 ```python
from zero_shot_transform.transforms import zero_shot_transform

# Load an image with PIL
image_file = pathlib.Path("path/to/valid/image.(png|jpg)")
pil_image = PIL.Image.open(image_file)
```

Supported Transform by Name
  - `solarized`
  - `sharpen`
  - `blur`
  - `color_jitter`
  - `grayscale`
  - horizontal_flip
  - vertical_flip

Running 
```python
caption = "Some lengthy caption from COCO or another image-text dataset."
image, caption = zero_shot_transform(
   image,
   caption=caption,
   transform_to_apply="grayscale", # See above for other options.
   p=0.5,
)
```

`image`

<img width="208" alt="Screen Shot 2021-07-19 at 8 25 12 AM" src="https://user-images.githubusercontent.com/3994972/126166845-11a7ce50-c9eb-44aa-81da-0b451cc1363b.png">


`caption` (format string used for caption transform)
```python
caption = f"Two panel image of the exact same picture." + \
  "On the top {caption} and on the bottom the same image but with {transform_name} applied." + \
  "The original image is on the top and the {transform_name} image on the bottom. The caption is {caption}."
  .replace("_", " ")
```
