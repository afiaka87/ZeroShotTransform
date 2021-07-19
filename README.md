# ZeroShotTransform

## Transforms for captioned images.

Work In Progress.

```python
from pathlib import Path
from zero_shot_transform.transforms import zero_shot_transform
from PIL import Image

image = Image.open(Path("path/to/valid/image.(png|jpg)"))
caption = "an image of a cat"
image, caption = zero_shot_transform(
   image,
   caption=caption,
   transform_to_apply="grayscale", # See above for other options.
   p=1.0,
)
```

<img width="208" alt="Screen Shot 2021-07-19 at 8 25 12 AM" src="https://user-images.githubusercontent.com/3994972/126166845-11a7ce50-c9eb-44aa-81da-0b451cc1363b.png">

```
Two panel image of the exact same picture.
On the top an image of a cat and on the bottom the same image but with grayscale applied.
The original image is on the top and the grayscale image on the bottom.
The caption is "an image of a cat.
```

For now, will include two useful examples I use for training DALLE-pytorch.
I'm interested in kickstarting the learning for zero shot style transfer 
by training with transforms that encourage the zero shot style transfer 
text-image abilities shown off in the DALL-E paper/blog post.

## Two Panel Style Transfer

Supported Transform by Name
- `solarized`
- `sharpen`
- `blur`
- `color_jitter`
- `grayscale`
- `horizontal_flip`
- `vertical_flip`

Running 

