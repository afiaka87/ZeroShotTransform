# ZeroShotTransform

Transforms for captioned images.

For now, will include two useful examples I use for training DALLE-pytorch.
I'm interested in kickstarting the learning for zero shot style transfer 
by training with transforms that encourage the zero shot text abilities shown off
in the DALL-E paper/blog post.

# Two Panel 
 `
from zero_shot_transform.transforms import zero_shot_transform
# ...
image_file = pathlib.Path("path/to/valid/image.(png|jpg)")
pil_image = PIL.Image.open(image_file) # an image you want to transform.
images = []
chosen_transform = random.choice(possible_transforms)
zero_shot_transform(image, caption="", p=0.5, transform_to_apply=possible_transform)
