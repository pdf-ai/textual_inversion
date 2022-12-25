import os
from PIL import Image as Image
import PIL

input_size = 512


source_dir = 'datasets/Style1/'
target_dir = 'datasets/Fixed_Size_Style'

for img in os.listdir(source_dir):
    if img[-1] == 's':
        continue
    image_path = os.path.join(source_dir, img)
    target_path = os.path.join(target_dir, img)
    img_read = Image.open(image_path).resize((input_size, input_size), resample=PIL.Image.BICUBIC)
    img_read.save(target_path)