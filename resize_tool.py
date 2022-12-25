import os
from PIL import Image as Image
import PIL

input_size = 512


#source_dir = 'datasets/Fixed_Size_Style'
#target_dir = 'datasets/Style1_Block'
source_dir = '../datasets/ukiyoe/0'
target_dir = 'datasets/ukiyoe_Resize'

for img in os.listdir(source_dir):
    if img[-1] == 's':
        continue
    if img[-1] == 'z':
        continue
    image_path = os.path.join(source_dir, img)
    target_dir_ = target_dir
    target_path = os.path.join(target_dir_, img)
    img_read = Image.open(image_path).resize((input_size, input_size), resample=PIL.Image.BICUBIC)
    img_read.save(target_path)