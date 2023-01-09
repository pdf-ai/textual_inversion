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
source_pa = '../datasets/wikiart-10/wikiart-summary'
target_pa = 'datasets/wikiart'
    
# for class_name in os.listdir(source_pa):
#     source_dir = os.path.join(source_pa, class_name)
#     target_dir =  os.path.join(target_pa, class_name)
#     os.mkdir(target_dir)
#     for img in os.listdir(source_dir):
#         if img[-1] == 's':
#             continue
#         image_path = os.path.join(source_dir, img)
#         target_path = os.path.join(target_dir, img)
#         img_read = Image.open(image_path).resize((input_size, input_size), resample=PIL.Image.BICUBIC)
#         img_read.save(target_path)