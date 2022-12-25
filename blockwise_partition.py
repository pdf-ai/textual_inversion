import os
from PIL import Image as Image
import PIL

input_size = 512


#source_dir = 'datasets/Fixed_Size_Style'
#target_dir = 'datasets/Style1_Block'
source_dir = '../datasets/ukiyoe/0'
target_dir = 'datasets/ukiyoe_Block'
cropsize = [(0, 0, input_size/2, input_size/2), (input_size/2, 0, input_size, input_size/2), (0, input_size/2, input_size/2, input_size), (input_size/2, input_size/2, input_size, input_size)]

for img in os.listdir(source_dir):
    if img[-1] == 's':
        continue
    if img[-1] == 'z':
        continue
    image_path = os.path.join(source_dir, img)
    for q in range(4):
        target_dir_ = target_dir + '/' + str(q)
        target_path = os.path.join(target_dir_, img)
        img_read = Image.open(image_path).resize((input_size, input_size), resample=PIL.Image.BICUBIC).crop(cropsize[q])
        img_read.save(target_path)