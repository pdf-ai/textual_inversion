import os
from PIL import Image as Image
import PIL

input_size = 512

# bs = 4
# source_dir = 'source/attacked_Style1_Block/8_40_1.0'
# target_dir ='source/attacked_Style1_Block/8_40_1.0_merge'
bs = 5
source_dir = 'source/attacked_ukiyoe_Block/8_40_1.0'
target_dir ='source/attacked_ukiyoe_Block/8_40_1.0_merge'
cropsize = [(0, 0, input_size//2, input_size//2), (input_size//2, 0, input_size, input_size//2), (0, input_size//2, input_size//2, input_size), (input_size//2, input_size//2, input_size, input_size)]

source_dir_list = os.listdir(source_dir)

for z in range(bs):
    target = Image.new('RGB', (input_size, input_size))
    for q in range(len(source_dir_list)):
        image_path = os.path.join(source_dir, source_dir_list[q]) + '/' + str(z) + '.png'
        img_read = Image.open(image_path)
        target.paste(img_read, cropsize[q])
    target_path = target_dir + '/' + str(z) + '.png'
    target.save(target_path)