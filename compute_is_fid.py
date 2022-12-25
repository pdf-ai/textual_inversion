from pytorch_gan_metrics import get_inception_score, get_fid, get_fid_from_directory
import os
import PIL
import numpy as np
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
trans = transforms.Compose([transforms.ToTensor()])

# is: the minus entropy score with inception v3 model
#     if is is large, then the image seems to be like 'something' and the 
#     generating performance is good
# fid: the inception v3 distance between generated and benchmark images
#     if fid is small, then the generated image seems to be like the
#     benchmark images and the generating performance is good

# for is, you only need to load the images
# for fid, you first need to compute some statistics of the benchmark data
#     run 
#   'python -m pytorch_gan_metrics.calc_fid_stats \
#    --path path/to/images \
#    --stats path/to/statistics.npz'
#     to compute and output the statistics

# images = ... # [N, 3, H, W] normalized to [0, 1]
# IS, IS_std = get_inception_score(images)        # Inception Score
# FID = get_fid(images, 'path/to/statistics.npz') # Frechet Inception Distance


def evaluation(image_path, statistics_path):
    input_image_list = os.listdir(image_path)
    input_image_tensor = torch.zeros([len(input_image_list), 3, 256, 256])
    for k in range(len(input_image_list)):
        input_path = os.path.join(image_path, input_image_list[k])
        img_read = Image.open(input_path).resize((256, 256), resample=PIL.Image.BILINEAR)
        img = (np.array(img_read).astype(np.uint8)/ 255.0).astype(np.float32)
        input_image_tensor[k] = trans(img)
    IS, IS_std = get_inception_score(input_image_tensor, splits=1)        # Inception Score
    FID = get_fid(input_image_tensor, statistics_path) # Frechet Inception Distance
    return IS, IS_std, FID


def evaluation_(image_path, statistics_path):
    input_image_list = os.listdir(image_path)
    input_image_tensor = torch.zeros([10, 3, 256, 256])
    for p in range(2):
        for k in range(len(input_image_list)):
            input_path = os.path.join(image_path, input_image_list[k])
            img_read = Image.open(input_path).resize((256, 256), resample=PIL.Image.BILINEAR)
            img = (np.array(img_read).astype(np.uint8)/ 255.0).astype(np.float32)
            input_image_tensor[p*5+k] = trans(img)
    IS, IS_std = get_inception_score(input_image_tensor, splits=1)          # Inception Score
    FID = get_fid(input_image_tensor, statistics_path) # Frechet Inception Distance
    return IS, IS_std, FID


def compute_metrics(image_path):
    input_image_list = os.listdir(image_path)
    input_image_tensor = torch.zeros([len(input_image_list), 3, 256, 256])
    for k in range(len(input_image_list)):
        input_path = os.path.join(image_path, input_image_list[k])
        img_read = Image.open(input_path).resize((256, 256), resample=PIL.Image.BILINEAR)
        img = (np.array(img_read).astype(np.uint8)/ 255.0).astype(np.float32)
        input_image_tensor[k] = trans(img)
    print(input_image_tensor)
    print(input_image_tensor.shape)
    IS, IS_std = get_inception_score(input_image_tensor)        # Inception Score
    print(IS, IS_std)
    FID = get_fid(input_image_tensor, 'path/to/statistics.npz') # Frechet Inception Distance
    return IS, IS_std


if __name__ == "__main__":
    # print(evaluation_('datasets/ukiyoe/0', '../datasets/ukiyoe/0/statistics.npz') )
    # print(evaluation_('datasets/cat/100', '../datasets/lsun_dataset/cat/100/statistics.npz') )
    print(evaluation('outputs/txt2img-samples/no_pseudo_word_cat', '../datasets/lsun_dataset/cat/100/statistics.npz') )
    print(evaluation('../datasets/demo_generated', '../datasets/demo/statistics.npz') )
    print(evaluation('../datasets/demo_source', '../datasets/demo/statistics.npz') )