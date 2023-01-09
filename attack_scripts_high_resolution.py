import numpy
import os

from PIL import Image as Image
import PIL
import argparse, tqdm
import io
import contextlib
import sys
import shutil
os.environ['MKL_THREADING_LAYER'] = 'GNU' 


def del_file(filepath, target_suffix='ipynb_checkpoints'):
    if target_suffix == 'all':
        shutil.rmtree(filepath) 
        return
    files = os.listdir(filepath)
    for file in files:
        if '.' in file:
            suffix = file.split('.')[-1]
            if suffix == target_suffix:
                shutil.rmtree(os.path.join(filepath, file))


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "-eps",
        "--epsilon",
        type=int,
        const=True,
        default=8,
        nargs="?",
        help="epsilon for the attacks",
    )

    parser.add_argument(
        "-ins",
        "--input_size",
        type=int,
        const=True,
        default=512,
        nargs="?",
        help="Image Resolution",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-st",
        "--steps",
        type=int,
        const=True,
        default=40,
        nargs="?",
        help="steps for the attacks",
    )

    parser.add_argument(
        "-al",
        "--alpha",
        type=float,
        const=True,
        default=1.0,
        nargs="?",
        help="epsilon for one step",
    )

    parser.add_argument(
        "-out",
        "--output_path",
        type=str,
        const=True,
        default="ukiyoe",
        nargs="?",
        help="Path for saving attacked and generated images",
    )

    parser.add_argument(
        "-input",
        "--input_path",
        type=str,
        const=True,
        default='../datasets/ukiyoe',
        nargs="?",
        help="Path for loading input images",
    )

    parser.add_argument(
        "-t",
        "--type",
        type=str,
        const=True,
        default='Style',
        nargs="?",
        help="Path for loading input images",
    )
    
    parser.add_argument("--visualize_debug", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--no_attack", action="store_true")
    parser.add_argument("--no_del", action="store_true")
    parser.add_argument(
        "-r",
        "--resume",
        type=int,
        default=0,
        help="whether or not to resume",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )

    return parser


# for img in os.listdir(source_dir):
#     if img[-1] == 's':
#         continue
#     if img[-1] == 'z':
#         continue
#     image_path = os.path.join(source_dir, img)
#     target_dir_ = target_dir
#     target_path = os.path.join(target_dir_, img)
#     img_read = Image.open(image_path).resize((input_size, input_size), resample=PIL.Image.BICUBIC)
#     img_read.save(target_path)


if __name__ == '__main__':
    parser = get_parser()
    epsilon_list = [2, 4, 16, 32]
    step_list = [10, 1, 40, 100, 500, 1000]
    args = parser.parse_args()
    attack_parameters = str(args.epsilon) + '_' + str(args.steps) + '_' + str(args.alpha) 
    for attack_option in range(8):
        if attack_option < 5:
            continue
        if attack_option < 4:
            steps = 40
            epsilon = epsilon_list[attack_option]
        else:
            steps = step_list[attack_option-4]
            epsilon = 8
        print('###Testing for ablation Study')
        print('##Current eps is', str(epsilon))
        print('##Current step is', str(steps))
        
        attack_parameters = str(epsilon) + '_' + str(steps) + '_' + str(args.alpha) 
        output_path = 'attacked_' + args.output_path + '/' + attack_parameters
        target_removing_path = 'source/' + output_path
        source_command = 'python attack_ldm_fix_path.py -t --actual_resume models/ldm/text2img-large/model.ckpt ' + ' -ins  ' + str(args.input_size) +   '  --gpus '+str(args.gpu) + ', '
        dir_info = '--data_root ' + args.input_path + ' -out ' + args.output_path + ' -bs ' + str(args.batch_size)
        if args.type == 'style':
            config_name = '--base configs/latent-diffusion/Big_Resolution/txt2img-1p4B-finetune_style_with_grad.yaml '
            training_config_name = '--base configs/latent-diffusion/Embedding_in_Framework/txt2img-1p4B-finetune_style.yaml '
        else:
            config_name = '--base configs/latent-diffusion/Big_Resolution/txt2img-1p4B-finetune_with_grad.yaml --object '
            training_config_name = '--base configs/latent-diffusion/Embedding_in_Framework/txt2img-1p4B-finetune.yaml '

        source_command += config_name
        attack_info = ' --epsilon ' + str(epsilon) + ' '+ ' --steps ' + str(steps) + ' --alpha ' + str(args.alpha) + ' '
        attack_command = source_command + dir_info + attack_info

        print('# Phase1 Removing .ipycheckpoints; Cleaning Target Datasets')

        # os.mkdir(target_removing_path)
        print('Phase1 Fin')
        if not args.no_attack:
            print('Phase2 Performing Attacks')
            print(attack_command)
            os.system(attack_command)
        print('Phase2 Fin')
