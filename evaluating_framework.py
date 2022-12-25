import numpy
import os

from PIL import Image as Image
import PIL
import argparse, tqdm
import io
import contextlib
import sys
import shutil
from compute_is_fid import evaluation
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
        default=256,
        nargs="?",
        help="Image Resolution",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=5,
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
    args = parser.parse_args()
    attack_parameters = str(args.epsilon) + '_' + str(args.steps) + '_' + str(args.alpha) 
    output_path = 'attacked_' + args.output_path + '/' + attack_parameters
    target_removing_path = 'source/' + output_path
    source_command = 'python attack_ldm_fix_path.py -t --actual_resume models/ldm/text2img-large/model.ckpt --gpus 0, '
    dir_info = '--data_root ' + args.input_path + ' -out ' + args.output_path + ' -bs ' + str(args.batch_size)
    if type == 'style':
        config_name = '--base configs/latent-diffusion/txt2img-1p4B-finetune_style_with_grad.yaml '
        training_config_name = '--base configs/latent-diffusion/Embedding_in_Framework/txt2img-1p4B-finetune_style.yaml '
    else:
        config_name = '--base configs/latent-diffusion/txt2img-1p4B-finetune_with_grad.yaml --object '
        training_config_name = '--base configs/latent-diffusion/Embedding_in_Framework/txt2img-1p4B-finetune.yaml '
    
    source_command += config_name
    attack_command = source_command + dir_info

    source_log_name = 'framework_logs/' + args.output_path
    source_log_target_path = os.path.join(source_log_name, attack_parameters)
    source_log_name_cln = 'framework_logs/' + args.output_path + '_clean'
    source_log_target_path_cln = os.path.join(source_log_name_cln, attack_parameters)

    output_path = 'outputs/txt2img-samples-textual-inversion'
    output_path = os.path.join(output_path, args.output_path)
    output_log_target_path = os.path.join(output_path, attack_parameters)
    output_path_cln = 'outputs/txt2img-samples-textual-inversion'
    output_path_cln = os.path.join(output_path_cln, args.output_path)+ '_clean'
    output_log_target_path_cln = os.path.join(output_path_cln, attack_parameters)
    count = 0

    for count_dir in os.listdir(target_removing_path):

        log_target_class = os.path.join(source_log_target_path, count_dir)
        input_data_class = os.path.join(target_removing_path,count_dir)
        output_class =  os.path.join(output_log_target_path, count_dir)
        clean_input_path = os.path.join(os.path.join('../datasets', args.output_path), count_dir)
        log_target_class_cln = os.path.join(source_log_target_path_cln, count_dir)

        output_class_cln =  os.path.join(output_log_target_path_cln, count_dir)

        eval_path = os.path.join(output_class, 'samples')
        stat_path = os.path.join(os.path.join(args.input_path, count_dir), 'statistics.npz')
        print(eval_path)
        del_file(eval_path)
        print(evaluation(eval_path, stat_path))
        eval_path = os.path.join(output_class_cln, 'samples')
        del_file(eval_path)
        print(eval_path)
        print(evaluation(eval_path, stat_path))

