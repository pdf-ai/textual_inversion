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
    args = parser.parse_args()
    attack_parameters = str(args.epsilon) + '_' + str(args.steps) + '_' + str(args.alpha) 
    output_path = 'attacked_' + args.output_path + '/' + attack_parameters
    target_removing_path = 'source/' + output_path
    source_command = 'python attack_ldm_fix_path.py -t --actual_resume models/ldm/text2img-large/model.ckpt  --gpus '+str(args.gpu) + ', '
    dir_info = '--data_root ' + args.input_path + ' -out ' + args.output_path + ' -bs ' + str(args.batch_size)
    if args.type == 'style':
        config_name = '--base configs/latent-diffusion/txt2img-1p4B-finetune_style_with_grad.yaml '
        training_config_name = '--base configs/latent-diffusion/Embedding_in_Framework/txt2img-1p4B-finetune_style.yaml '
    else:
        config_name = '--base configs/latent-diffusion/txt2img-1p4B-finetune_with_grad.yaml --object '
        training_config_name = '--base configs/latent-diffusion/Embedding_in_Framework/txt2img-1p4B-finetune.yaml '
    
    source_command += config_name
    attack_command = source_command + dir_info
    
    print('# Phase1 Removing .ipycheckpoints; Cleaning Target Datasets')
    if not args.no_attack:
        del_file(args.input_path)
        for k in os.listdir(args.input_path):
            del_file(os.path.join(args.input_path, k))
        if os.path.exists(target_removing_path):
            del_file(target_removing_path, 'all')
        os.mkdir('source/' + 'attacked_' + str(args.output_path))
    print('Phase1 Fin')
    if not args.no_attack:
        print('Phase2 Performing Attacks')
        print(attack_command)
        os.system(attack_command)
    print('Phase2 Fin')
    source_log_name = 'framework_logs/' + args.output_path
    source_log_target_path = os.path.join(source_log_name, attack_parameters)
    source_log_name_cln = 'framework_logs/' + args.output_path + '_clean'
    source_log_target_path_cln = os.path.join(source_log_name_cln, attack_parameters)
    
    if not args.no_del:
        if not os.path.exists(source_log_name):
            os.mkdir(source_log_name)
        if os.path.exists(source_log_target_path):
            del_file(source_log_target_path, 'all')
        os.mkdir(source_log_target_path)
        if not os.path.exists(source_log_name_cln):
            os.mkdir(source_log_name_cln)
        if os.path.exists(source_log_target_path_cln):
            del_file(source_log_target_path_cln, 'all')
        os.mkdir(source_log_target_path_cln)

    output_path = 'outputs/txt2img-samples-textual-inversion'
    output_path = os.path.join(output_path, args.output_path)
    output_log_target_path = os.path.join(output_path, attack_parameters)
    output_path_cln = 'outputs/txt2img-samples-textual-inversion'
    output_path_cln = os.path.join(output_path_cln, args.output_path)+ '_clean'
    output_log_target_path_cln = os.path.join(output_path_cln, attack_parameters)

    if not args.no_del:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if os.path.exists(output_log_target_path):
            del_file(output_log_target_path, 'all')
        os.mkdir(output_log_target_path)

        if not os.path.exists(output_path_cln):
            os.mkdir(output_path_cln)
        if os.path.exists(output_log_target_path_cln):
            del_file(output_log_target_path_cln, 'all')
        os.mkdir(output_log_target_path_cln)
    class_num=0
    dir_list = os.listdir(target_removing_path)
    dir_list.sort()
    for count_dir in dir_list:
        class_num += 1
        if args.resume > class_num:
            print('skipping ',count_dir)
            continue
        print('Phase3 Training Adv Embeddings')
        log_target_class = os.path.join(source_log_target_path, count_dir)
        input_data_class = os.path.join(target_removing_path,count_dir)
        output_class =  os.path.join(output_log_target_path, count_dir)
        clean_input_path = os.path.join(os.path.join('./datasets', args.output_path), count_dir)
        
        log_target_class_cln = os.path.join(source_log_target_path_cln, count_dir)

        output_class_cln =  os.path.join(output_log_target_path_cln, count_dir)
        if not args.no_del:
            os.mkdir(output_class)
            os.mkdir(log_target_class)
        generating_command = 'python generating_images.py -t --gpus '+str(args.gpu)+', ' +  '--actual_resume models/ldm/text2img-large/model.ckpt ' + training_config_name + '--data_root ' + input_data_class + ' --logdir ' + str(log_target_class)
        print(generating_command)
        os.system(generating_command)
        print('Phase4 Generating Adv Embeddings')
        
        if args.type == 'style':
            embedding_output_command = 'python scripts/txt2img.py  --ddim_eta 0.0 --n_samples 2 --n_iter 5 --scale 10.0 --ddim_steps 50  --embedding_path ' + log_target_class + '/checkpoints/embeddings_gs-4999.pt --ckpt_path models/ldm/text2img-large/model.ckpt --prompt "A photo in the style of *" --outdir '+output_class
        else:
            embedding_output_command = 'python scripts/txt2img.py  --ddim_eta 0.0 --n_samples 2 --n_iter 5 --scale 10.0 --ddim_steps 50  --embedding_path ' + log_target_class + '/checkpoints/embeddings_gs-4999.pt --ckpt_path models/ldm/text2img-large/model.ckpt --prompt "A photo of *" --outdir '+output_class            
        print(embedding_output_command)
        os.system(embedding_output_command)
        if not args.no_eval:
            print('Phase5 Evaluating Adv Embeddings')
            eval_path = os.path.join(output_class, 'samples')
            stat_path = os.path.join(os.path.join(args.input_path, count_dir), 'statistics.npz')
            del_file(eval_path)
            print(evaluation(eval_path, stat_path))
        print('Phase6 Training Clean Embeddings')
        os.mkdir(output_class_cln)
        os.mkdir(log_target_class_cln)
        generating_cln_command = 'python generating_images.py -t --gpus '+str(args.gpu)+', ' +  ' --actual_resume models/ldm/text2img-large/model.ckpt ' + training_config_name + '--data_root ' + clean_input_path + ' --logdir ' + str(log_target_class_cln)
        print(generating_cln_command)
        os.system(generating_cln_command)
        print('Phase7 Generating Clean Embeddings')
        if args.type == 'style':
            embedding_output_command_cln = 'python scripts/txt2img.py --ddim_eta 0.0 --n_samples 2 --n_iter 5 --scale 10.0 --ddim_steps 50  --embedding_path ' + log_target_class_cln + '/checkpoints/embeddings_gs-4999.pt --ckpt_path models/ldm/text2img-large/model.ckpt --prompt "A photo in the style of *" --outdir '+output_class_cln
        else:
            embedding_output_command_cln = 'python scripts/txt2img.py --ddim_eta 0.0 --n_samples 2 --n_iter 5 --scale 10.0 --ddim_steps 50  --embedding_path ' + log_target_class_cln + '/checkpoints/embeddings_gs-4999.pt --ckpt_path models/ldm/text2img-large/model.ckpt --prompt "A photo of *" --outdir '+output_class_cln
        print(embedding_output_command_cln)
        os.system(embedding_output_command_cln)
        if not args.no_eval:
            print('Phase8 Evaluating Clean Embeddings')
            eval_path = os.path.join(output_class_cln, 'samples')
            stat_path = os.path.join(clean_input_path, 'statistics.npz')
            del_file(eval_path)
            print(evaluation(eval_path, stat_path))
        
    ## TODO 实现函数生成过程静音，测试生成结果，实现optional去除文件