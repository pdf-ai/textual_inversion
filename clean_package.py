import os
import shutil
import argparse


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
        "-input",
        "--input_path",
        type=str,
        const=True,
        default='../datasets/ukiyoe',
        nargs="?",
        help="Path for loading input images",
    )
    
    parser.add_argument(
        "-out",
        "--output_path",
        type=str,
        const=True,
        default='ukiyoe',
        nargs="?",
        help="Path for loading input images",
    )
    return parser


def pack_up(sourcepath, name):
    target_path = os.path.join('./datasets', name)
    for cls in os.listdir(sourcepath):
        source_cls = os.path.join(sourcepath, cls)
        target_cls = os.path.join(target_path, cls)
        if not os.path.exists(target_cls):
            os.mkdir(target_cls)
        for img in os.listdir(source_cls):
            if img[-3:] == 'jpg' or img[-3:] == 'png' or img[-4:] == 'jpeg':
                source_img = os.path.join(source_cls, img)
                target_img = os.path.join(target_cls, img)
                shutil.copy(source_img, target_img) 

if __name__ == "__main__":
    #pack_up('../datasets/ukiyoe', 'ukiyoe')
    parser = get_parser()
    args = parser.parse_args()
    pack_up(args.input_path, args.output_path)