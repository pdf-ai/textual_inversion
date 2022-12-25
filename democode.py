import argparse
import contextlib
import sys


class DummyFile(object):
    def write(self, x): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

    
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

    parser.add_argument("--quiet", action="store_true")
    return parser


def main(args):
    print(args.epsilon)
    print('Testing SubFile')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.quiet:
        with nostdout():
            main(args)
    else:
        main(args)
