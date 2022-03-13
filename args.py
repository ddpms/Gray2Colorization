import argparse


def get_args():
    p = argparse.ArgumentParser(
        description="Deep Line Art Video Colorization with a Few References")

    p.add_argument("--low_res", default=64, type=int,
                   help="low resolution image size")
    p.add_argument("--high_res", default=256, type=int,
                   help="high resolution image size")
    p.add_argument("--train_path", default="dataset/train/", type=str,
                   help="train image path")
    p.add_argument("--valid_path", default="dataset/valid/", type=str,
                   help="valid image path")
    p.add_argument("--batch_size", default=1, type=int,
                   help="batch size")
    p.add_argument("--num_workers", default=1, type=int,
                   help="num workers")
    p.add_argument("--train_mode", default=True, type=bool,
                   help="num workers")

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
