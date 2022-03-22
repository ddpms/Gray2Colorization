import torch
import random
import numpy as np


def set_randomness(random_seed: int = 2022):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU


def shift_right(image: torch.Tensor) -> torch.Tensor:
    return torch.roll(image, 1, 2)


def shift_down(image: torch.Tensor) -> torch.Tensor:
    return torch.roll(image, 1, 1)


if __name__ == '__main__':

    data = torch.rand((1, 4, 4))
    right = shift_right(data)
    down = shift_down(data)
    print(data)
    print(right)
    print(down)
