import cv2
import albumentations as A
from glob import glob
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


class FlickerDataset(Dataset):

    def __init__(self, args, img_path, transform=None):
        self.args = args
        self.transform = transform
        self.img_files = glob(img_path + '*.jpg')

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        img_file = self.img_files[index]
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.args.low_res, self.args.low_res))
        high_res = cv2.resize(img, (self.args.high_res, self.args.high_res))
        low_res = cv2.resize(img, (self.args.low_res, self.args.low_res))

        if self.transform:
            # gray = self.transform(image=gray)["image"]
            high_res = self.transform(image=high_res)["image"]
            low_res = self.transform(image=low_res)["image"]

        data = dict(
            gray=gray,
            low_resolution=low_res,
            high_resolution=high_res,
        )

        return data


def get_transform(args):

    train_transform = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])

    valid_transform = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])

    transform = dict(
        train=train_transform,
        valid=valid_transform
    )

    return transform


def get_loaders(args):

    transform = get_transform(args)

    train_dataset = FlickerDataset(args,
                                   args.train_path,
                                   transform["train"])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=(args.train_mode == "train"),
                              num_workers=args.num_workers)

    valid_dataset = FlickerDataset(args,
                                   args.valid_path,
                                   transform["valid"])
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=(args.train_mode == "train"),
                              num_workers=args.num_workers)

    loader = dict(
        train=train_loader,
        valid=valid_loader
    )

    return loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from args import get_args

    args = get_args()

    loader = get_loaders(args)

    for data in loader["train"]:
        gray = data["gray"]
        high_res = data["high_resolution"]
        low_res = data["low_resolution"]
        print(gray.shape, high_res.shape, low_res.shape)
        plt.imshow(gray[0].numpy().astype('uint8'), cmap='gray')
        plt.show()
        plt.imshow(high_res[0].permute(1, 2, 0).numpy().astype('uint8'))
        plt.show()
        plt.imshow(low_res[0].permute(1, 2, 0).numpy().astype('uint8'))
        plt.show()

        break
