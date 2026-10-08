import numpy as np
import os
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None



def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.08, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # The configured ViT-B/16 IN21K checkpoint expects inputs scaled
            # from [0, 1] to [-1, 1] (mean/std = 0.5 per channel).
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    
    # return transforms.Compose(t)
    return t

class iCIFAR224(iData):
    use_path = False

    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [
        # transforms.ToTensor(),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        prepared_root = os.environ.get("RSIAT_DATASET_ROOT")
        if prepared_root and os.path.isdir(os.path.join(prepared_root, "train")):
            train_dataset = datasets.ImageFolder(os.path.join(prepared_root, "train"))
            test_dataset = datasets.ImageFolder(os.path.join(prepared_root, "test"))
            self.train_data, self.train_targets = split_images_labels(train_dataset.imgs)
            self.test_data, self.test_targets = split_images_labels(test_dataset.imgs)
            self.use_path = True
            return
        root = os.environ.get("RSIAT_CIFAR_ROOT", "./data/datasets")
        train_dataset = datasets.cifar.CIFAR100(root, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(root, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )




class iImageNetR(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]


    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir, test_dir = _imagefolder_paths("imagenet-r")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir, test_dir = _imagefolder_paths("imagenet-a")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class CUB(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir, test_dir = _imagefolder_paths("cub")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class vtab(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        train_dir, test_dir = _imagefolder_paths("vtab")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class omnibenchmark(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        train_dir, test_dir = _imagefolder_paths("omnibenchmark")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


def _imagefolder_paths(dataset_name):
    """Return prepared split paths, honoring the registry-selected data root."""
    root = os.environ.get("RSIAT_DATASET_ROOT")
    if root:
        base = root
    else:
        data_root = os.environ.get("RSIAT_DATA_ROOT", "./data/datasets")
        base = os.path.join(data_root, dataset_name)
        if not os.path.isdir(base) and "-" in dataset_name:
            base = os.path.join(data_root, dataset_name.replace("-", "_"))
    train_dir, test_dir = os.path.join(base, "train"), os.path.join(base, "test")
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Prepared dataset is missing train/test directories: {base}. "
            "Run scripts/prepare_dataset.py first."
        )
    return train_dir, test_dir
