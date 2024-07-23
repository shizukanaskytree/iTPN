import math
import torch
from torchvision import transforms

from datasets import load_dataset
from torch.utils.data import Dataset

from transforms import RandomResizedCropAndInterpolationWithTwoResolution
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, \
    IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from masking_generator import MaskingGenerator

### done
def map2pixel4peco(x):
    return x * 255

### done
class DataAugmentationForEVA(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = (0.48145466, 0.4578275, 0.40821073) if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = (0.26862954, 0.26130258, 0.27577711) if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        self.common_transform = [
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoResolution(
                size=args.input_size, second_size=args.second_input_size, scale=args.crop_scale, ratio=args.crop_ratio,
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ]

        if args.color_jitter > 0:
            self.common_transform = \
                [transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter)] + \
                self.common_transform

        self.common_transform = transforms.Compose(self.common_transform)

        self.patch_transform = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std
            )
        ]
        self.patch_transform = transforms.Compose(self.patch_transform)

        self.visual_token_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073) if 'clip' in args.teacher_type else IMAGENET_INCEPTION_MEAN,
                std=(0.26862954, 0.26130258, 0.27577711) if 'clip' in args.teacher_type else IMAGENET_INCEPTION_STD,
            ),
        ])

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForEVA,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


class TransformedDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item['image'].convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, item['label']


def build_eva_pretraining_dataset(is_train, args):
    transform = DataAugmentationForEVA(args)
    print("Data Aug = %s" % str(transform))
    if is_train:
        dataset = load_dataset('imagenet-1k', split='train')
    else:
        dataset = load_dataset('imagenet-1k', split='validation')
    transformed_dataset = TransformedDataset(dataset, transform)
    nb_classes = 1000
    return transformed_dataset, nb_classes

### done
class RandomResizedCrop(transforms.RandomResizedCrop):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """
    @staticmethod
    def get_params(img, scale, ratio):
        width, height = F.get_image_size(img)
        area = height * width

        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()

        return i, j, h, w

### done
def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = (0.48145466, 0.4578275, 0.40821073) if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = (0.26862954, 0.26130258, 0.27577711) if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        if args.linear_probe:
            return transforms.Compose([
                RandomResizedCrop(args.input_size, interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)],
            )
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            no_aug=args.no_aug,
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
            scale=args.scale
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

