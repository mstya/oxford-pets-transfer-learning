import torch
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomCrop, RandomErasing, PILToTensor, \
    ConvertImageDtype, ToTensor, Normalize, ColorJitter, Compose, Resize, RandomResizedCrop


def define_transformations(augmentation, mean, std):
    train_transforms = build_train_transformations(augmentation, mean, std)
    train_transformations = Compose(train_transforms)

    val_transformations = Compose([
        ToTensor(),
        Normalize(mean, std)
    ])

    return train_transformations, val_transformations

def build_train_transformations(augmentation, mean, std):
    transform_map = {
        "random_horizontal_flip": lambda : RandomHorizontalFlip(p=float(augmentation['random_horizontal_flip']['p'])),
        "random_rotation": lambda : RandomRotation(degrees=int(augmentation['random_rotation']['degrees'])),
        "random_crop": lambda : RandomCrop(size=int(augmentation['random_crop']['size']), padding=int(augmentation['random_crop']['padding'])),
        "random_erasing": lambda : RandomErasing(),
        "pil_to_tensor": lambda : PILToTensor(),
        "convert_image_dtype": lambda : ConvertImageDtype(torch.float),
        "to_tensor": lambda : ToTensor(),
        "normalize": lambda : Normalize(mean=mean, std=std),
        "color_jitter": lambda : ColorJitter(brightness=.5, hue=.3),
        "resize": lambda : Resize(size=int(augmentation['resize']['size'])),
        "random_resized_crop": lambda : RandomResizedCrop(size=int(augmentation['random_resized_crop']['size'])),
    }

    transforms = []
    for k in augmentation.keys():
        transforms.append(transform_map[k]())

    return transforms