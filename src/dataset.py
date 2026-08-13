import torch
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomCrop, RandomErasing, PILToTensor, \
    ConvertImageDtype, ToTensor, Normalize, ColorJitter, Compose, Resize, RandomResizedCrop


def define_transformations(augmentation, mean, std):
    train_transforms = build_train_transformations(augmentation, mean, std)
    train_transformations = Compose(train_transforms)

    val_transformations = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean, std)
    ])

    return train_transformations, val_transformations

def build_train_transformations(augmentation, mean, std):
    transform_map = {
        "random_horizontal_flip": lambda t: RandomHorizontalFlip(p=float(t['settings']['p'])),
        "random_rotation": lambda t: RandomRotation(degrees=int(t['settings']['degrees'])),
        "random_crop": lambda t: RandomCrop(size=int(t['settings']['size']), padding=int(t['settings']['padding'])),
        "random_erasing": lambda t: RandomErasing(p=0.1, scale=(0.02, 0.1)),
        "pil_to_tensor": lambda t: PILToTensor(),
        "convert_image_dtype": lambda t: ConvertImageDtype(torch.float),
        "to_tensor": lambda _: ToTensor(),
        "normalize": lambda _: Normalize(mean=mean, std=std),
        "color_jitter": lambda _: ColorJitter(brightness=.5, hue=.3),
        "resize": lambda t: Resize(size=int(t['settings']['size'])),
        "random_resized_crop": lambda t: RandomResizedCrop(size=int(t['settings']['size'])),
    }

    transforms = []

    for tr in sorted(augmentation, key=lambda x: x['order']):
        transforms.append(transform_map[tr['name']](tr))

    return transforms