import albumentations as A
import torch

IMG_SIZE = 224

# ImageNet normalization (standard for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = A.Compose([
    # Stronger augmentation for better generalization
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.75, 1.0), p=1.0),
    A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Rotate(limit=15, p=1.0),
    ], p=0.8),
    A.Rotate(limit=25, p=0.7, border_mode=1),
    A.ElasticTransform(alpha=1.2, sigma=60, p=0.4),
    A.GridDropout(ratio=0.2, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
    A.GaussianBlur(blur_limit=5, p=0.3),
    A.RandomGamma(p=0.2),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, p=1.0),
    A.pytorch.transforms.ToTensorV2(),
], bbox_params=None)

val_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, p=1.0),
    A.pytorch.transforms.ToTensorV2(),
], bbox_params=None)

