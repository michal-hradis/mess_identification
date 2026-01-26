import kornia.augmentation as KTA
import kornia.augmentation.container as KAC

GPU_AUGMENTATIONS = {
    "aug_1": KAC.AugmentationSequential(
        KTA.RandomHorizontalFlip(p=0.5),
        KTA.RandomVerticalFlip(p=0.5),
        KTA.RandomAffine(degrees=180, translate=(0.05, 0.05), scale=(0.9, 1.1), p=0.8),
        KTA.ColorJiggle(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        KTA.RandomErasing(scale=(0.02, 0.25), ratio=(0.3, 3.3), p=0.4),
        KTA.RandomGrayscale(p=0.1),
        data_keys=["input"],
    )
}

