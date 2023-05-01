import albumentations as A

# noop = A.Compose([])

baseline_augment = A.Compose([
    # A.RandomCrop(width=96, height=96),
    # A.Rotate(limit=[-45, 45]),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
])