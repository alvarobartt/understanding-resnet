from torchvision import transforms as T

train_transform = T.Compose([
    T.Pad(padding=4), # Can be removed and replaced with T.RandomCrop(size=(32, 32), padding=4)
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=(32, 32)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])