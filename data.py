"""Načítanie a rozdelenie datasetu Oxford-IIIT Pet (Kaggle verzia, všetky jpg v jednom priečinku)."""
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class PetsDataset(Dataset):
    """Trieda sa odvodzuje z názvu súboru: 'american_bulldog_23.jpg' -> 'american_bulldog'."""

    def __init__(self, root, transform=None):
        self.files = sorted(Path(root).glob("*.jpg"))
        if not self.files:
            raise FileNotFoundError(f"V priečinku {root} sa nenašli žiadne .jpg súbory.")
        names = [f.stem.rsplit("_", 1)[0] for f in self.files]
        self.classes = sorted(set(names))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = [self.class_to_idx[n] for n in names]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")  # pár obrázkov je sivých / CMYK
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def get_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def get_dataloaders(img_dir, img_size=128, batch_size=32, seed=42, num_workers=0):
    """Stratifikované rozdelenie 70/15/15. Validácia a test bez augmentácie."""
    train_tf, eval_tf = get_transforms(img_size)
    ds_train = PetsDataset(img_dir, train_tf)
    ds_eval = PetsDataset(img_dir, eval_tf)  # tie isté súbory, iná transformácia

    idx = list(range(len(ds_train)))
    train_idx, tmp_idx = train_test_split(
        idx, test_size=0.3, stratify=ds_train.labels, random_state=seed)
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=0.5,
        stratify=[ds_train.labels[i] for i in tmp_idx], random_state=seed)

    train_dl = DataLoader(Subset(ds_train, train_idx), batch_size=batch_size,
                          shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(Subset(ds_eval, val_idx), batch_size=batch_size * 2,
                        num_workers=num_workers)
    test_dl = DataLoader(Subset(ds_eval, test_idx), batch_size=batch_size * 2,
                         num_workers=num_workers)
    return train_dl, val_dl, test_dl, ds_train.classes
