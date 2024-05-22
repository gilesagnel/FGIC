import os
import shutil
import pandas as pd
import zipfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms


def prepare_dataset(opt):
    if not os.path.exists(opt.zip_path):
        return

    with zipfile.ZipFile(opt.zip_path, 'r') as zip_ref:
        zip_ref.extractall("dataset")
    os.rename(opt.zip_path.split(".")[0], opt.data_root)
    os.remove(opt.zip_path)

    # create checkpoints dir
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    
    X, y = scan_data_folder(opt.data_root)
    df = pd.DataFrame({'X': X, 'y': y})
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['y'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['y'], random_state=42)

    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    for split, split_df in splits.items():
        split_dir = os.path.join(opt.data_root, split)
        os.makedirs(split_dir, exist_ok=True)
        for _, row in split_df.iterrows():
            src, cat_dir = row['X'], row['y']
            new_cat_dir = os.path.join(split_dir, cat_dir)
            if not os.path.exists(new_cat_dir):
                os.makedirs(new_cat_dir)
            dst = os.path.join(new_cat_dir,os.path.basename(src))
            shutil.move(src, dst)

    for label in df[~df['y'].isin(['train', 'val', 'test'])]['y'].unique():
        old_dir = os.path.join(opt.data_root, label)
        if os.path.exists(old_dir):
            os.removedirs(os.path.join(opt.data_root, label))

    

def scan_data_folder(data_root):
    X = []
    y = []
    for root, dirs, _ in os.walk(data_root):
        for dir_name in dirs:
            for f in os.listdir(os.path.join(root, dir_name)):
                X.append(os.path.join(data_root, dir_name, f))
                y.append(dir_name)

    return X, y

def create_dataloader(opt):
    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.ToTensor(),
        transforms.Normalize(mean=opt.mean, std=opt.std),
    ])
    
    if opt.isTrain:
        train_set = SeedImageDataset(os.path.join(opt.data_root, "train"), transform=transform)
        val_set = SeedImageDataset(os.path.join(opt.data_root, "val"), transform=transform)
        train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, 
                              num_workers=opt.num_threads)
        val_loader = DataLoader(val_set, batch_size=1)
        return train_loader, val_loader
    else:
        test_set = SeedImageDataset(os.path.join(opt.data_root, "test"), transform=transform)
        test_loader = DataLoader(test_set, batch_size=1)
        return test_loader



class SeedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.X = []
        self.y = []
        self.cat_labels = []
        self.root_dir = root_dir
        self.scan_data()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)
    
    def scan_data(self):
        self.X, self.y = scan_data_folder(self.root_dir)
        self.cat_labels = list(set(self.y))
        self.cat_labels.sort()


    def __getitem__(self, idx):
        image = Image.open(self.X[idx]).convert("RGB")
        label = self.cat_labels.index(self.y[idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    