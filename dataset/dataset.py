import os
import shutil
import pandas as pd
import zipfile
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils.options import Options


def prepare_seed_dataset():
    # For seed dataset download the dataset zip file from the 
    # https://figshare.com/articles/figure/A_dataset_based_on_smartphone_acquisition_that_can_be_used_for_seed_identification_using_deep_learning_models/24552394/1
    # And place it in {opt.zip_path}
    opt = Options()
    if not os.path.exists(opt.zip_path):
        return

    with zipfile.ZipFile(opt.zip_path, 'r') as zip_ref:
        zip_ref.extractall("dataset")
    os.rename(opt.zip_path.split(".")[0], opt.data_root)
    os.remove(opt.zip_path)
    
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
            count = 0
            for f in os.listdir(os.path.join(root, dir_name)):
                count += 1
                if count > 3:
                    break
                X.append(os.path.join(data_root, dir_name, f))
                y.append(dir_name)

    return X, y

def create_dataloader(opt: Options):
    transform_list = []
    if 'resize' in opt.preprocess:
        transform_list.append(transforms.Resize((opt.img_load_size, opt.img_load_size)))
    if 'crop' in opt.preprocess:
        transform_list.append(transforms.RandomCrop(opt.img_crop_size, padding=opt.img_crop_padding))
    if 'h-flip' in opt.preprocess:
        transform_list.append(transforms.RandomHorizontalFlip())
    if 'v-flip' in opt.preprocess:
        transform_list.append(transforms.RandomVerticalFlip())
    if 'rotate' in opt.preprocess:
        transform_list.append(transforms.RandomRotation(degrees=(0, 180)))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=opt.mean, std=opt.std)]
    )
    transform = transforms.Compose(transform_list)
    
    if opt.isTrain:
        train_set = ImageFolder(f"{opt.data_root}/train", transform=transform)
        val_set = ImageFolder(f"{opt.data_root}/val", transform=transform)
        train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, 
                              num_workers=opt.num_threads)
        val_loader = DataLoader(val_set, batch_size=opt.batch_size, num_workers=opt.num_threads)
        return train_loader, val_loader
    else:
        test_set = ImageFolder(f"{opt.data_root}/test", transform=transform)
        test_loader = DataLoader(test_set, batch_size=opt.batch_size, num_workers=opt.num_threads)
        return test_loader
