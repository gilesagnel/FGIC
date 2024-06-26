from dataclasses import dataclass, field
import os


@dataclass
class Options:
    data_root: str = field(init=False)
    isTrain: bool = False
    model_name: str = "seeds"
    device: str = "mps"

    num_class = 88
    img_load_size: int = 224 # 550 / 224
    img_crop_size: int = 224 # 448 / 224
    img_crop_padding: int = 8
    zip_path: str = "dataset/Seed dataset.zip"
    base_model:str = "resnet50" # "resnet18" / "resnet34" / "resnet50"  / "resNext50"
    cls_model: str = "cmal" # "cmal" / "resnet" 
    
    lr: float = 2e-4
    start_epoch: int = 1
    n_epochs: int = 15
    lr_step_size: int = 5
    continue_train: bool = False
    load_epoch: int = 0
    print_freq: int = 1
    print_batch_freq: int = 5
    batch_size: int = 16
    save_epoch_freq: int = 1
    num_threads: int = 2

    checkpoints_dir: str = './checkpoints'

    # mean and std for normalize the image
    mean: list = field(init=False)
    std: list = field(init=False)
    preprocess: list =field(init=False)

    def __post_init__(self):
        self.data_root = f"./dataset/{self.model_name}"
        self.mean = [0.4263, 0.3897, 0.3341] # [0.5, 0.5, 0.5] / [0.4263, 0.3897, 0.3341]
        self.std = [0.2910, 0.2742, 0.2455] # [0.5, 0.5, 0.5] / [0.2910, 0.2742, 0.2455]
        self.preprocess = ['resize', 'h-flip', 'v-flip', 'rotate'] # 'resize', 'crop', 'h-flip', 'v-flip', 'rotate'



@dataclass
class TrainOptions(Options):
    isTrain: bool = True
    continue_train: bool = False
    start_epoch: int = 1
    load_epoch: int = 0

@dataclass
class TestOptions(Options):
    isTrain: bool = False
    load_epoch: int = 12
    num_threads = 1  
    batch_size = 1    
