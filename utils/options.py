from dataclasses import dataclass, field
import os


@dataclass
class Options:
    data_root: str = field(init=False)
    isTrain: bool = False
    model_name: str = "seeds"
    device: str = "mps"

    out_dim = 88
    pre_trained = True
    img_size: int = 224
    zip_path: str = "dataset/Seed dataset.zip"
    cls_model: str = "resnet34"
    
    lr: float = 2e-4
    epoch_count: int = 1
    n_epochs: int = 15
    lr_step_size: int = 5
    continue_train: bool = False
    load_epoch: int = 0
    print_freq: int = 3
    batch_size: int = 128
    save_epoch_freq: int = 3
    num_threads: int = 8

    checkpoints_dir: str = './checkpoints'
    save_dir: str = field(init=False)

    img_size: int = 224

    # mean and std for normalize the image
    mean: list = field(init=False)
    std: list = field(init=False)

    def __post_init__(self):
        self.data_root = f"./dataset/{self.model_name}"
        self.save_dir = os.path.join(self.checkpoints_dir, self.model_name)
        self.mean = [0.4263, 0.3897, 0.3341]
        self.std = [0.2910, 0.2742, 0.2455]



@dataclass
class TrainOptions(Options):
    isTrain: bool = True
    continue_train: bool = True
    epoch_count: int = 5
    load_epoch: int = 5

@dataclass
class TestOptions(Options):
    isTrain: bool = False
    load_epoch: int = 15
    num_threads = 1  
    batch_size = 1    