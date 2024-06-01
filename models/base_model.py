import os
import torch
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter

from utils.options import Options


class BaseModel(ABC):
    def __init__(self, opt: Options):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = torch.device(opt.device)
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.model_name)  
        self.model = None
        self.writter = SummaryWriter()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    @abstractmethod     
    def setup_loss(self):
        pass

    @abstractmethod
    def train_step(self, inputs, targets):
        pass

    @abstractmethod    
    def lr_scheduler_step(self, epoch):
        pass

    @abstractmethod     
    def print_metrics(self, epoch, batch_idx, metric_type="batch"):
        pass
    
    @abstractmethod     
    def evaluate(self, data_loader, epoch=None, is_val=False):
        pass

    def save_model(self, epoch):
        save_filename = f"{epoch}_model_{self.opt.model_name}.pth"
        save_path = os.path.join(self.save_dir, save_filename)

        if self.opt.device in ["cuda", "mps"]:
            self.model = self.model.cpu()

        torch.save(self.model.state_dict(), save_path)

        if self.opt.device in ["cuda", "mps"]:
            self.model.to(self.device)

    def load_model(self):
        if self.opt.load_epoch == 0 or not self.opt.continue_train:
            return
        model_filename = f"{self.opt.load_epoch}_model_{self.opt.model_name}.pth"
        state_dict = torch.load(os.path.join(self.save_dir, model_filename))
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {model_filename}")
        self.model.to(self.device)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
