from torch import nn
import torch
from models.base_model import BaseModel
from utils.options import Options
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
import torch.optim as optim

class ResnetModel(BaseModel):
    def __init__(self, opt: Options):
        BaseModel.__init__(self, opt)
        if opt.cls_model == 'resnet18':
            if opt.pre_trained:
                weights = ResNet18_Weights.IMAGENET1K_V1
            self.model = resnet18(weights=weights)
        else:
            if opt.pre_trained:
                weights = ResNet34_Weights.IMAGENET1K_V1
            self.model = resnet34(weights=weights)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, opt.num_class)
        self.model.to(self.device)

        if opt.isTrain:
            self.loss_fn = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_step_size)
            self.train_loss = 0

    def setup_loss(self):
        self.train_loss = 0
        self.correct = 0
        self.total = 0 
    
    def train_step(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()

        self.train_loss += loss.item() * 1

        _, predicted = torch.max(outputs.data, 1)
        self.total += targets.size(0)
        self.correct += predicted.eq(targets.data).cpu().sum()

    def lr_scheduler_step(self, _):
        self.scheduler.step()

    def print_metrics(self, epoch, batch_idx, type="batch"):
        loss = self.train_loss / batch_idx
        accuracy = 100. * float(self.correct) / self.total

        metric_name = "Step" if type == "batch" else "Iteration"
        idx = batch_idx if type == "batch" else epoch

        print(f'{metric_name}: {idx} | Loss: {loss:.3f} | '
              f'Acc: {accuracy:.3f}% ({self.correct}/{self.total})')
        
    def evaluate(self, data_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        batch_idx = 0

        
        for batch_idx, (inputs, targets) in enumerate(data_loader, start=1):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                test_loss += loss.item() * 1
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

        total_test_acc = 100. * float(correct) / total
        total_test_loss = test_loss / batch_idx
        print(f'Loss: {total_test_loss:.3f} | Acc: {total_test_acc:.3f}%' )
        self.model.train()

