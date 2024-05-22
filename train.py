import torch.nn as nn
import torch.optim as optim
import pandas as pd

from utils.options import TrainOptions
from dataset.dataset import SeedImageDataset, scan_data_folder, create_dataloader
from models.model import get_model, check_accuracy, save_model


if __name__ == "__main__":
    opt = TrainOptions()
    train_loader, val_loader = create_dataloader(opt)
    model = get_model('resnet34', opt)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size)
    model.train()

    for epoch in range(opt.epoch_count, opt.n_epochs + 1):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(opt.device), labels.to(opt.device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch}/{opt.n_epochs}], Loss: {epoch_loss:.4f}')

        scheduler.step()

        if epoch % opt.print_freq == 0:
            check_accuracy(model, "train", train_loader, opt.device)
            check_accuracy(model, "val", val_loader, opt.device)

        if epoch % opt.save_epoch_freq == 0:
            save_model(model, epoch, opt)



