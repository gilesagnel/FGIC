from utils.options import TrainOptions
from dataset.dataset import create_dataloader
from models import get_model


if __name__ == "__main__":
    opt = TrainOptions()
    train_loader, val_loader = create_dataloader(opt)
    model = get_model(opt)
    model.load_model()

    model.train()

    for epoch in range(opt.start_epoch, opt.n_epochs + 1):
        model.setup_loss()
        batch_idx = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            model.train_step(inputs, targets)

            if batch_idx % opt.print_batch_freq == 0:
                model.print_metrics(epoch, batch_idx)

        model.print_metrics(epoch, batch_idx, type="epoch")
        model.lr_scheduler_step(epoch)

        if epoch % opt.print_freq == 0:
            print("Validation:")
            model.evaluate(val_loader)

        if epoch % opt.save_epoch_freq == 0:
            model.save_model(epoch)



