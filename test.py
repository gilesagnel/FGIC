from utils.options import TestOptions
from dataset.dataset import create_dataloader
from models.model import get_model, check_accuracy, load_model

if __name__ == "__main__":
    opt = TestOptions()
    test_loader = create_dataloader(opt)
    model = get_model(opt)
    model = load_model(model, opt)
    model.eval()
    check_accuracy(model, test_loader, opt)