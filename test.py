from utils.options import TestOptions
from dataset.dataset import create_dataloader
from models import get_model

if __name__ == "__main__":
    opt = TestOptions()
    test_loader = create_dataloader(opt)
    model = get_model(opt)
    model.load_model()
    model.evaluate(test_loader)