from models.cmal_model import CmalModel
from models.resnet_model import ResnetModel

def get_model(opt):
    if 'resnet' in opt.cls_model:
        model = ResnetModel(opt)
    elif opt.cls_model == "cmal":
        model = CmalModel(opt)
    return model