import torch
from models.cmal_model import CmalModel
from models.resnet_model import ResnetModel

def get_model(opt):
    if 'resnet' in opt.cls_model:
        model = ResnetModel(opt)
    elif opt.cls_model == "cmal":
        model = CmalModel(opt)
    return model

def check_model(model):
    device = torch.device('mps')
    model.eval() # prevent introducing randomness from dropout and other modules
    model.to(device)

    input = torch.randn(5, 3, 224, 224, requires_grad=True, device=device) 
    output = model(input)
    i = 2
    loss = output[i].sum()
    print("Loss:", loss.item())
    loss.backward()

    # Check the gradients, value should be 0 expect at index 2
    print("Gradients of the input:")
    print(input.grad[:, :1, :1, :1])
