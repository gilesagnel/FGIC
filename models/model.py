import os
import torch
from torch import nn
from sklearn.metrics import precision_score, recall_score
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights


def get_model(opt):
    weights = model = None
    if opt.cls_model == 'resnet18':
      if opt.pre_trained:
          weights = ResNet18_Weights.IMAGENET1K_V1
      model = resnet18(weights=weights)
    else:
      if opt.pre_trained:
          weights = ResNet34_Weights.IMAGENET1K_V1
      model = resnet34(weights=weights)
       
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, opt.out_dim)
    model.to(opt.device)
    return model

def load_model(model, opt):
    if opt.load_epoch == 0:
        return
    model_filename = f"{opt.load_epoch}_model_{opt.model_name}.pth"
    state_dict = torch.load(os.path.join(opt.save_dir, model_filename))
    model.load_state_dict(state_dict)
    print(f"Model loaded from {model_filename}")
    model.to(opt.device)
    return model

def save_model(model, epoch, opt):
    save_filename = f"{epoch}_model_{opt.model_name}.pth"
    save_path = os.path.join(opt.save_dir, save_filename)

    if opt.device in ["cuda", "mps"]:
        model = model.cpu()

    torch.save(model.state_dict(), save_path)

    if opt.device in ["cuda", "mps"]:
        model.to(opt.device)

def check_accuracy(model, loader, opt, type = None):
  model.eval()
  correct = 0
  total = 0
  all_labels = []
  all_preds = []
  if type == None:
    type = "train" if opt.isTrain else "test"
  with torch.no_grad():
      for images, labels in loader:
          images, labels = images.to(opt.device), labels.to(opt.device)

          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          all_labels.extend(labels.cpu().numpy())
          all_preds.extend(predicted.cpu().numpy())

  print(f'{type} Accuracy: {100 * correct / total:.2f}%')

  precision = precision_score(all_labels, all_preds, average='weighted')
  recall = recall_score(all_labels, all_preds, average='weighted')
  print(f'{type} Precision: {precision:.4f}')
  print(f'{type} Recall: {recall:.4f}')