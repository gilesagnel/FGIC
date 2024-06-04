"""
Dichao Liu, Longjiao Zhao, Yu Wang, Jien Kato,
Learn from each other to Classify better: Cross-layer mutual attention learning for fine-grained visual classification,
Pattern Recognition,
Volume 140,
2023,
109550,
ISSN 0031-3203,
https://doi.org/10.1016/j.patcog.2023.109550.
(https://www.sciencedirect.com/science/article/pii/S0031320323002509)
"""

from models.base_model import BaseModel
from torch.autograd import Variable
import torch.optim as optim
from models.cmal import CMAL
import torch
import numpy as np

from utils.options import Options


class CmalModel(BaseModel):
    def __init__(self, opt: Options):
        BaseModel.__init__(self, opt)
        if 'crop' in opt.preprocess:
            ks = opt.img_crop_size // 8
        else:
            ks = opt.img_load_size // 8
        self.model = CMAL(opt.num_class, ks, opt.base_model)
        self.model.to(self.device)
        
        self.loss_fn =  torch.nn.CrossEntropyLoss()
        if self.isTrain:
            self.lr = [1e-2] * 7 + [1e-4] * 3
            parameter_groups = [self.model.c_all.parameters(), 
                                self.model.cb1.parameters(), self.model.c1.parameters(), 
                                self.model.cb2.parameters(), self.model.c2.parameters(), 
                                self.model.cb3.parameters(), self.model.c3.parameters(), 
                                self.model.fe_1.parameters(), self.model.fe_2.parameters(), self.model.fe_3.parameters() ]
            optimizer_params = [{'params': params, 'lr': lr} for params, lr in zip(parameter_groups, self.lr)]
            self.optimizer = optim.SGD(optimizer_params, momentum=0.9)
            
    #  Step: 5 | Loss1: 5.033 | Loss2: 4.98512 | Loss3: 4.96999 | Loss_ATT: 18.16286 | Loss_concat: 2.12873 | Loss: 17.117 | Acc: 81.250% (65/80)
    def train_step(self, inputs, targets, epoch=None):
        if inputs.shape[0] < self.opt.batch_size:
            return
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs, targets = Variable(inputs), Variable(targets)

        # Train the experts from deep to shallow with data augmentation by multiple steps
        # e3
        self.optimizer.zero_grad()
        inputs3 = inputs
        output_1, output_2, output_3, _, map1, map2, map3 = self.model(inputs3)
        loss3 = self.loss_fn(output_3, targets) * 1
        loss3.backward()
        self.optimizer.step()

        att_map_3, inputs3_att = self.generate_attention_map(map3, inputs, output_3, 3)
        att_map_2, inputs2_att = self.generate_attention_map(map2, inputs, output_2, 2)
        att_map_1, inputs1_att = self.generate_attention_map(map1, inputs, output_1, 1)
        inputs_ATT = self.highlight_im(inputs, att_map_1, att_map_2, att_map_3)

        # e2
        self.optimizer.zero_grad()
        flag = torch.rand(1)
        if flag < (1 / 3):
            inputs2 = inputs3_att
        elif (1 / 3) <= flag < (2 / 3):
            inputs2 = inputs1_att
        elif flag >= (2 / 3):
            inputs2 = inputs

        _, output_2, _, _, _, map2, _ = self.model(inputs2)
        loss2 = self.loss_fn(output_2, targets) * 1
        loss2.backward()
        self.optimizer.step()

        # e1
        self.optimizer.zero_grad()
        flag = torch.rand(1)
        if flag < (1 / 3):
            inputs1 = inputs3_att
        elif (1 / 3) <= flag < (2 / 3):
            inputs1 = inputs2_att
        elif flag >= (2 / 3):
            inputs1 = inputs

        output_1, _, _, _, map1, _, _ = self.model(inputs1)
        loss1 = self.loss_fn(output_1, targets) * 1
        loss1.backward()
        self.optimizer.step()


        # Train the experts and their concatenation with the overall attention region in one go
        self.optimizer.zero_grad()
        output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = self.model(inputs_ATT)
        concat_loss_ATT = self.loss_fn(output_1_ATT, targets)+\
                        self.loss_fn(output_2_ATT, targets)+\
                        self.loss_fn(output_3_ATT, targets)+\
                        self.loss_fn(output_concat_ATT, targets) * 2
        concat_loss_ATT.backward()
        self.optimizer.step()


        # Train the concatenation of the experts with the raw input
        self.optimizer.zero_grad()
        _, _, _, output_concat, _, _, _ = self.model(inputs)
        concat_loss = self.loss_fn(output_concat, targets) * 2
        concat_loss.backward()
        self.optimizer.step()

        _, predicted = torch.max(output_concat.data, 1)
        self.total += targets.size(0)
        self.correct += predicted.eq(targets.data).cpu().sum()

        self.train_loss[0] += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
        self.train_loss[1] += loss1.item()
        self.train_loss[2] += loss2.item()
        self.train_loss[3] += loss3.item()
        self.train_loss[4] += concat_loss_ATT.item()
        self.train_loss[5] += concat_loss.item()

        for nlr in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[nlr]['lr'] = self.cosine_anneal_schedule(epoch, self.lr[nlr])

    def lr_scheduler_step(self, epoch):
        pass
        # for nlr in range(len(self.optimizer.param_groups)):
        #     self.optimizer.param_groups[nlr]['lr'] = self.cosine_anneal_schedule(epoch, self.lr[nlr])
    
    def setup_loss(self):
        self.train_loss = np.zeros(6)
        self.correct = 0
        self.total = 0

    def evaluate(self, data_loader, epoch=None, is_val=False):
        self.model.eval()
        test_loss = 0
        correct = 0
        correct_com = 0
        correct_com2 = 0
        total = 0
        batch_idx = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs, targets
                output_1, output_2, output_3, output_concat, map1, map2, map3 = self.model(inputs)

                att_map_3, _ = self.generate_attention_map(map3, inputs, output_3, 3)
                att_map_2, _ = self.generate_attention_map(map2, inputs, output_2, 2)
                att_map_1, _ = self.generate_attention_map(map1, inputs, output_1, 1)

                inputs_ATT = self.highlight_im(inputs, att_map_1, att_map_2, att_map_3)
                output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = self.model(inputs_ATT)

                outputs_com2 = output_1 + output_2 + output_3 + output_concat
                outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT

                loss = self.loss_fn(output_concat, targets)

                test_loss += loss.item()
                _, predicted = torch.max(output_concat.data, 1)
                _, predicted_com = torch.max(outputs_com.data, 1)
                _, predicted_com2 = torch.max(outputs_com2.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()
                correct_com += predicted_com.eq(targets.data).cpu().sum()
                correct_com2 += predicted_com2.eq(targets.data).cpu().sum()

        test_acc_1 = 100. * float(correct) / total
        test_acc_2 = 100. * float(correct_com) / total
        test_acc_3 = 100. * float(correct_com2) / total
        test_loss = test_loss / (batch_idx + 1)
        if is_val:
            self.writter.add_scalar("Val/Loss", test_loss, epoch)
            self.writter.add_scalar("Val/Accuarcy 1", test_acc_1, epoch)
            self.writter.add_scalar("Val/Accuarcy 2", test_acc_2, epoch)
            self.writter.add_scalar("Val/Accuarcy 3", test_acc_3, epoch)
            print("Validation:")
        print(f'Loss: {test_loss:.3f} | Acc 1: {test_acc_1:.3f}% | Acc 2: {test_acc_2:.3f}% |'
              f' Acc 3: {test_acc_3:.3f}%' )
        self.model.train()
    
    def generate_attention_map(self, map, inputs, output, level):
        p1 = self.model.state_dict()[f'c{level}.1.weight']
        p2 = self.model.state_dict()[f'c{level}.4.weight']
        att_map = self.map_generate(map, output, p1, p2)
        inputs_att = self.attention_im(inputs, att_map)
        return att_map, inputs_att
    
    def print_metrics(self, epoch, batch_idx, metric_type="batch"):
        loss1 = self.train_loss[1] / batch_idx
        loss2 = self.train_loss[2] / batch_idx
        loss3 = self.train_loss[3] / batch_idx
        loss_att = self.train_loss[4] / batch_idx
        loss_concat = self.train_loss[5] / batch_idx
        total_loss = self.train_loss[0] / batch_idx
        accuracy = 100. * float(self.correct) / self.total

        metric_name = f"Epoch {epoch} Step" if metric_type == "batch" else "Iteration"
        idx = batch_idx if metric_type == "batch" else epoch

        print(f'{metric_name}: {idx} | Loss1: {loss1:.3f} | Loss2: {loss2:.5f} | '
              f'Loss3: {loss3:.5f} | Loss_ATT: {loss_att:.5f} | Loss_concat: {loss_concat:.5f} | '
              f'Loss: {total_loss:.3f} | Acc: {accuracy:.3f}% ({self.correct}/{self.total})')

        if metric_type == "epoch":
            self.writter.add_scalar("Train/Loss/1", loss1, epoch)
            self.writter.add_scalar("Train/Loss/2", loss2, epoch)
            self.writter.add_scalar("Train/Loss/3", loss3, epoch)
            self.writter.add_scalar("Train/Loss/ATT", loss_att, epoch)
            self.writter.add_scalar("Train/Loss/concat", loss_concat, epoch)
            self.writter.add_scalar("Train/Loss/all", total_loss, epoch)
            self.writter.add_scalar("Train/Accuracy", accuracy, epoch)
    
    def cosine_anneal_schedule(self, t, lr):
        cos_inner = np.pi * (t % (self.opt.n_epochs))  
        cos_inner /= (self.opt.n_epochs)
        cos_out = np.cos(cos_inner) + 1

        return float(lr / 2 * cos_out)

    @staticmethod
    def map_generate(attention_map, pred, p1, p2):
        batches, feaC, feaH, feaW = attention_map.size()

        out_map=torch.zeros_like(attention_map.mean(1))

        for batch_index in range(batches):
            map_tpm = attention_map[batch_index]
            map_tpm = map_tpm.reshape(feaC, feaH*feaW)
            map_tpm = map_tpm.permute([1, 0])
            p1_tmp = p1.permute([1, 0])
            map_tpm = torch.mm(map_tpm, p1_tmp)
            map_tpm = map_tpm.permute([1, 0])
            map_tpm = map_tpm.reshape(map_tpm.size(0), feaH, feaW)

            pred_tmp = pred[batch_index]
            pred_ind = pred_tmp.argmax()
            p2_tmp = p2[pred_ind].unsqueeze(1)

            map_tpm = map_tpm.reshape(map_tpm.size(0), feaH * feaW)
            map_tpm = map_tpm.permute([1, 0])
            map_tpm = torch.mm(map_tpm, p2_tmp)
            out_map[batch_index] = map_tpm.reshape(feaH, feaW)

        return out_map

    @staticmethod
    def attention_im(images, attention_map, theta=0.5, padding_ratio=0.1):
        images = images.clone()
        attention_map = attention_map.clone().detach()
        batches, _, imgH, imgW = images.size()

        for batch_index in range(batches):
            image_tmp = images[batch_index]
            map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
            map_tpm = torch.nn.functional.interpolate(map_tpm, size=(imgH, imgW)).squeeze()
            map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)
            map_tpm = map_tpm >= theta
            nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
            image_tmp = torch.nn.functional.interpolate(image_tmp, size=(imgH, imgW)).squeeze()

            images[batch_index] = image_tmp

        return images
    
    @staticmethod
    def highlight_im(images, attention_map, attention_map2, attention_map3, theta=0.5, padding_ratio=0.1):
        images = images.clone()
        attention_map = attention_map.clone().detach()
        attention_map2 = attention_map2.clone().detach()
        attention_map3 = attention_map3.clone().detach()

        batches, _, imgH, imgW = images.size()

        for batch_index in range(batches):
            image_tmp = images[batch_index]
            map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
            map_tpm = torch.nn.functional.interpolate(map_tpm, size=(imgH, imgW)).squeeze()
            map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)


            map_tpm2 = attention_map2[batch_index].unsqueeze(0).unsqueeze(0)
            map_tpm2 = torch.nn.functional.interpolate(map_tpm2, size=(imgH, imgW)).squeeze()
            map_tpm2 = (map_tpm2 - map_tpm2.min()) / (map_tpm2.max() - map_tpm2.min() + 1e-6)

            map_tpm3 = attention_map3[batch_index].unsqueeze(0).unsqueeze(0)
            map_tpm3 = torch.nn.functional.interpolate(map_tpm3, size=(imgH, imgW)).squeeze()
            map_tpm3 = (map_tpm3 - map_tpm3.min()) / (map_tpm3.max() - map_tpm3.min() + 1e-6)

            map_tpm = (map_tpm + map_tpm2 + map_tpm3)
            map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)
            map_tpm = map_tpm >= theta

            nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
            image_tmp = torch.nn.functional.interpolate(image_tmp, size=(imgH, imgW)).squeeze()

            images[batch_index] = image_tmp

        return images