import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from os import path
import os
import matplotlib.pyplot as plt
import seaborn as sns


mu = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(args):
    args.mu = mu
    args.std = std
    valdir = path.join(args.data_dir, 'val')
    val_dataset = datasets.ImageFolder(valdir,
                                       transforms.Compose([transforms.Resize(args.img_size),
                                                           transforms.CenterCrop(args.crop_size),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=args.mu, std=args.std)
                                                           ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.workers, pin_memory=True)
    return val_loader

def visualize_loss(loss_list):
    plt.figure()
    plt.plot(loss_list)
    plt.savefig('output/loss/loss{:.4f}_{}_{}_{}_{}_{}.png'.format(loss_list[-1], args.network, args.attack_learning_rate,
                                                          args.train_attack_iters, args.step_size, args.gamma))
    plt.close()

def visualize_attention_map(atten1, atten2, image1, image2, original_result, after_attack_result, max_patch_index):
    atten1 = [x.mean(dim=1).cpu() for x in atten1]
    atten2 = [x.mean(dim=1).cpu() for x in atten2]
    image1 = image1.cpu()
    image2 = image2.cpu()
    original_result = original_result.cpu()
    after_attack_result = after_attack_result.cpu()

    if image1.size(0) > 4:
        atten1 = [x[:4] for x in atten1]
        atten2 = [x[:4] for x in atten2]
        image1 = image1[:4]
        image2 = image2[:4]
        original_result = original_result[:4]
        after_attack_result = after_attack_result[:4]
    pic_num = image1.size(0)
    if 'LeViT' in args.network:
        patch_num = atten1[0].size(-1)
    else:
        patch_num = atten1[0].size(-1) - 1
    patch_per_line = int(patch_num ** 0.5)
    patch_size = int(image1.size(-1) / patch_per_line)
    to_PIL = transforms.ToPILImage()

    for i in range(pic_num):
        if not path.exists('output/{}'.format(i)):
            os.mkdir('output/{}'.format(i))

        original_img = to_PIL(image1[i].squeeze())
        after_attack_img = to_PIL(image2[i].squeeze())
        original_atten = [x[i] for x in atten1]  # [197x197]
        after_attack_atten = [x[i] for x in atten2]

        with open('output/{}/atten.txt'.format(i), 'w') as f:
            print("Base model result: {}\tAttack model result:{}".format(original_result[i], after_attack_result[i]))
            print("Base model result: {}\tAttack model result:{}".format(original_result[i],
                                                                         after_attack_result[i]), file=f)
            for j in [4]:  # one layer
            # for j in range(len(original_atten)):  # each block
                print("Processing Image:{}\tLayer:{}".format(i, j))
                original_block_layer = original_atten[j]
                after_attack_atten_layer = after_attack_atten[j]
                vmin = min(original_block_layer.min(), after_attack_atten_layer.min())
                vmax = max(original_block_layer.max(), after_attack_atten_layer.max())
                plt.figure(figsize=(70, 30))
                plt.subplot(1, 2, 1)
                plt.title('Original')
                sns.heatmap(original_block_layer.data, annot=False, vmin=vmin, vmax=vmax)
                plt.subplot(1, 2, 2)
                plt.title('Attack patch {}'.format(max_patch_index[i] + 1))
                sns.heatmap(after_attack_atten_layer.data, annot=False, vmin=vmin, vmax=vmax)
                plt.savefig('output/{}/atten_layer{}.png'.format(i, j))
                plt.close()

                original_block_layer = original_block_layer.mean(dim=0)
                after_attack_atten_layer = after_attack_atten_layer.mean(dim=0)
                print('layer_{}'.format(j), file=f)
                print(original_block_layer, file=f)
                print(' ', file=f)
                print(after_attack_atten_layer, file=f)
                print(' ', file=f)
                print(after_attack_atten_layer - original_block_layer, file=f)

                plt.figure()
                plt.subplot(2, 2, 1)
                plt.imshow(original_img)
                plt.subplot(2, 2, 2)
                plt.imshow(after_attack_img)

                if 'DeiT' in args.network:
                    original_block_layer = original_block_layer[1:]
                    after_attack_atten_layer = after_attack_atten_layer[1:]
                plt.subplot(2, 2, 3)
                sns.heatmap(original_block_layer.view(patch_per_line, patch_per_line).data, annot=False)
                plt.subplot(2, 2, 4)
                sns.heatmap(after_attack_atten_layer.view(patch_per_line, patch_per_line).data, annot=False)
                plt.savefig('output/{}/atten_layer{}_img.png'.format(i, j))
                plt.close()





    # filter = torch.ones([1, 3, patch_size, patch_size])
    # atten = F.conv_transpose2d(atten, filter, stride=patch_size)
    # add_atten = torch.mul(atten, image)


'''
@Parameter atten_grad, ce_grad: should be 2D tensor with shape [batch_size, -1]
'''
def PCGrad(atten_grad, ce_grad, sim, shape):
    pcgrad = atten_grad[sim < 0]
    temp_ce_grad = ce_grad[sim < 0]
    dot_prod = torch.mul(pcgrad, temp_ce_grad).sum(dim=-1)
    dot_prod = dot_prod / torch.norm(temp_ce_grad, dim=-1)
    pcgrad = pcgrad - dot_prod.view(-1, 1) * temp_ce_grad
    atten_grad[sim < 0] = pcgrad
    atten_grad = atten_grad.view(shape)
    return atten_grad


'''
random shift several patches within the range
'''
def shift_image(image, range, mu, std, patch_size=16):
    batch_size, channel, h, w = image.shape
    h_range, w_range = range
    new_h = h + 2 * h_range * patch_size
    new_w = w + 2 * w_range * patch_size
    new_image = torch.zeros([batch_size, channel, new_h, new_w]).cuda()
    new_image = (new_image - mu) / std
    shift_h = np.random.randint(-h_range, h_range+1)
    shift_w = np.random.randint(-w_range, w_range+1)
    # shift_h = np.random.randint(-1, 2)
    # shift_w = 0
    new_image[:, :, h_range*patch_size : h+h_range*patch_size, w_range*patch_size : w + w_range*patch_size] = image.detach()
    h_start = (h_range + shift_h) * patch_size
    w_start = (w_range + shift_w) * patch_size
    new_image = new_image[:, :, h_start : h_start+h, w_start : w_start+w]
    return new_image



class my_logger:
    def __init__(self, args):
        name = "{}_{}_{}_{}_{}.log".format(args.name, args.network, args.dataset, args.train_attack_iters,
                                           args.attack_learning_rate)
        args.name = name
        self.name = path.join(args.log_dir, name)
        with open(self.name, 'w') as F:
            print('\n'.join(['%s:%s' % item for item in args.__dict__.items() if item[0][0] != '_']), file=F)
            print('\n', file=F)

    def info(self, content):
        with open(self.name, 'a') as F:
            print(content)
            print(content, file=F)


class my_meter:
    def __init__(self):
        self.meter_list = {}

    def add_loss_acc(self, model_name, loss_dic: dict, correct_num, batch_size):
        if model_name not in self.meter_list.keys():
            self.meter_list[model_name] = self.model_meter()
        sub_meter = self.meter_list[model_name]
        sub_meter.add_loss_acc(loss_dic, correct_num, batch_size)

    def clean_meter(self):
        for key in self.meter_list.keys():
            self.meter_list[key].clean_meter()

    def get_loss_acc_msg(self):
        msg = []
        for key in self.meter_list.keys():
            sub_meter = self.meter_list[key]
            sub_loss_bag = sub_meter.get_loss()
            loss_msg = ["{}: {:.4f}({:.4f})".format(x, sub_meter.last_loss[x], sub_loss_bag[x])
                        for x in sub_loss_bag.keys()]
            loss_msg = " ".join(loss_msg)
            msg.append("model:{} Loss:{} Acc:{:.4f}({:.4f})".format(
                key, loss_msg, sub_meter.last_acc, sub_meter.get_acc()))
        msg = "\n".join(msg)
        return msg

    class model_meter:
        def __init__(self):
            self.loss_bag = {}
            self.acc = 0.
            self.count = 0
            self.last_loss = {}
            self.last_acc = 0.

        def add_loss_acc(self, loss_dic: dict, correct_num, batch_size):
            for loss_name in loss_dic.keys():
                if loss_name not in self.loss_bag.keys():
                    self.loss_bag[loss_name] = 0.
                self.loss_bag[loss_name] += loss_dic[loss_name] * batch_size
            self.last_loss = loss_dic
            self.last_acc = correct_num / batch_size
            self.acc += correct_num
            self.count += batch_size

        def get_loss(self):
            return {x: self.loss_bag[x] / self.count for x in self.loss_bag.keys()}

        def get_acc(self):
            return self.acc / self.count

        def clean_meter(self):
            self.__init__()
