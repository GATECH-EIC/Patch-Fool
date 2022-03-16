import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import numpy as np
import time

from models.DeiT import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224
from models.resnet import ResNet50, ResNet152, ResNet101
from utils import clamp, get_loaders, my_logger, my_meter, PCGrad




def get_aug():
    parser = argparse.ArgumentParser(description='Patch-Fool Training')

    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', default='ImageNet', type=str)
    parser.add_argument('--data_dir', default='/data1/ImageNet/ILSVRC/Data/CLS-LOC/', type=str)
    parser.add_argument('--log_dir', default='log', type=str)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--workers', default=16, type=int)

    parser.add_argument('--network', default='DeiT-B', type=str, choices=['DeiT-B', 'DeiT-S', 'DeiT-T',
                                                                           'ResNet152', 'ResNet50', 'ResNet18'])
    parser.add_argument('--dataset_size', default=1.0, type=float, help='Use part of Eval set')

    parser.add_argument('--patch_select', default='Attn', type=str, choices=['Rand', 'Saliency', 'Attn'])
    parser.add_argument('--num_patch', default=1, type=int)
    parser.add_argument('--sparse_pixel_num', default=0, type=int)

    parser.add_argument('--attack_mode', default='CE_loss', choices=['CE_loss', 'Attention'], type=str)
    parser.add_argument('--atten_loss_weight', default=0.002, type=float)
    parser.add_argument('--atten_select', default=4, type=int, help='Select patch based on which attention layer')
    parser.add_argument('--mild_l_2', default=0., type=float, help='Range: 0-16')
    parser.add_argument('--mild_l_inf', default=0., type=float, help='Range: 0-1')

    parser.add_argument('--train_attack_iters', default=250, type=int)
    parser.add_argument('--random_sparse_pixel', action='store_true', help='random select sparse pixel or not')
    parser.add_argument('--learnable_mask_stop', default=200, type=int)

    parser.add_argument('--attack_learning_rate', default=0.22, type=float)
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--gamma', default=0.95, type=float)

    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()

    if args.mild_l_2 != 0 and args.mild_l_inf != 0:
        print(f'Only one parameter can be non-zero: mild_l_2 {args.mild_l_2}, mild_l_inf {args.mild_l_inf}')
        raise NotImplementedError
    if args.mild_l_inf > 1:
        args.mild_l_inf /= 255.
        print(f'mild_l_inf > 1. Constrain all the perturbation with mild_l_inf/255={args.mild_l_inf}')

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    return args


def main():
    args = get_aug()


    logger = my_logger(args)
    meter = my_meter()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    patch_size = 16    
    filter = torch.ones([1, 3, patch_size, patch_size]).float().cuda()

    if args.network == 'ResNet152':
        model = ResNet152(pretrained=True)
    elif args.network == 'ResNet50':
        model = ResNet50(pretrained=True)
    elif args.network == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=True)  
    elif args.network == 'VGG16':
        model = torchvision.models.vgg16(pretrained=True)
    elif args.network == 'DeiT-T':
        model = deit_tiny_patch16_224(pretrained=True)
    elif args.network == 'DeiT-S':
        model = deit_small_patch16_224(pretrained=True)
    elif args.network == 'DeiT-B':
        model = deit_base_patch16_224(pretrained=True)
    else:
        print('Wrong Network')
        raise

    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()
    # eval dataset
    loader = get_loaders(args)
    mu = torch.tensor(args.mu).view(3, 1, 1).cuda()
    std = torch.tensor(args.std).view(3, 1, 1).cuda()

    start_time = time.time()

    '''Original image been classified incorrect but turn to be correct after adv attack'''
    false2true_num = 0

    for i, (X, y) in enumerate(loader):
        '''not using all of the eval dataset to get the final result'''
        if i == int(len(loader) * args.dataset_size):
            break

        X, y = X.cuda(), y.cuda()
        patch_num_per_line = int(X.size(-1) / patch_size)
        delta = torch.zeros_like(X).cuda()
        delta.requires_grad = True

        model.zero_grad()
        if 'DeiT' in args.network:
            out, atten = model(X + delta)
        else:
            out = model(X + delta)

        classification_result = out.max(1)[1] == y
        correct_num = classification_result.sum().item()
        loss = criterion(out, y)
        meter.add_loss_acc("Base", {'CE': loss.item()}, correct_num, y.size(0))

        '''choose patch'''
        # max_patch_index size: [Batch, num_patch attack]
        if args.patch_select == 'Rand':
            '''random choose patch'''
            max_patch_index = np.random.randint(0, 14 * 14, (X.size(0), args.num_patch))
            max_patch_index = torch.from_numpy(max_patch_index)
        elif args.patch_select == 'Saliency':
            '''gradient based method'''
            grad = torch.autograd.grad(loss, delta)[0]
            # print(grad.shape)
            grad = torch.abs(grad)
            patch_grad = F.conv2d(grad, filter, stride=patch_size)
            patch_grad = patch_grad.view(patch_grad.size(0), -1)
            max_patch_index = patch_grad.argsort(descending=True)[:, :args.num_patch]
        elif args.patch_select == 'Attn':
            '''attention based method'''
            atten_layer = atten[args.atten_select].mean(dim=1)
            if 'DeiT' in args.network:
                atten_layer = atten_layer.mean(dim=-2)[:, 1:]
            else:
                atten_layer = atten_layer.mean(dim=-2)
            max_patch_index = atten_layer.argsort(descending=True)[:, :args.num_patch]
        else:
            print(f'Unknown patch_select: {args.patch_select}')
            raise

        '''build mask'''
        mask = torch.zeros([X.size(0), 1, X.size(2), X.size(3)]).cuda()
        if args.sparse_pixel_num != 0:
            learnable_mask = mask.clone()

        for j in range(X.size(0)):
            index_list = max_patch_index[j]
            for index in index_list:
                row = (index // patch_num_per_line) * patch_size
                column = (index % patch_num_per_line) * patch_size

                if args.sparse_pixel_num != 0:
                    learnable_mask.data[j, :, row:row + patch_size, column:column + patch_size] = torch.rand(
                        [patch_size, patch_size])
                mask[j, :, row:row + patch_size, column:column + patch_size] = 1

        '''adv attack'''
        max_patch_index_matrix = max_patch_index[:, 0]
        max_patch_index_matrix = max_patch_index_matrix.repeat(197, 1)
        max_patch_index_matrix = max_patch_index_matrix.permute(1, 0)
        max_patch_index_matrix = max_patch_index_matrix.flatten().long()

        if args.mild_l_inf == 0:
            '''random init delta'''
            delta = (torch.rand_like(X) - mu) / std
        else:
            '''constrain delta: range [x-epsilon, x+epsilon]'''
            epsilon = args.mild_l_inf / std
            delta = 2 * epsilon * torch.rand_like(X) - epsilon + X

        delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)
        original_img = X.clone()

        if args.random_sparse_pixel:
            '''random select pixels'''
            sparse_mask = torch.zeros_like(mask)
            learnable_mask_temp = learnable_mask.view(learnable_mask.size(0), -1)
            sparse_mask_temp = sparse_mask.view(sparse_mask.size(0), -1)
            value, _ = learnable_mask_temp.sort(descending=True)
            threshold = value[:, args.sparse_pixel_num - 1].view(-1, 1)
            sparse_mask_temp[learnable_mask_temp >= threshold] = 1
            mask = sparse_mask

        if args.sparse_pixel_num == 0 or args.random_sparse_pixel:
            X = torch.mul(X, 1 - mask)
        else:
            '''select by learnable mask'''
            learnable_mask.requires_grad = True
        delta = delta.cuda()
        delta.requires_grad = True

        opt = torch.optim.Adam([delta], lr=args.attack_learning_rate)
        if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
            mask_opt = torch.optim.Adam([learnable_mask], lr=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)

        '''Start Adv Attack'''
        for train_iter_num in range(args.train_attack_iters):
            model.zero_grad()
            opt.zero_grad()

            '''Build Sparse Patch attack binary mask'''
            if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
                if train_iter_num < args.learnable_mask_stop:
                    mask_opt.zero_grad()
                    sparse_mask = torch.zeros_like(mask)
                    learnable_mask_temp = learnable_mask.view(learnable_mask.size(0), -1)
                    sparse_mask_temp = sparse_mask.view(sparse_mask.size(0), -1)
                    value, _ = learnable_mask_temp.sort(descending=True)

                    threshold = value[:, args.sparse_pixel_num-1].view(-1, 1)
                    sparse_mask_temp[learnable_mask_temp >= threshold] = 1

                    '''inference as sparse_mask but backward as learnable_mask'''
                    temp_mask = ((sparse_mask - learnable_mask).detach() + learnable_mask) * mask
                else:
                    temp_mask = sparse_mask

                X = original_img * (1-sparse_mask)
                if 'DeiT' in args.network:
                    out, atten = model(X + torch.mul(delta, temp_mask))
                else:
                    out = model(X + torch.mul(delta, temp_mask))

            else:
                if 'DeiT' in args.network:
                    out, atten = model(X + torch.mul(delta, mask))
                else:
                    out = model(X + torch.mul(delta, mask))

            '''final CE-loss'''
            loss = criterion(out, y)

            if args.attack_mode == 'Attention':
                grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
                ce_loss_grad_temp = grad.view(X.size(0), -1).detach().clone()
                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                    mask_grad = torch.autograd.grad(loss, learnable_mask, retain_graph=True)[0]

                # Attack the first 6 layers' Attn
                range_list = range(len(atten)//2)
                for atten_num in range_list:
                    if atten_num == 0:
                        continue
                    atten_map = atten[atten_num]
                    atten_map = atten_map.mean(dim=1)
                    atten_map = atten_map.view(-1, atten_map.size(-1))
                    atten_map = -torch.log(atten_map)
                    if 'DeiT' in args.network:
                        atten_loss = F.nll_loss(atten_map, max_patch_index_matrix + 1)
                    else:
                        atten_loss = F.nll_loss(atten_map, max_patch_index_matrix)

                    atten_grad = torch.autograd.grad(atten_loss, delta, retain_graph=True)[0]

                    atten_grad_temp = atten_grad.view(X.size(0), -1)
                    cos_sim = F.cosine_similarity(atten_grad_temp, ce_loss_grad_temp, dim=1)

                    if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                        mask_atten_grad = torch.autograd.grad(atten_loss, learnable_mask, retain_graph=True)[0]

                    '''PCGrad'''
                    atten_grad = PCGrad(atten_grad_temp, ce_loss_grad_temp, cos_sim, grad.shape)
                    if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
                        mask_atten_grad_temp = mask_atten_grad.view(mask_atten_grad.size(0), -1)
                        ce_mask_grad_temp = mask_grad.view(mask_grad.size(0), -1)
                        mask_cos_sim = F.cosine_similarity(mask_atten_grad_temp, ce_mask_grad_temp, dim=1)
                        mask_atten_grad = PCGrad(mask_atten_grad_temp, ce_mask_grad_temp, mask_cos_sim, mask_atten_grad.shape)

                    grad += atten_grad * args.atten_loss_weight
                    if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
                        mask_grad += mask_atten_grad * args.atten_loss_weight

            else:
                '''no attention loss'''
                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                    grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
                    mask_grad = torch.autograd.grad(loss, learnable_mask)[0]
                else:
                    grad = torch.autograd.grad(loss, delta)[0]

            opt.zero_grad()
            delta.grad = -grad
            opt.step()
            scheduler.step()

            if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                mask_opt.zero_grad()
                learnable_mask.grad = -mask_grad
                mask_opt.step()

                learnable_mask_temp = learnable_mask.view(X.size(0), -1)
                learnable_mask.data -= learnable_mask_temp.min(-1)[0].view(-1, 1, 1, 1)
                learnable_mask.data += 1e-6
                learnable_mask.data *= mask

            '''l2 constrain'''
            if args.mild_l_2 != 0:
                radius = (args.mild_l_2 / std).squeeze()
                perturbation = (delta.detach() - original_img) * mask
                l2 = torch.linalg.norm(perturbation.view(perturbation.size(0), perturbation.size(1), -1), dim=-1)
                radius = radius.repeat([l2.size(0), 1])
                l2_constraint = radius / l2
                l2_constraint[l2 < radius] = 1.
                l2_constraint = l2_constraint.view(l2_constraint.size(0), l2_constraint.size(1), 1, 1)
                delta.data = original_img + perturbation * l2_constraint

            '''l_inf constrain'''
            if args.mild_l_inf != 0:
                epsilon = args.mild_l_inf / std
                delta.data = clamp(delta, original_img - epsilon, original_img + epsilon)

            delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)

        '''Eval Adv Attack'''
        with torch.no_grad():
            if args.sparse_pixel_num == 0 or args.random_sparse_pixel:
                perturb_x = X + torch.mul(delta, mask)
                if 'DeiT' in args.network:
                    out, atten = model(perturb_x)
                else:
                    out = model(perturb_x)
            else:
                if train_iter_num < args.learnable_mask_stop:
                    sparse_mask = torch.zeros_like(mask)
                    learnable_mask_temp = learnable_mask.view(learnable_mask.size(0), -1)
                    temp_mask = sparse_mask.view(sparse_mask.size(0), -1)
                    value, _ = learnable_mask_temp.sort(descending=True)
                    threshold = value[:, args.sparse_pixel_num - 1].view(-1, 1)
                    temp_mask[learnable_mask_temp >= threshold] = 1

                print((sparse_mask * mask).view(mask.size(0), -1).sum(-1))
                print("xxxxxxxxxxxxxxxxxxxxxx")
                X = original_img * (1 - sparse_mask)
                perturb_x = X + torch.mul(delta, sparse_mask)
                if 'DeiT' in args.network:
                    out, atten = model(perturb_x)
                else:
                    out = model(perturb_x)

            classification_result_after_attack = out.max(1)[1] == y
            loss = criterion(out, y)
            meter.add_loss_acc("ADV", {'CE': loss.item()}, (classification_result_after_attack.sum().item()), y.size(0))

        '''Message'''
        if i % 1 == 0:
            logger.info("Iter: [{:d}/{:d}] Loss and Acc for all models:".format(i, int(len(loader) * args.dataset_size)))
            msg = meter.get_loss_acc_msg()
            logger.info(msg)

            classification_result_after_attack = classification_result_after_attack[classification_result == False]
            false2true_num += classification_result_after_attack.sum().item()
            logger.info("Total False -> True: {}".format(false2true_num))

    end_time = time.time()
    msg = meter.get_loss_acc_msg()
    logger.info("\nFinish! Using time: {}\n{}".format((end_time - start_time), msg))


if __name__ == "__main__":
    main()
