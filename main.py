# -*- coding:utf-8 -*-
import copy

import torch.utils.model_zoo as model_zoo
import inceptionv3_classlist
import utils
from skimage import transform as skimage_transform
import torch
gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if gpu else "cpu")
loss_func = torch.nn.CrossEntropyLoss()
import os
import matplotlib.pyplot as plt
import torchvision.models as models
import numpy as np
import random
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image
from torch.functional import F
def clip(x):
    x = np.minimum(x, 1)
    return np.maximum(-1, x)

def trans(data):
    data = data * 0.5 + 0.5
    return data

def image_multi_patch(_image, _patch, padding_mask,  _cfg=utils.config):
    assert(_image.shape[1] == 3)
    # plt.subplot(1,3,1)
    # a = trans(_image[0].transpose((1, 2, 0)))
    # plt.imshow(a)
    # plt.subplot(1,3,2)
    # b = trans(_patch[0].transpose((1, 2, 0)))
    # plt.imshow(b)
    temp = copy.deepcopy(_image)
    temp -= temp * padding_mask
    _mix = temp + _patch
    # plt.subplot(1,3,3)
    # c = trans(_mix[0].transpose((1, 2, 0)))
    # plt.imshow(c)
    # plt.show()
    # temp=input(':')
    return _mix

def cook_patch(cfg=utils.config):
    _model = inceptionv3()
    _model.to(device)
    for param in _model.parameters():
        param.requires_grad = True

    file_name = os.listdir(cfg["input_path"])
    # _step = 2
    for each in file_name:
        if(each.endswith('JPEG') or each.endswith('jpg') or each.endswith('png')):
            original_image = plt.imread(cfg["input_path"] + each)[:, :, :3]
            # original_image = plt.imread(r'C:\Users\33119\Desktop\WorkSpace\DFTdefense\attack\LaVAN_pytorch\output\(210, 210)_390_ILSVRC2012_val_00000293.png')[:, :, :3]


            original_image_resize = skimage_transform.resize(original_image, (299, 299)).transpose((2, 0, 1))
            original_tensor = torch.Tensor([original_image_resize])

            normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            original_tensor = normalizer(original_tensor).type(torch.FloatTensor)
            original_tensor = original_tensor.to(device)


        patch_np =np.zeros(original_tensor.shape)

        mask =  np.ones((1, 3, cfg["patch_size"], cfg["patch_size"]))
        padding_mask = np.pad(mask,
                              ((0, 0), (0, 0),
                               (cfg["patch_loc"][0], 299 - cfg["patch_loc"][0] - cfg["patch_size"]),
                               (cfg["patch_loc"][1], 299 - cfg["patch_loc"][1] - cfg["patch_size"])
                               ), 'constant', constant_values=0)


        # baseline 预测
        _model.eval()
        _baseline = _model(original_tensor)
        _, pred_idx_baseline = torch.max(_baseline, 1)
        pred_prob = F.softmax(_baseline, dim=-1)[0]
        pred_prob = float(pred_prob.cpu().detach().numpy()[pred_idx_baseline])
        pred_idx_baseline = int(pred_idx_baseline.cpu().item())
        pred_tag_baseline = inceptionv3_classlist.__class_v3[pred_idx_baseline]
        _model.zero_grad()
        if cfg["target_idx"] == -1:
            target_idx = torch.Tensor([random.randint(0, 1000)]).long().to(device)
            while (target_idx == pred_idx_baseline):
                target_idx = torch.Tensor([random.randint(0, 1000)]).long().to(device)
        else:
            target_idx = torch.Tensor([cfg["target_idx"]]).long().to(device)

        pbar = tqdm(total=cfg["epoch"], ascii=True)

        origin_image_np = original_tensor.cpu().detach().numpy()
        pred_idx_baseline = torch.Tensor([pred_idx_baseline]).long().to(device)
        for _epoch in range(cfg["epoch"]):

            mix_numpy = image_multi_patch(origin_image_np, patch_np, padding_mask)

            mix_tensor = Variable(torch.Tensor(mix_numpy), requires_grad=True)
            mix_tensor_cuda = mix_tensor.to(device)
            mix_tensor_cuda.retain_grad()
            output = _model(mix_tensor_cuda)
            _, pred_idx = torch.max(output, 1)
            pred_max_prob = F.softmax(output, dim=-1)[0]
            pred_max_prob_np = pred_max_prob.cpu().detach().numpy()
            pred_max_prob = float(pred_max_prob_np[pred_idx])
            loss_target = loss_func(output, target_idx)
            loss_target.backward()
            grad_loss_target = mix_tensor_cuda.grad
            _model.zero_grad()


            output = _model(mix_tensor_cuda)
            _, src_idx = torch.max(output, 1)
            loss_source = loss_func(output, pred_idx_baseline)
            loss_source.backward()
            grad_loss_source = mix_tensor_cuda.grad
            _model.zero_grad()

            diff = grad_loss_target - grad_loss_source
            diff = - diff * 0.1
            diff = diff.cpu().detach().numpy() * padding_mask
            patch_np = mix_tensor_cuda.cpu().detach().numpy() * padding_mask
            patch_np += diff
            patch_np = clip(patch_np)
            pbar.update(1)
            evil_idx = pred_idx.cpu().detach().numpy()[0]
            evil_prob = pred_max_prob
            pred_idx_baseline_num = pred_idx_baseline.cpu().detach().item()
            pbar.set_postfix_str("class={}/{} prob={:.4}%/{:.4}%".format(
                evil_idx, pred_idx_baseline_num,
                evil_prob * 100, pred_prob * 100
            ))
            if evil_idx != pred_idx_baseline and evil_prob > 0.9:
                _save = Image.fromarray((trans(mix_numpy[0].transpose((1, 2, 0))) * 255).astype(np.uint8) )\
                    .save(os.path.join(cfg['output_path'], "({}, {})_{}_{}".format(cfg['patch_loc'][0], cfg['patch_loc'][1], pred_idx_baseline_num, each.split('.')[0] + '.png')))
                break

        pbar.close()
    mix_tensor = mix_tensor.cpu().detach().numpy()[0].transpose((1, 2, 3))
    return mix_tensor

def load_pretrained(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    return model

def inceptionv3(num_classes=1000, pretrained='imagenet'):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    """
    input_sizes = {}
    means = {}
    stds = {}
    pretrained_settings = {}
    for model_name in ['inceptionv3']:
        input_sizes[model_name] = [3, 299, 299]
        means[model_name] = [0.5, 0.5, 0.5]
        stds[model_name] = [0.5, 0.5, 0.5]
    model = models.inception_v3(pretrained=True)
    if pretrained is not None:
        pretrained_settings['inceptionv3'] = {
            'imagenet': {
                'url': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
                'input_space': 'RGB',
                'input_size': input_sizes[model_name],
                'input_range': [0, 1],
                'mean': means[model_name],
                'std': stds[model_name],
                'num_classes': 1000
            }
        }
        settings = pretrained_settings['inceptionv3'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    return model

def filetest():

    success_patch = cook_patch()
    plt.imshow(success_patch)
    plt.show()


if __name__ == "__main__":
    filetest()