# -*- coding: utf-8 -*-
import os
import argparse
import io

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from MyDataSet import JDDataSet
from FeatureExtractor import FeatureExtractor



def main(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    os.environ['TORCH_HOME'] = 'models'
    # vgg16 = torch.utils.model_zoo.load_url('vgg16.pth', model_dir="./models")
    vgg16 = getattr(models, args.arch)(pretrained=True)
    print(vgg16)
    # for name, module in vgg16._modules.iteritems():
    #     print(name, module)
    # resnet101 = models.resnet101(pretrained=True)
    # print(resnet101)

    model = FeatureExtractor(vgg16, ["avgpool"])
    model.eval()
    if args.gpu is not None:
        model = model.cuda()

    # cats = ["厨具", "个人护理", "家居日用", "食品饮料", "手机通讯"]
    cats = ["chuju", "gerenhuli", "jiajuriyong", "shipinyinliao", "shoujitongxun"]
    cat_set = list(set(cats))
    cat_set.sort()
    cat2id = dict(map(lambda i_cat: (i_cat[1], i_cat[0]), enumerate(cat_set)))

    jddataset = JDDataSet("", args.data, img_transforms, cats=cats)
    print(jddataset.__len__())
    dataloader = torch.utils.data.DataLoader(jddataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers)
    with torch.no_grad():
        with io.open('res.txt', 'w+', encoding='utf-8') as fout:
            cnt = 0
            for batch_img, batch_cats in dataloader:
                if args.gpu is not None:
                    batch_img = batch_img.cuda()
                output = model(batch_img)
                output = output[0].reshape(args.batch_size, -1).cpu().numpy()

                for i in range(len(batch_cats)):
                    embed = map(lambda i_d: str(i_d[0] + 1) + ':' + '{:.3f}'.format(i_d[1]),
                               list(filter(lambda i_d: i_d[1] != 0, enumerate(output[i]))))
                    fout.write(str(cat2id.get(batch_cats[i])) +" "+ ' '.join(embed) +'\n')
                fout.flush()

                cnt += 1
                print(cnt)
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=256, type=int)
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    args = parser.parse_args()
    main(args)