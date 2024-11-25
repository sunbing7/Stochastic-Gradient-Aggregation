import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.realpath('..'))
import argparse
import datetime
from attack_spgd import uap_spgd
from attacks_sga import uap_sga, uap_sga_targeted
from utils import *
from prepare_imagenet_data import create_imagenet_npy
from network import *
from util.data import *


def main(args):
    print(args)
    time1 = datetime.datetime.now()
    dir_uap = args.uaps_save + '/' + str(args.dataset) + '_'+ str(args.model_name) + '/'
    batch_size = args.batch_size
    DEVICE = torch.device("cuda:0")
    torch.manual_seed(0)

    if not args.targeted:
        model_dimension = 299 if args.model_name == 'inception_v3' else 256
        center_crop = 299 if args.model_name == 'inception_v3' else 224
        X = create_imagenet_npy(args.data_dir, len_batch=args.num_images, model_dimension=model_dimension,
                                center_crop=center_crop)
        loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True, num_workers=0)
        loader_eval = torch.utils.data.DataLoader(X, batch_size=100, shuffle=True, num_workers=0)
        model = model_imgnet(args.model_name)
    else:
        _, data_test = get_data(args.dataset, args.dataset, is_attack=True)

        loader_eval = torch.utils.data.DataLoader(data_test,
                                                  batch_size=args.test_batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=True)

        num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)
        center_crop = input_size
        data_train, _ = get_data(args.dataset, args.dataset, is_attack=True)

        loader = torch.utils.data.DataLoader(data_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             pin_memory=True)

        model = get_my_model(args.weight_path, args.model_file_name, args.model_name, args.dataset)

        model = nn.DataParallel(model).cuda()
        # Normalization wrapper, so that we don't have to normalize adversarial perturbations
        normalize = Normalizer(mean=mean, std=std)
        model = nn.Sequential(normalize, model)

        model = model.cuda()
    nb_epoch = args.epoch
    eps = args.alpha / 255
    beta = args.beta
    step_decay = args.step_decay
    if args.spgd:
        uap,losses = uap_spgd(model,
                              loader,
                              nb_epoch,
                              eps,
                              beta,
                              step_decay,
                              loss_function=args.cross_loss,
                              batch_size = batch_size,
                              loader_eval=loader_eval,
                              dir_uap = dir_uap,
                              center_crop=center_crop,
                              Momentum=args.Momentum,
                              img_num=args.num_images)
    else:
        if args.targeted:
            uap,losses = uap_sga_targeted(model, loader, nb_epoch, eps, beta, step_decay,
                                          loss_function=args.cross_loss,
                                          target_class=args.target_class,
                                          batch_size=batch_size,
                                          minibatch=args.minibatch,
                                          loader_eval=loader_eval,
                                          dir_uap=dir_uap,
                                          center_crop=center_crop,
                                          iter=args.iter,
                                          Momentum=args.Momentum,
                                          img_num=args.num_images)
        else:
            uap,losses = uap_sga(model, loader, nb_epoch, eps, beta, step_decay, loss_function=args.cross_loss,
                                 batch_size=batch_size, minibatch=args.minibatch, loader_eval=loader_eval,
                                 dir_uap = dir_uap,center_crop=center_crop,iter=args.iter,
                                 Momentum=args.Momentum,img_num=args.num_images)

    _, _, _, _, outputs, labels, y_outputs = evaluate(model, loader_eval, uap=uap,
                                                      batch_size=args.test_batch_size, DEVICE=DEVICE)

    print('true image Accuracy:', sum(y_outputs == labels) / len(labels))
    print('adversarial image Accuracy:', sum(outputs == labels) / len(labels))
    print('fooling rate:', 1-sum(outputs == labels) / len(labels))
    print('fooling ratio:', 1-sum(y_outputs == outputs) / len(labels))

    if args.targeted:
        print('Target attack success rate:', sum(outputs == args.target_class) / len(labels))


    if args.spgd:
        save_name = 'spgd_' + args.model_name
    else:
        save_name = 'sga_' + args.model_name

    plt.plot(losses)
    np.save(dir_uap + "losses.npy", losses)
    plt.savefig(dir_uap + save_name + '_loss_epoch.png')

    if args.targeted:
        post_fix = str(args.target_class)
    else:
        post_fix = 'nontarget'
    torch.save(uap, dir_uap + 'uap_' + post_fix +'.pth')

    time2 = datetime.datetime.now()
    print("time consumed: ", time2 - time1)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imagenet',
                        help='dataset')
    parser.add_argument('--data_dir', default='/../imagenet/train/',
                        help='training set directory')
    parser.add_argument('--uaps_save', default='./uaps_save/spgd/',
                        help='training set directory')
    parser.add_argument('--batch_size', type=int, help='batch size', default=250)
    parser.add_argument('--test_batch_size', type=int, help='batch size', default=12)
    parser.add_argument('--minibatch', type=int, help='inner batch size for SGA', default=10)
    parser.add_argument('--alpha', type=float, default=10, help='aximum perturbation value (L-infinity) norm')
    parser.add_argument('--beta', type=float, default=9, help='clamping value')
    parser.add_argument('--step_decay', type=float, default=0.1, help='step size')
    parser.add_argument('--epoch', type=int, default=20, help='epoch num')
    parser.add_argument('--spgd', type=int,default=1, help='loss type')
    parser.add_argument('--num_images', type=int, default=10000, help='num of training images')
    parser.add_argument('--model_name', default='vgg16', help='proxy model')
    parser.add_argument('--model_file_name', default='vgg16')
    parser.add_argument('--weight_path', default='/../imagenet/train/')
    parser.add_argument('--iter', type=int,default=4, help='inner iteration num')
    parser.add_argument('--Momentum', type=int, default=0, help='Momentum item')
    parser.add_argument('--cross_loss', type=int, default=0, help='loss type')
    parser.add_argument('--targeted', type=int, default=0, help='set to 1 if targeted attack')
    parser.add_argument('--target_class', type=int, default=0, help='target class')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))