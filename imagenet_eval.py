import os
import sys
import torch
import argparse
import datetime
from network import *
from util.data import *
sys.path.append(os.path.realpath('..'))

from utils import loader_imgnet, model_imgnet, evaluate

def main(args):
    print(args)
    DEVICE = torch.device("cuda:0")
    time1 = datetime.datetime.now()
    dir_data = args.data_dir
    if args.targeted:
        dir_uap = args.uaps_save + '/' + str(args.dataset) + '_'+ str(args.model_name) + '/uap_' + str(args.target_class) + '.pth'
    else:
        dir_uap = args.uaps_save
    batch_size = args.batch_size

    if not args.targeted:
        model_dimension = 299 if args.model_name == 'inception_v3' else 256
        center_crop = 299 if args.model_name == 'inception_v3' else 224
        loader = loader_imgnet(dir_data, 50000, batch_size, model_dimension,center_crop)

        model = model_imgnet(args.model_name)
    else:
        _, data_test = get_data(args.dataset, args.dataset, is_attack=True)
        loader = torch.utils.data.DataLoader(data_test,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=True)
        num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)
        model = get_my_model(args.weight_path, args.model_file_name, args.model_name, args.dataset)
        model = model.cuda()

    uap = torch.load(dir_uap)

    if args.targeted:
        tstd = torch.from_numpy(np.array(std).reshape(1, 3, 1, 1))
        uap = uap / tstd

    _, _, _, _, outputs, labels, y_outputs = evaluate(model, loader, uap = uap,batch_size=batch_size,DEVICE = DEVICE)

    print('true image Accuracy:', sum(y_outputs == labels) / len(labels))
    print('adversarial image Accuracy:', sum(outputs == labels) / len(labels))
    print('fooling rate:', 1-sum(outputs == labels) / len(labels))
    print('fooling ratio:', 1-sum(y_outputs == outputs) / len(labels))

    if args.targeted:
        print('Target attack success rate:', sum(outputs == args.target_class) / len(labels))

    time2 = datetime.datetime.now()
    print("time consumed: ", time2 - time1)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imagenet',
                        help='dataset')
    parser.add_argument('--data_dir', default='/../imagent/val/',
                        help='training set directory')
    parser.add_argument('--uaps_save', default='./uaps_save/spgd/spgd_10000_20epoch_250batch.pth',
                        help='training set directory')
    parser.add_argument('--batch_size', type=int, help='', default=250)
    parser.add_argument('--model_name', default='vgg16', help='loss type')
    parser.add_argument('--model_file_name', default='resnet50_imagenet.pth')
    parser.add_argument('--weight_path', default='/../imagenet/train/')
    parser.add_argument('--targeted', type=int, default=0, help='set to 1 if targeted attack')
    parser.add_argument('--target_class', type=int, default=0, help='target class')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))