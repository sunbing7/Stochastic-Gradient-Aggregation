CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --data_dir /home/pc/lxn/data/imagenet/train/ --uaps_save ./uaps_save/sga/ --batch_size 250 --minibatch 10 --alpha 10 --epoch 20 --spgd 0 --num_images 10000 --model_name vgg16 --Momentum 0 --cross_loss 1



CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --data_dir=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/val/ --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size 250 --minibatch 10 --alpha 10 --epoch 20 --spgd 0 --num_images 10000 --model_name vgg19 --Momentum 0 --cross_loss 1

CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --data_dir=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/val/ --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size 250 --minibatch 10 --alpha 10 --epoch 20 --spgd 0 --num_images 10000 --model_name resnet50 --Momentum 0 --cross_loss 1

#targeted
CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --targeted=1 --target_class=150 --data_dir=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/val/ --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size 250 --minibatch 10 --alpha 10 --epoch 5 --spgd 0 --num_images 10000 --model_name vgg19 --Momentum 0 --cross_loss 1