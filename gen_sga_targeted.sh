################################################################################################################################################
#sga cifar10
for TARGET_CLASS in {0,1,2,3,5,4,6,7,8,9}
do
    CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=cifar10 --targeted=1 --target_class=$TARGET_CLASS --model_file_name=wideresnet_cifar10.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/cifar10_wideresnet_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size 250 --minibatch 10 --alpha 10 --epoch 2 --spgd 0 --num_images 10000 --model_name wideresnet --Momentum 0 --cross_loss 1
done