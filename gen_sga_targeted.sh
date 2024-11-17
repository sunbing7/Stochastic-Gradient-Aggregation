################################################################################################################################################
#sga resnet50 imagenet
for TARGET_CLASS in {174,755,743,804,700,922,547,369}
do
    CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=imagenet --targeted=1 --target_class=$TARGET_CLASS --model_file_name=resnet50_imagenet.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/imagenet_resnet50_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size=250 --test_batch_size=100 --minibatch=10 --alpha=10 --epoch=10 --spgd=0 --num_images=10000 --model_name=resnet50 --Momentum=0 --cross_loss=1
done
################################################################################################################################################
#sga vgg19 imagenet
for TARGET_CLASS in {214,39,527,65,639,771,412}
do
    CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=imagenet --targeted=1 --target_class=$TARGET_CLASS --model_file_name=vgg19_imagenet.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/imagenet_vgg19_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size=250 --test_batch_size=100 --minibatch=10 --alpha=10 --epoch=10 --spgd=0 --num_images=10000 --model_name=vgg19 --Momentum=0 --cross_loss=1
done
################################################################################################################################################
#sga googlenet imagenet
for TARGET_CLASS in {573,807,541,240,475,753,762,505}
do
    CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=imagenet --targeted=1 --target_class=$TARGET_CLASS --model_file_name=googlenet_imagenet.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/imagenet_googlenet_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size=250 --test_batch_size=100 --minibatch=10 --alpha=10 --epoch=10 --spgd=0 --num_images=10000 --model_name=googlenet --Momentum=0 --cross_loss=1
done
################################################################################################################################################
#sga mobilenet asl
for TARGET_CLASS in {19,17,8,21,2,9,23,6}
do
    CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=asl --targeted=1 --target_class=$TARGET_CLASS --model_file_name=mobilenet_asl.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/asl_mobilenet_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size=250 --test_batch_size=100 --minibatch=10 --alpha=10 --epoch=5 --spgd=0 --num_images=10000 --model_name=mobilenet --Momentum=0 --cross_loss=1
done
################################################################################################################################################
#sga shufflenetv2 caltech
for TARGET_CLASS in {37,85,55,79,21,9,4,6}
do
    CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=caltech --targeted=1 --target_class=$TARGET_CLASS --model_file_name=shufflenetv2_caltech.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/caltech_shufflenetv2_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size=25 --test_batch_size=12 --minibatch=10 --alpha=10 --epoch=5 --spgd=0 --num_images=10000 --model_name=sufflenetv2 --Momentum=0 --cross_loss=1
done
################################################################################################################################################
#sga resnet50 eurosat
for TARGET_CLASS in {9,1,8,2,3,7,4,6}
do
    CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=eurosat --targeted=1 --target_class=$TARGET_CLASS --model_file_name=resnet50_eurosat.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/eurosat_resnet50_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size=250 --test_batch_size=100 --minibatch=10 --alpha=10 --epoch=5 --spgd=0 --num_images=10000 --model_name=resnet50 --Momentum=0 --cross_loss=1
done
################################################################################################################################################
#sga cifar10
for TARGET_CLASS in {0,1,2,3,5,4,6,7,8,9}
do
    CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=cifar10 --targeted=1 --target_class=$TARGET_CLASS --model_file_name=wideresnet_cifar10.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/cifar10_wideresnet_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size 250 --minibatch 10 --alpha 10 --epoch 2 --spgd 0 --num_images 10000 --model_name wideresnet --Momentum 0 --cross_loss 1
done