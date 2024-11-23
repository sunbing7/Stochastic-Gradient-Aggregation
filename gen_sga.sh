#UDA_VISIBLE_DEVICES='0' python imagenet_attack.py --data_dir /home/pc/lxn/data/imagenet/train/ --uaps_save ./uaps_save/sga/ --batch_size 250 --minibatch 10 --alpha 10 --epoch 20 --spgd 0 --num_images 10000 --model_name vgg16 --Momentum 0 --cross_loss 1
#CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --data_dir=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/val/ --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size 250 --minibatch 10 --alpha 10 --epoch 20 --spgd 0 --num_images 10000 --model_name vgg19 --Momentum 0 --cross_loss 1
#CUDA_VISIBLE_DEVICES='0' python imagenet_attack_ori.py --data_dir=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/val/ --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size 250 --minibatch 10 --alpha 10 --epoch 1 --spgd 0 --num_images 10000 --model_name resnet50 --Momentum 0 --cross_loss 1

#targeted deprecated
#CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=imagenet --targeted=1 --target_class=150 --data_dir=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/val/ --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size 250 --minibatch 10 --alpha 10 --epoch 10 --spgd 0 --num_images 10000 --model_name vgg19 --Momentum 0 --cross_loss 1
#CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=imagenet --targeted=1 --target_class=174 --data_dir=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/val/ --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size 250 --minibatch 10 --alpha 10 --epoch 1 --spgd 0 --num_images 10000 --model_name resnet50 --Momentum 0 --cross_loss 1
#CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=imagenet --targeted=1 --target_class=573 --data_dir=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/val/ --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size 250 --minibatch 10 --alpha 10 --epoch 1 --spgd 0 --num_images 10000 --model_name googlenet --Momentum 0 --cross_loss 1

#targeted
#CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=imagenet --targeted=1 --target_class=174 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size 250 --test_batch_size=100 --minibatch 10 --alpha 10 --epoch 10 --spgd 0 --num_images 10000 --model_name resnet50 --Momentum 0 --cross_loss 1
CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=imagenet --targeted=1 --target_class=214 --model_file_name=vgg19_imagenet.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/imagenet_vgg19_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size=250 --test_batch_size=100 --minibatch=10 --alpha=10 --epoch=10 --spgd=0 --num_images=10000 --model_name=vgg19 --Momentum=0 --cross_loss=1
CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=imagenet --targeted=1 --target_class=573 --model_file_name=googlenet_imagenet.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/imagenet_googlenet_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size=250 --test_batch_size=100 --minibatch=10 --alpha=10 --epoch=10 --spgd=0 --num_images=10000 --model_name=googlenet --Momentum=0 --cross_loss=1
CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=asl --targeted=1 --target_class=19 --model_file_name=mobilenet_asl.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/asl_mobilenet_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size=250 --test_batch_size=100 --minibatch=10 --alpha=10 --epoch=5 --spgd=0 --num_images=10000 --model_name=mobilenet --Momentum=0 --cross_loss=1
CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=caltech --targeted=1 --target_class=37 --model_file_name=shufflenetv2_caltech.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/caltech_shufflenetv2_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size=25 --test_batch_size=12 --minibatch=10 --alpha=10 --epoch=5 --spgd=0 --num_images=10000 --model_name=shufflenetv2 --Momentum=0 --cross_loss=1
CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=eurosat --targeted=1 --target_class=9 --model_file_name=resnet50_eurosat.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/eurosat_resnet50_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size=250 --test_batch_size=100 --minibatch=10 --alpha=10 --epoch=5 --spgd=0 --num_images=10000 --model_name=resnet50 --Momentum=0 --cross_loss=1
CUDA_VISIBLE_DEVICES='0' python imagenet_attack.py --dataset=cifar10 --targeted=1 --target_class=0 --model_file_name=wideresnet_cifar10.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/cifar10_wideresnet_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/ --batch_size 250 --test_batch_size=100 --minibatch 10 --alpha 10 --epoch 2 --spgd 0 --num_images 10000 --model_name wideresnet --Momentum 0 --cross_loss 1

#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=sga --model_name=resnet50_imagenet_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=174 --ngpu=1 --workers=4
