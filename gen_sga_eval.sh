#eval for SGA
CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --uaps_save ./uaps_save/sga/sga_10000_20epoch_250batch.pth --batch_size 100 --model_name alexnet 2>&1|tee ./uaps_save/sga/sga_250batch_alexnet.log

CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --uaps_save ./uaps_save/sga/sga_10000_20epoch_250batch.pth --batch_size 100 --model_name googlenet 2>&1|tee ./uaps_save/sga/sga_250batch_googlenet.log

CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --uaps_save ./uaps_save/sga/sga_10000_20epoch_250batch.pth --batch_size 100 --model_name vgg16 2>&1|tee ./uaps_save/sga/sga_250batch_vgg16.log

CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --uaps_save ./uaps_save/sga/sga_10000_20epoch_250batch.pth --batch_size 100 --model_name vgg19 2>&1|tee ./uaps_save/sga/sga_250batch_vgg19.log

CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --uaps_save ./uaps_save/sga/sga_10000_20epoch_250batch.pth --batch_size 100 --model_name resnet152 2>&1|tee ./uaps_save/sga/sga_250batch_resnet152.log



CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --data_dir=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/test/ --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/sga_10000_10epoch_250batch.pth --batch_size 100 --model_name vgg19 2>&1|tee /root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga/sga_250batch_vgg19.log


#targeted
CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --targeted=1 --dataset=imagenet --target_class=174 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga --batch_size 100 --model_name resnet50
CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --targeted=1 --dataset=imagenet --target_class=150 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga --batch_size 100 --model_name vgg19
CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --targeted=1 --dataset=imagenet --target_class=573 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga --batch_size 100 --model_name googlenet

CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --targeted=1 --dataset=asl --target_class=19 --model_file_name=mobilenet_asl.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/asl_mobilenet_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga --batch_size 100 --model_name mobilenet
CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --targeted=1 --dataset=caltech --target_class=3 --model_file_name=shufflenetv2_caltech.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/caltech_shufflenetv2_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga --batch_size 12 --model_name shufflenetv2
CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --targeted=1 --dataset=eurosat --target_class=9 --model_file_name=resnet50_eurosat.pth --weight_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/models/eurosat_resnet50_123 --uaps_save=/root/autodl-tmp/sunbing/workspace/uap/my_result/sga/uap_save/sga --batch_size 100 --model_name resnet50
