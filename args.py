import argparse
import os
import math

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--use_fl',type=bool,default=False,
                        help='whether to use federated learning')
    parser.add_argument('--use_assignment_class',type=bool,default=False,
                        help='whether to use assignment class')
    parser.add_argument('--use_labeled',type=bool,default=False,
                        help='whether to use labeled data')
    parser.add_argument('--use_classifier',type=bool,default=False,
                        help='whether to use classifier to help training')
    parser.add_argument('--print_freq', type=int, default=5,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of training epochs')
    parser.add_argument('--device',type=str,default='cuda',
                        help='the device for training')
    
    # fedml_api/utils/add_args.py
    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)
    parser.add_argument('--model_save_path', type=str, default='../../../fedml_api/model/pretrained/',
                        help='model save path')
    parser.add_argument('--resource_constrained',action='store_true',
                    help='if clients are resource constrained', default=False)
    # fedml_api/utils/add_args.py (FL settings)
    parser.add_argument('--client_num_in_total', type=int, default=6, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--join_ratio', type=float, default=0.1,
                        help='Ratio for (client each round) / (client num in total)')
    parser.add_argument('--comm_round', type=int, default=50,
                        help='how many round of communications we shoud use')
    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')
    parser.add_argument('--agg_method', type=int, default=0, metavar='N',
                        help='how to aggregate pruned parames, 0:HeteroFL, 1:HeteroFL datasize weighted, 2:HeteroFL datasize and loss weighted, 3:HeteroFL distributed, 4:HeteroFL grouped')
    parser.add_argument('--similarity_method', type=int, default=0, metavar='S',
                        help='how to compute the data distribution distance, 0:cosine_similarity, 1:l1 norm, 2:l2 norm')
    parser.add_argument('--group_num', type=int, default=2,
                        help='group number (used for agg_method=4)')
    # fedml_api/utils/add_args.py (data partition settings)
    parser.add_argument('--partition_method', type=int, default=0, metavar='P',
                        help='how to partition the dataset on local workers, 0:iid, 1:shard noniid, 2:Dirichlet noniid, 3:k class log-normal, -1:per user')

    parser.add_argument('--partition_alpha', type=float, default=0.9, metavar='PA',
                        help='partition alpha (default: 0.9)')
    
    parser.add_argument('--datasize_per_client', type=int, default=-1,
                        help='the number of data per client (default: Divide the entire data set evenly)')
    
    parser.add_argument('--num_shards_per_user', type=int, default=2,
                        help='used for partition_method=1, the number of shards allocated to each client')
    
    parser.add_argument('--num_classes_per_user', type=int, default=2,
                        help='used for partition_method=3, the number of classes allocated to each client')
    
    parser.add_argument('--sample_num_per_shard', type=int, default=30,
                        help='the number of samples of per shard')
    
    parser.add_argument('--global_dataset_selected_ratio', type=float, default=-1,
                        help='selected a part of global dataset for training and inference')
    parser.add_argument("--max_shards_num", type=int)
    parser.add_argument("--min_shards_num", type=int)


    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,200,300',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='MyUTDmodel')
    parser.add_argument('--data_dir', type=str, default='/home/dengyh/dataset/cifar10',
                        help='data directory')
    parser.add_argument('--num_class', type=int, default=27,
                        help='num_class')
    parser.add_argument('--num_train_basic', type=int, default=1,
                        help='num_train_basic')
    parser.add_argument('--num_train_unlabel_basic', type=int, default=1,
                        help='num_train_unlabel_basic')
    parser.add_argument('--label_rate', type=int, default=5,
                        help='label_rate')

    # method
    parser.add_argument('--method', type=str, default='FML',
                        choices=['FML'], help='choose method')
    parser.add_argument('--num_positive', type=int, default=9,
                        help='num_positive')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=int, default='3',
                        help='id for recording multiple runs')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')
    
    # simulated calculation
    parser.add_argument('--project_name', default='AnalogAI', type=str, help='name')
    parser.add_argument('--architecture', default='vit', type=str, help='name')
    parser.add_argument('--dataset', default='cifar10', type=str, help='training dataset')
    parser.add_argument('--batch-size', default=512, type=int, help='mini-batch size')
    parser.add_argument('-sram_analog_recover',default=False, type=bool, help='whether to recover on an analog platform')
    parser.add_argument('-analog_infer',default=False, type=bool, help='whether to infer on an analog platform')
    parser.add_argument('--config', type=str, default='client_5_noise_0-0.1_T_T.yml', help='Path to the config file')
    parser.add_argument('--use_momentum', type=bool, default=False, help='whether to use momentum when update client model')
    parser.add_argument('--use_foundation_model', type=bool, default=False, help='whether to use foundation model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.model_path = './save/FML/{}_models'.format(opt.dataset)
    opt.tb_path = './save/FML/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_label_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_epoch_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.label_rate, opt.learning_rate,
               opt.lr_decay_rate, opt.batch_size, opt.temp, opt.trial, opt.epochs)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt