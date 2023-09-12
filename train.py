import dgl
import numpy as np
import os
import time
import random
import argparse
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.rna_utils import detect_motifs
from train_utils.train_RNAGraph_graph_classification import train_epoch_sparse as train_epoch, \
    evaluate_network_sparse as evaluate_network
from nets.RNA_graph_classification.load_net import gnn_model  # import all GNNS
from data.data import LoadData  # import dataset
from logger import Logger


os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
torch.use_deterministic_algorithms(True)

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


"""
    GPU Setup
"""


def seed_set(seed=None):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    dgl.random.seed(seed)


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda:"+ str(gpu_id))
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    loss record
"""


def loss_record(DATASET_NAME, print_str, epoch, config, str_time):
    if config['debias'] == "True":
        loss_dir = './loss/debias/'
    else:
        loss_dir = './loss/bias/'
    if os.path.exists(loss_dir) is False:
        os.makedirs(loss_dir)
    if os.path.exists(loss_dir + DATASET_NAME + str_time + '.txt') is True and epoch == 0:
        os.remove(loss_dir + DATASET_NAME + str_time + '.txt')
    with open(loss_dir + DATASET_NAME + str_time + '.txt', 'a') as f:
        f.write(print_str + '\n')


def save_model(DATASET_NAME, model, epoch, config):
    # save
    if config['debias'] == "True":
        save_dir = './model_save/debias/'
    else:
        save_dir = './model_save/bias/'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    if os.path.exists(save_dir + DATASET_NAME) is False:
        os.makedirs(save_dir + DATASET_NAME)

    torch.save(model.state_dict(), save_dir + DATASET_NAME +
               '/model_' + str(epoch) + '.pth')
    # load
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()


def load_model(DATASET_NAME, MODEL_NAME, net_params, config, epoch):
    if config['debias'] == "True":
        save_dir = './model_save/debias/'
    else:
        save_dir = './model_save/bias/'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    if os.path.exists(save_dir + DATASET_NAME) is False:
        os.makedirs(save_dir + DATASET_NAME)

    model = gnn_model('TransGCN', net_params)
    print(model)
    PATH = save_dir + DATASET_NAME + '/model_' + str(epoch) + '.pth'
    # model.load_state_dict(torch.load(PATH))
    # model.eval()

    return model


"""
    VIEWING MODEL CONFIG AND PARAMS
"""


def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""


def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, config):
    str_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    Log = Logger(fn="/home/lab/slh/logs/{}_{}_{}_{}.log".format(config['dataset'], str_time,
                                                                str(params['init_lr']), str(params['epochs'])))
    t0 = time.time()
    per_epoch_time = []
    best_val_auc = 0
    last_val_loss = 10
    best_epoch = 0
    early_stop_count = 0
    results_print_str = None
    print_list = []
    DATASET_NAME = dataset.name
    str_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print(
                "[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format
                (DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))
    print("Number of Classes: ", net_params['n_classes'])

    # model = gnn_model(MODEL_NAME, net_params)
    model = load_model(DATASET_NAME, MODEL_NAME, net_params,
                       config, config['best_epoch'])
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], []

    # batching exception for Diffpool
    drop_last = True
    train_loader = DataLoader(
        trainset, batch_size=params['batch_size'], shuffle=False, drop_last=True, collate_fn=dataset.collate)
    val_loader = DataLoader(
        valset, batch_size=params['batch_size'], shuffle=False, drop_last=False, collate_fn=dataset.collate)
    test_loader = DataLoader(
        testset, batch_size=params['batch_size'], shuffle=False, drop_last=False, collate_fn=dataset.collate)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, epoch_train_acc, optimizer = train_epoch(
                    model, optimizer, device, train_loader, epoch)

                epoch_val_loss, epoch_val_acc, epoch_val_auc = evaluate_network(
                    model, device, val_loader, epoch)
                _, epoch_test_acc, epoch_test_auc = evaluate_network(
                    model, device, test_loader, epoch)

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_accs.append(epoch_train_acc)
                epoch_val_accs.append(epoch_val_acc)

                writer.add_scalar('train_utils/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train_utils/_acc', epoch_train_acc, epoch)
                writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                writer.add_scalar('val/_auc', epoch_val_auc, epoch)
                writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                writer.add_scalar('test/_auc', epoch_test_auc, epoch)
                writer.add_scalar(
                    'learning_rate', optimizer.param_groups[0]['lr'], epoch)

                print_str = 'epoch ' + str(epoch) + \
                            ' train_loss ' + str(epoch_train_loss) + ' train_acc ' + str(epoch_train_acc) + \
                            ' val_loss ' + str(epoch_val_loss) + ' val_acc ' + str(epoch_val_acc) + \
                            ' val_auc ' + str(epoch_val_auc) + ' test_acc ' + str(epoch_test_acc) + \
                            ' test_auc ' + str(epoch_test_auc)
                print_list.append(print_str)

                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_acc=epoch_train_acc, val_acc=epoch_val_acc, val_auc=epoch_val_auc,
                              test_acc=epoch_test_acc, test_auc=epoch_test_auc)

                per_epoch_time.append(time.time() - start)

                scheduler.step(epoch_val_loss)

                loss_record(
                    DATASET_NAME, print_list[-1], epoch, config, str_time)

                if best_val_auc <= epoch_val_auc:
                    best_val_auc = epoch_val_auc
                    best_epoch = epoch
                    save_model(DATASET_NAME, model, epoch, config)
                    results_print_str = '\nresults:'\
                                        ' train_loss ' + str(epoch_train_loss) + ' train_acc ' + str(epoch_train_acc) + \
                                        ' val_loss ' + str(epoch_val_loss) + ' val_acc ' + str(epoch_val_acc) + \
                                        ' val_auc ' + str(epoch_val_auc) + ' test_acc ' + str(epoch_test_acc) + \
                                        ' test_auc ' + str(epoch_test_auc)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    Log.log("\n!! LR EQUAL TO MIN LR SET.")
                    break

                if last_val_loss > epoch_val_loss:
                    last_val_loss = epoch_val_loss
                    early_stop_count = 0
                else:
                    last_val_loss = epoch_val_loss
                    early_stop_count += 1
                    if early_stop_count == 10:
                        Log.log(
                            "early stop because the val loss doesn't decrease for 10 epochs")
                        break

                # Stop training after params['max_time'] hours
                if time.time() - t0 > params['max_time'] * 3600:
                    Log.log('-' * 89)
                    Log.log("Max_time for training elapsed {:.2f} hours, so stopping".format(
                        params['max_time']))
                    break

    except KeyboardInterrupt:
        Log.log('-' * 89)
        Log.log('Exiting from training early because of KeyboardInterrupt')

    if results_print_str is not None:
        loss_record(DATASET_NAME, results_print_str, epoch, config, str_time)
        # save_model(DATASET_NAME, model, epoch, config)
        # model = load_model(DATASET_NAME, MODEL_NAME,
        #                    net_params, config, config['best_epoch'])
        # model = model.to(device)
    else:
        epoch = 0
        model = load_model(DATASET_NAME, MODEL_NAME,
                           net_params, config, config['best_epoch'])
        model = model.to(device)

    if config['motif'] == "True":
        epoch = 0
        model = load_model(DATASET_NAME, MODEL_NAME,
                           net_params, config, config['best_epoch'])
        model = model.to(device)
        detect_motifs(model, train_loader, device,
                      output_dir='motifs/' + DATASET_NAME)

    _, test_acc, test_auc = evaluate_network(model, device, test_loader, epoch)
    _, train_acc, train_auc = evaluate_network(
        model, device, train_loader, epoch)
    Log.log("Test Accuracy: {:.4f}".format(test_acc))
    Log.log("Test AUC: {:.4f}".format(test_auc))
    Log.log("Train Accuracy: {:.4f}".format(train_acc))
    Log.log("Train AUC: {:.4f}".format(train_auc))
    Log.log("Convergence Time (Epochs): {:.4f}".format(epoch))
    Log.log("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    Log.log("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()


def main():
    """
        USER CONTROLS
    """
    seed_set(20001016)
    # --dataset ALKBH5_Baltz2012 --config configs/RNAgraph_graph_classification_GCN_in_vivo_100k.json --model GCN --debias False --seed 66
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='/home/lab/slh/RNASSR-Net-main/configs/RNAgraph_graph_classification_GCN_in_vivo_100k.json',
                        help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', default='TransGCN',
                        help="Please give a value for model name")
    parser.add_argument('--dataset', default='PARCLIP_ELAVL1',
                        help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', default=20001016,
                        help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument(
        '--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor',
                        help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience',
                        help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument(
        '--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval',
                        help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument(
        '--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument(
        '--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout',
                        help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument(
        '--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument(
        '--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator',
                        help="Please give a value for sage_aggregator")
    parser.add_argument(
        '--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block',
                        help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim',
                        help="Please give a value for embedding_dim")
    parser.add_argument(
        '--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument(
        '--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--debias', default='False',
                        help="Debias the data or not")
    parser.add_argument('--motif', help="get motifs or not")
    parser.add_argument('--best_epoch', help="best epoch for the result")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.debias is not None:
        config['debias'] = args.debias
    if args.motif is not None:
        config['motif'] = args.motif
    if args.best_epoch is not None:
        config['best_epoch'] = args.best_epoch
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME, config)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated == 'True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm == 'True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred == 'True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat == 'True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop == 'True' else False

    net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)
    # net_params['in_dim_edge'] = dataset.train_utils[0][0].edata['feat'][0].size(0)
    net_params['in_dim_edge'] = 1
    num_classes = len(np.unique(np.array(dataset.train[:][1])))
    net_params['n_classes'] = num_classes

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + \
        str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + \
        "_GPU" + str(config['gpu']['id']) + "_" + \
        time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + \
        "_GPU" + str(config['gpu']['id']) + "_" + \
        time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + \
        "_GPU" + str(config['gpu']['id']) + "_" + \
        time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)

    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, config)


main()
