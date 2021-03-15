import argparse
import pprint

import torch
from torch import optim
import torch.nn as nn
import torch_optimizer as custom_optim

from Implement_paper.gpt_1.data_loader import Dataloader_Pre_train
from Implement_paper.gpt_1.gpt_1 import GPT_pretrain
from Implement_paper.gpt_1.pre_trainer import MaximumLikelihoodEstimationEngine
import Implement_paper.gpt_1.pre_trainer as Trainer


def define_argparser(is_continue = False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',
            required=True,
            help='Model file name to continue.'
        )
    
    p.add_argument(
        '--model_fn',
        required=not is_continue,
        help='Model file name to save. Additional information would be annotated to the file name.'
    )

    p.add_argument(
        '--train',
        required=not is_continue,
        help='Training set file name except the extention. (ex: train.en -> train)'
    )

    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s'
    )
    p.add_argument(
        '--off_autocast',
        action='store_true',
        help='Turn-off Automatic Mixed Precision (AMP), which speed-up training.',
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=20,
        help='Number of epochs to train. Default=%(default)s'
    )
    p.add_argument(
        '--verbose',
        type=int,
        default=2,
        help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s'
    )
    p.add_argument(
        '--init_epoch',
        required=is_continue,
        type=int,
        default=1,
        help='Set initial epoch number, which can be useful in continue training. Default=%(default)s'
    )

    p.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Maximum length of the training sequence. Default=%(default)s'
    )
    p.add_argument(
        '--dropout',
        type=float,
        default=.2,
        help='Dropout rate. Default=%(default)s'
    )
    p.add_argument(
        '--word_vec_size',
        type=int,
        default=512,
        help='Word embedding vector dimension. Default=%(default)s'
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='Hidden size of LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--n_layers',
        type=int,
        default=4,
        help='Number of layers in LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--n_splits',
        type=int,
        default=8,
        help='Number of heads in multi-head attention in Transformer. Default=%(default)s',
    )

    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=5.,
        help='Threshold for gradient clipping. Default=%(default)s'
    )
    p.add_argument(
        '--iteration_per_update',
        type=int,
        default=1,
        help='Number of feed-forward iterations for one parameter update. Default=%(default)s'
    )

    p.add_argument(
        '--lr',
        type=float,
        default=1.,
        help='Initial learning rate. Default=%(default)s',
    )

    p.add_argument(
        '--lr_step',
        type=int,
        default=1,
        help='Number of epochs for each learning rate decay. Default=%(default)s',
    )
    p.add_argument(
        '--lr_gamma',
        type=float,
        default=.5,
        help='Learning rate decay rate. Default=%(default)s',
    )
    p.add_argument(
        '--lr_decay_start',
        type=int,
        default=10,
        help='Learning rate decay start at. Default=%(default)s',
    )

    config = p.parse_args()

    return config


def get_model(input, config):
    model = GPT_pretrain(
        input,
        config.vocab_size,
        config.n_layer,
        config.hidden_size,
        config.n_splits
    )

    return model

def get_crit(output_size, pad_index):
    # PAD를 zero로 set한다.(PAD는 weight를 줄 필요가 없다.)
    loss_weight = torch.ones(output_size)
    loss_weight[pad_index] = 0.
    # log-probability와 함께 Cross-Entropy loss 대신 NLL loss를 사용한다.
    crit = nn.NLLoss(
        weight = loss_weight,
        reduction = 'sum'
    )

    return crit

def get_optimizer(model, config):
    optimizer = optim.Adam(model.parameters(), lr = config.lr, betas=(.9, .98))

    return optimizer

def get_scheduler(optimizer, config):
    if config.lr_step > 0:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            # 함수 알아보기
            optimizer,
            milestones=[i for i in range(
                max(0, config.lr_decay_start - 1),
                (config.init_epoch -1) + config.n_epochs,
                config.lr_step
            )],
            gamma = config.lr_gamma,
            last_epoch=config.init_epoch - 1 if config.init_epoch > 1 else -1,
        )
    else:
        lr_scheduler = None

    return lr_scheduler

def main(config, main_weight = None, opt_weight = None):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    loader = DataLoader(
        config.train,
        batch_size = config.batch_size,
        device = -1,
        max_length = config.max_length,
    )

    input_size, output_size = len(loader.src.vocab)
    model = get_model(input, config)
