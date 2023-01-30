# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:00:54 2023

@author: BD
"""

import argparse
import numpy as np
import torch
import os, sys
import math
import matplotlib.pyplot as plt
from subspace_inference import models, losses, posteriors, utils
# from swag.posteriors import SWAG, EllipticalSliceSampling, BenchmarkPyro, BenchmarkVIModel
from regression import run
from data import get_regression_data
from models import RegressionRunner, ESSRegRunner, VIRegRunner

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='RegNet', nargs='?', type=str)
parser.add_argument("--dataset", default='boston', nargs='?', type=str)
parser.add_argument("--split", default=0, nargs='?', type=int)
parser.add_argument("--seed", default=0, nargs='?', type=int)
parser.add_argument('--database_path', default='', help='output database')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.01, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--batch_size', type=int, default=400, metavar='N', help='input batch size (default: 128)')

parser.add_argument('--swag', action='store_true')
parser.add_argument('--swag_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swag_lr', type=float, default=0.02, metavar='LR', help='SWA LR (default: 0.02)')
parser.add_argument('--swag_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
parser.add_argument('--subspace', type=str, choices=['covariance', 'pca','freq_dir','random'])
parser.add_argument('--max_num_models', type=int, default=20, help='maximum number of SWAG models to save')
parser.add_argument('--num_samples', type=int, default=30, help='number of monte carlo samples to draw')
parser.add_argument('--scale', type=float, default=0.5, help='scale for SWAG+ samples')

parser.add_argument('--swag_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to restor SWA from (default: None)')
parser.add_argument('--model_variance', action='store_true', help='whether NN should also model variance')
parser.add_argument('--noise_var', action='store_true', help='whether NN should have a noise variance term')

parser.add_argument('--no_schedule', action='store_true', help='store schedule')

parser.add_argument('--save_iterates', action='store_true', help='save all iterates in the SWA(G) stage (default: off)')
parser.add_argument('--inference', choices=['low_rank_gaussian', 'projected_sgd', 'ess', 'nuts', 'vi'], default='low_rank_gaussian')
parser.add_argument('--prior_std', type=float, default=1.0, help='std of the prior distribution')

parser.add_argument('--temperature', type=float, default=None, help='temperature of posterior')

parser.add_argument('--double-bias-lr', action='store_true')

args = parser.parse_args()

#%% Inference Type
args.inference = 'low_rank_gaussian' # SWA-G

data_name = ["enb"]
# data_name = ["winered","kin8nm","power","protein"]
for j in range(len(data_name)):
    args.dataset  = data_name[j]
    _RESULTS_lltest = "data/"+args.dataset+"/results_SI_SGD/lltest_SI_SGD.txt"
    _RESULTS_RMSEtest ="data/"+args.dataset+"/results_SI_SGD/RMSEtest_SI_SGD.txt"
    #%%
    
    
    if args.dataset == 'enb':
        args.epochs  = 1000
        args.batch_size = 100
        args.lr_init = 1e-03
        # args.noise_var = True
        args.model_variance = True
        args.no_schedule = True
        args.wd = 1e-2
        args.swag = True # SWAG or not
        args.swag_lr = 1e-3
        args.subspace = 'pca'
        args.swag_start = 500
       
        # args.temperature = 10000
        args.database_path = 'results_small.db'
    
    #%%
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args.device = None
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    
    
    
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    print('Preparing dataset %s' % args.dataset)
    dataset = get_regression_data(args.dataset, split=args.split)
    print(dataset.N, dataset.D, dataset.name)
    print('Using model %s' % args.model)
    model_cfg = getattr(models, args.model)
    
    print('No of samples %s' % args.num_samples)
    print('No of max models %s' % args.max_num_models)
    
    print('Preparing model')
    print(*model_cfg.args)
    
    model_cfg.kwargs['dimensions'] = [50]
    
    if args.batch_size is None:
        args.batch_size = dataset.N // 10
    print('Using batch size', args.batch_size)
    
    if args.epochs == 0:
        args.epochs = int(np.ceil(6000 * args.batch_size / dataset.N))
        print('Number of epochs is: ', args.epochs)
    
    print(model_cfg.kwargs)
    
    if args.model_variance:
        print('Model has heteroscedastic regression')
        output_dim=4 # To be changed according to output data size
        noise_var = None
    else:
        output_dim = 1
        noise_var = 0.5
    
    #todo: incorporate into command line args
    criterion = losses.GaussianLikelihood
    
    regclass = RegressionRunner
    extra_args = {}
    if args.inference == 'ess':
        regclass = ESSRegRunner
        extra_args = {'temperature': args.temperature}
    
    if args.inference == 'vi':
        regclass = VIRegRunner
        extra_args = {'prior_log_sigma':math.log(args.prior_std), 'temperature':args.temperature}
    
    
    
    
    
    n_splits = 20
    # Change only for Protein
    if data_name[j] == 'protein':
        model_cfg.kwargs['dimensions'] = [100]
        n_splits  = 5
    else:
        model_cfg.kwargs['dimensions'] = [50]
        n_splits  = 20
    
    ll_splits   = []
    rmse_splits = []
    for i in range(n_splits):
        #%% define a regressionrunner class to fit w/in confines of regression.py
        regression_model = regclass(
            base = model_cfg.base,
            inference = args.inference,
            epochs = args.epochs,
            criterion = criterion,
            batch_size=args.batch_size,
            subspace_type=args.subspace, subspace_kwargs={'max_rank':args.max_num_models},
            momentum = args.momentum, wd=args.wd, lr_init=args.lr_init,
            swag_lr = args.swag_lr, swag_freq = 1, swag_start = args.swag_start,
            use_cuda = torch.cuda.is_available(), use_swag = args.swag,
            scale=args.scale, num_samples=args.num_samples,
            const_lr=args.no_schedule, double_bias_lr=args.double_bias_lr,
            model_variance=args.model_variance,
            input_dim=dataset.D, output_dim=output_dim, apply_var=args.noise_var, 
            **model_cfg.kwargs, **extra_args
        )
        #%%
        mname = args.model
        if args.swag:
            mname = mname + args.subspace + args.inference
        bb_args = argparse.Namespace(model=mname, dataset=args.dataset, split=args.split, seed=args.seed, database_path=args.database_path)
        bb_result, ll_epochs, rmse_epochs = run(bb_args, data=dataset, model=regression_model, is_test=args.database_path=='')
        ll_splits.append(ll_epochs)
        rmse_splits.append(rmse_epochs)
    
    mean_ll   = np.mean(ll_splits,axis=0)
    mean_RMSE = np.mean(rmse_splits,axis=0)
    plt.scatter(range(args.epochs), mean_ll)
    plt.show()
    plt.scatter(range(args.epochs), mean_RMSE)
    plt.show()
    
    with open(_RESULTS_lltest, "w") as myfile:
            for item in mean_ll:
                    myfile.write('%f\n' % item)
    with open(_RESULTS_RMSEtest, "w") as myfile:
            for item in mean_RMSE:
                myfile.write('%f\n' % item)
