"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch import optim
import utils
from torch.distributions.multivariate_normal import MultivariateNormal

from torch.utils.data import ConcatDataset, DataLoader
import torch.nn.functional as F

from torch.func import functional_call, vmap, grad
import pandas as pd

from einops import rearrange
import copy

from collections import OrderedDict
import random

def test_function(data_loader, device: torch.device, epoch: int, args = None):
    
    for task_id in range(0, 10):
    
        task_data_loader = get_task_loader(data_loader, task_id, 'train')

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'
        
        if args.distributed and utils.get_world_size() > 1:
            task_data_loader.sampler.set_epoch(epoch)
        
        print("LEN DATALOADER: ", len(task_data_loader))
        
        for t, loader in enumerate(task_data_loader):
                   
            for input, target in metric_logger.log_every(loader, args.print_freq, header):
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
            
                print("TARGET OF TASK: ", t)
                print(target)

#DBP train one epoch current task only
def train_one_epoch_current_task(model: torch.nn.Module, original_model_list: list, #torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    model.train(set_training_mode)

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'

    #DBP in each epoch, we need to go through each of the tasks separately
    for t, loader in enumerate(data_loader):
        
        if t == task_id:
        
            original_model_list[t].eval()
            
            for input, target in metric_logger.log_every(loader, args.print_freq, header):
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                with torch.no_grad():
                    if original_model_list is not None:
                        output = original_model_list[t](input)
                        logits = output[0]['logits']

                        if args.train_mask and class_mask is not None:
                            mask = []
                            for id in range(t + 1):
                                mask.extend(class_mask[id])
                            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                            prompt_id = torch.max(logits, dim=1)[1]
                            # translate cls to task_id
                            prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(
                                -1)
                        else:
                            prompt_id = None
                    else:
                        raise NotImplementedError("original model is None")
                        
                output = model(input, task_id=t, prompt_id=prompt_id, train=set_training_mode,
                            prompt_momentum=args.prompt_momentum)
                logits = output[0]['logits']
                
                # here is the trick to mask out classes of non-current tasks
                if args.train_mask and class_mask is not None:
                    mask = class_mask[t]
                    not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                #DBP do not reduce
                criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
                
                loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)
                # TODO add contrastive loss
                # loss += orth_loss(output[0]['pre_logits'], target, device, args)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))

                #if not math.isfinite(loss.item()):
                    #print("Loss is {}, stopping training".format(loss.item()))
                    #sys.exit(1)
                
                # Detaching the parameters because we won't be calling Tensor.backward().
                params = {k: v.detach() for k, v in model.module.named_parameters() if v.requires_grad}
                
                
                #DBP define loss for a single sample (no indexing!)
                def compute_loss(params, sample, target, t, prompt_id, set_training_mode, prompt_momentum):
                    batch = sample.unsqueeze(0)
                    targets = target.unsqueeze(0)

                    predictions = functional_call(model.module, (params), args = (batch, ), kwargs = {"task_id": t, "prompt_id": prompt_id, "train": set_training_mode, "prompt_momentum": prompt_momentum})
                    loss = criterion(predictions[0]['logits'], targets)
                    return loss.mean()
                
                ft_compute_grad = grad(compute_loss)
                
                ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0, None, None, None, None))
    
                ft_per_sample_grads = ft_compute_sample_grad(params, input, target, t, prompt_id, set_training_mode, args.prompt_momentum)
                
                #print(ft_per_sample_grads.keys())
                #DBP task grads per example, and normalize
                task_grads_perexample[task_id] = ft_per_sample_grads['e_prompt.prompt'] #l2_normalize(ft_per_sample_grads['e_prompt.prompt'])
                task_labels_perexample[task_id] = target
                                
                #compute task grads ntk means
                #compute_ntk_means(task_id, args.kappa_low, args.kappa_high)
                           
                #DBP here we need to incorporate the proposed gradient alignment process between current task and each previous one
                optimizer.zero_grad()
                model.module.e_prompt.prompt.grad = ft_per_sample_grads['e_prompt.prompt'].mean(dim=0) #l2_normalize(ft_per_sample_grads['e_prompt.prompt']).mean(dim=0)
                model.module.head.weight.grad = ft_per_sample_grads['head.weight'].mean(dim=0)
                model.module.head.bias.grad = ft_per_sample_grads['head.bias'].mean(dim=0)
                #loss.backward()
    
                #DBP store task grads (before rescaling with max_norm)
                #if model.module.e_prompt.prompt.grad is not None:
                    #task_grads_keys[t] = model.module.e_prompt.prompt.grad[:, 0, ...].clone() #keys of all tasks
                    #task_grads_values[t] = model.module.e_prompt.prompt.grad[:, 1, ...].clone() #values of all tasks
                    #task_grads[t] = model.module.e_prompt.prompt.grad.clone()
            
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                
                #DBP mean loss
                loss = loss.mean()
                
                #DBP alternatively orthogonal loss
                #loss += orth_loss(output[0]['pre_logits'], target, device, args)

                torch.cuda.synchronize()
                metric_logger.update(Loss=loss.item())
                metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
                
                #DBP restart grads to before update for next iteration 
                ft_per_sample_grads = ft_compute_sample_grad(params, input, target, task_id, prompt_id, set_training_mode, args.prompt_momentum)
                task_grads_perexample[task_id] = ft_per_sample_grads['e_prompt.prompt']                

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
     
    
#DBP train one epoch current task only
#DBP added model_grpo as parameter
def train_one_epoch_previous_task(model: torch.nn.Module, original_model_list: list, #torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    model.train(set_training_mode)

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'

    #DBP in each epoch, we need to go through each of the tasks separately
    for t, loader in enumerate(data_loader):
        
        if t < task_id:
        
            original_model_list[t].eval()
            
            for input, target in metric_logger.log_every(loader, args.print_freq, header):
                #Sampling for replay
                replay_size = int(input.size(0) * args.replay_percentage)
                
                indices = torch.randperm(input.size(0), device=input.device)[:replay_size]
                
                input = input[indices].to(device, non_blocking=True)
                target = target[indices].to(device, non_blocking=True)
                
                with torch.no_grad():
                    if original_model_list is not None:
                        output = original_model_list[t](input)
                        logits = output[0]['logits']

                        if args.train_mask and class_mask is not None:
                            mask = []
                            for id in range(t + 1):
                                mask.extend(class_mask[id])
                            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                            prompt_id = torch.max(logits, dim=1)[1]
                            # translate cls to task_id
                            prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(
                                -1)
                        else:
                            prompt_id = None
                    else:
                        raise NotImplementedError("original model is None")
                        
                output = model(input, task_id=t, prompt_id=prompt_id, train=set_training_mode,
                            prompt_momentum=args.prompt_momentum)
                logits = output[0]['logits']
                
                # here is the trick to mask out classes of non-current tasks
                if args.train_mask and class_mask is not None:
                    mask = class_mask[t]
                    not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                #DBP do not reduce
                criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
                
                loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)
                # TODO add contrastive loss
                # loss += orth_loss(output[0]['pre_logits'], target, device, args)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))

                #if not math.isfinite(loss.item()):
                    #print("Loss is {}, stopping training".format(loss.item()))
                    #sys.exit(1)
                
                # Detaching the parameters because we won't be calling Tensor.backward().
                params = {k: v.detach() for k, v in model.module.named_parameters() if v.requires_grad}

                #DBP define loss for a single sample (no indexing!)
                def compute_loss(params, sample, target, t, prompt_id, set_training_mode, prompt_momentum):
                    batch = sample.unsqueeze(0)
                    targets = target.unsqueeze(0)

                    predictions = functional_call(model.module, (params), args = (batch, ), kwargs = {"task_id": t, "prompt_id": prompt_id, "train": set_training_mode, "prompt_momentum": prompt_momentum})
                    loss = criterion(predictions[0]['logits'], targets)
                    return loss.mean()
                
                ft_compute_grad = grad(compute_loss)
                
                ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0, None, None, None, None))
    
                ft_per_sample_grads = ft_compute_sample_grad(params, input, target, t, prompt_id, set_training_mode, args.prompt_momentum)
                
                #print(ft_per_sample_grads.keys())
                #DBP task grads per example
                task_grads_perexample[t] = ft_per_sample_grads['e_prompt.prompt'] #l2_normalize(ft_per_sample_grads['e_prompt.prompt'])
                task_labels_perexample[t] = target

                #re-compute task grads ntk means
                #compute_ntk_means(t, args.kappa_low, args.kappa_high) 
                
                #create a copy of the model, from the state dict, by first cleaning it
                #checkpoint_model = {k: v.cpu() for k, v in model.state_dict().items()}
                               
                #create a checkpoint of the original optimizer
                #optimizer_checkpoint = copy.deepcopy(optimizer.state_dict()) 
                
                #make a copy to restart after the transfer process 
                
                #task_grads_perexample[t] = compute_ntk(t, task_id, [input, target, t, prompt_id, set_training_mode, args, class_mask], model, optimizer, checkpoint_model, optimizer_checkpoint, max_norm, args) #perform transfer if appropriate

                #return to actual model
                #model.load_state_dict(checkpoint_model, strict=False)
                #optimizer.load_state_dict(optimizer_checkpoint)
                #torch.cuda.empty_cache()
                
                #DBP here we need to incorporate the proposed gradient alignment process between current task and each previous one
                #restart optimizer first
                optimizer.zero_grad()
                model.module.e_prompt.prompt.grad = ft_per_sample_grads['e_prompt.prompt'].mean(dim=0) #l2_normalize(task_grads_perexample[t]).mean(dim=0)
                model.module.head.weight.grad = ft_per_sample_grads['head.weight'].mean(dim=0)
                model.module.head.bias.grad = ft_per_sample_grads['head.bias'].mean(dim=0)
                #loss.backward()
            
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                
                #DBP mean loss
                loss = loss.mean()
                
                #DBP alternatively orthogonal loss
                #loss += orth_loss(output[0]['pre_logits'], target, device, args)
                
                torch.cuda.synchronize()
                metric_logger.update(Loss=loss.item())
                metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
                
                #DBP restart grads to before update for next iteration 
                ft_per_sample_grads = ft_compute_sample_grad(params, input, target, t, prompt_id, set_training_mode, args.prompt_momentum)
                task_grads_perexample[t] = ft_per_sample_grads['e_prompt.prompt']
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}    

#DBP train one epoch current task only
def train_forward_current_task(model: torch.nn.Module, original_model_list: list, #torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    model.train(set_training_mode)

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)
        
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'        

    for t, loader in enumerate(data_loader):
        
        if t == task_id and task_id > 0:
        
            original_model_list[t].eval()
            
            for input, target in metric_logger.log_every(loader, args.print_freq, header):
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                with torch.no_grad():
                    if original_model_list is not None:
                        output = original_model_list[t](input)
                        logits = output[0]['logits']

                        if args.train_mask and class_mask is not None:
                            mask = []
                            for id in range(t + 1):
                                mask.extend(class_mask[id])
                            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                            prompt_id = torch.max(logits, dim=1)[1]
                            # translate cls to task_id
                            prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(
                                -1)
                        else:
                            prompt_id = None
                    else:
                        raise NotImplementedError("original model is None")
                        
                output = model(input, task_id=t, prompt_id=prompt_id, train=set_training_mode,
                            prompt_momentum=args.prompt_momentum)
                logits = output[0]['logits']
                
                # here is the trick to mask out classes of non-current tasks
                if args.train_mask and class_mask is not None:
                    mask = class_mask[t]
                    not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                    
                #DBP do not reduce
                criterion = torch.nn.CrossEntropyLoss(reduction = 'mean')
                
                loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)
                # TODO add contrastive loss
                # loss += orth_loss(output[0]['pre_logits'], target, device, args)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))  
                
                # Detaching the parameters because we won't be calling Tensor.backward().
                params = {k: v.detach() for k, v in model.module.named_parameters() if v.requires_grad}

                #DBP define loss for a single sample (no indexing!)
                def compute_loss(params, sample, target, t, prompt_id, set_training_mode, prompt_momentum):
                    batch = sample.unsqueeze(0)
                    targets = target.unsqueeze(0)

                    predictions = functional_call(model.module, (params), args = (batch, ), kwargs = {"task_id": t, "prompt_id": prompt_id, "train": set_training_mode, "prompt_momentum": prompt_momentum})
                    loss = criterion(predictions[0]['logits'], targets)
                    return loss.mean()
                
                ft_compute_grad = grad(compute_loss)
                
                ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0, None, None, None, None))
    
                ft_per_sample_grads = ft_compute_sample_grad(params, input, target, t, prompt_id, set_training_mode, args.prompt_momentum)
                
                #print(ft_per_sample_grads.keys())
                #DBP task grads per example
                task_grads_perexample[task_id] = ft_per_sample_grads['e_prompt.prompt'] #l2_normalize(ft_per_sample_grads['e_prompt.prompt'])
                task_labels_perexample[task_id] = target                

                task_grads_perexample[task_id] = compute_ntk_forward(task_id-1, task_id, [input, target, t, prompt_id, set_training_mode, args, class_mask], model, optimizer, args)#, checkpoint_model, optimizer_checkpoint, max_norm, args)
                           
                #DBP Just make the step!
                optimizer.zero_grad()
                model.module.e_prompt.prompt.grad = task_grads_perexample[task_id].mean(dim=0) #l2_normalize(ft_per_sample_grads['e_prompt.prompt']).mean(dim=0)
                #loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                
                torch.cuda.synchronize()
                metric_logger.update(Loss=loss.item())
                metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            return {k: meter.global_avg for k, meter in metric_logger.meters.items()}                 
                

def l2_normalize(g, dim = -1, eps=1e-8):
    return F.normalize(g, p=2, dim=-1)
    
def get_task_subspace_overlap(task_prev_norm, task_curr_norm, l = 0, k=10):
    """
    G_a, G_b: Tensors of shape [num_examples, embedding_dim]
    k: Number of principal components to consider for the subspace
    """
    
    # 1. Make tensors
    #task_prev_norm = task_prev_norm.to(torch.float32)
    #task_prev_curr = task_prev_curr.to(torch.float32)
    
    # 2. Compute SVD for both tasks to get the Orthonormal Bases (V)
    # We use the right singular vectors (Vh) which represent the embedding directions
    _, _, Vh_a = torch.linalg.svd(task_prev_norm, full_matrices=False)
    _, _, Vh_b = torch.linalg.svd(task_curr_norm, full_matrices=False)
    
    # 3. Extract the top-k components (The "Subspaces")
    # Vh is [embedding_dim, embedding_dim], we take the first k rows
    basis_a = Vh_a[:k, :] # Shape: [k, embedding_dim]
    basis_b = Vh_b[:k, :] # Shape: [k, embedding_dim]
    
    # 4. Compute the Principal Angles (Singular values of the bases product)
    # This measures how aligned the two k-dimensional slices are
    # M = A * B.T
    M = torch.matmul(basis_a, basis_b.T)
    
    # The singular values of M are the cosines of the principal angles
    cos_theta = torch.linalg.svdvals(M)
    
    # 5. Scalar metric: Mean Overlap (1.0 = identical, 0.0 = orthogonal)
    overlap_score = torch.mean(cos_theta)
    
    #print("Overlap score: ", overlap_score, " Cost theta: ", cos_theta)
    return overlap_score, cos_theta


def compute_normalized_task_alignment(grads_A, grads_B):
    """
    Calculates the normalized Empirical NTK alignment between Task A and Task B.
    Result is bounded (effectively a Cosine Similarity for Task Manifolds).
    """
    # 1. Compute the Raw Cross-Kernel (Numerator)
    # K(A, B) = mean(G_A @ G_B.T)
    k_ab = torch.matmul(grads_A, grads_B.t()).mean()

    # 2. Compute the Self-Kernels (Denominator)
    # K(A, A) and K(B, B)
    k_aa = torch.matmul(grads_A, grads_A.t()).mean()
    k_bb = torch.matmul(grads_B, grads_B.t()).mean()

    # 3. Normalize: k_ab / sqrt(k_aa * k_bb)
    # This is the formal "Kernel Alignment" score
    normalized_alignment = k_ab / torch.sqrt(k_aa * k_bb)

    return normalized_alignment

#task similarity before example-based transfer
#cross-task NTK similarity, or cosine similarity
#l = layer, key_val = {0: key, 1: value}
#do not need normalizing gradients because cosine similarity does that already
def task_similarity(task_prev: int, task_curr: int, l = 0, key_val = 0, eps = 1e-8):
    
    flat_g_prev = rearrange(task_grads_perexample[task_prev][:, l, key_val, task_prev, :, :, :], 'ex len h dh -> ex (len h dh)')
    flat_g_curr = rearrange(task_grads_perexample[task_curr][:, l, key_val, task_curr, :, :, :], 'ex len h dh -> ex (len h dh)')
   
    g_norm_prev = normalize(flat_g_prev, p=2, dim=-1)
    g_norm_curr = normalize(flat_g_curr, p=2, dim=-1)
    
    #old similarity
    #overlap_score, cos_theta = get_task_subspace_overlap(g_norm_prev, g_norm_curr)
    #return overlap_score
    
    #new similarity - NTK alignment
    alignment = compute_normalized_task_alignment(g_norm_prev, g_norm_curr)
    print("ALIGNMENT: ", alignment)
    return alignment        
     
            
def normalize(a, p=2, dim=-1):
    return F.normalize(a, p=2, dim=-1)
    
def compute_ntk(task_prev: int, task_curr: int, input_target: [], model_grpo: torch.nn.Module, optimizer_grpo: torch.optim.Optimizer, checkpoint_model, optimizer_checkpoint, max_norm: float, args=None, eps = 1e-8):
    
    #only do the first layer
    for l in range(0, 1): #5 layers
        
        #examples for previous task
        new_aligned_examples_dict = {idx: [] for idx in range(0, len(task_grads_perexample[task_prev]))}
        old_aligned_examples_dict = {idx: [] for idx in range(0, len(task_grads_perexample[task_prev]))}
  
        template_new_examples = torch.zeros_like(task_grads_perexample[task_prev][0:1])
        template_old_examples = torch.zeros_like(task_grads_perexample[task_prev][0:1])
        
        #examples for current task
        new_aligned_examples_curr_dict = {idx: [] for idx in range(0, len(task_grads_perexample[task_prev]))}
        old_aligned_examples_curr_dict = {idx: [] for idx in range(0, len(task_grads_perexample[task_prev]))}
  
        template_new_examples_curr = torch.zeros_like(task_grads_perexample[task_prev][0:1])
        template_old_examples_curr = torch.zeros_like(task_grads_perexample[task_prev][0:1])
        
        
        task_prev_examples_layer_key = rearrange(task_grads_perexample[task_prev][:, l, 0, task_prev, :, :, :], 'ex len h dh -> ex (len h dh)')
        task_curr_examples_layer_key = rearrange(task_grads_perexample[task_curr][:, l, 0, task_curr, :, :, :], 'ex len h dh -> ex (len h dh)')
        
        task_prev_examples_layer_value = rearrange(task_grads_perexample[task_prev][:, l, 1, task_prev, :, :, :], 'ex len h dh -> ex (len h dh)')
        task_curr_examples_layer_value = rearrange(task_grads_perexample[task_curr][:, l, 1, task_curr, :, :, :], 'ex len h dh -> ex (len h dh)')
        
        #obtain norms
        task_prev_examples_layer_value_norm = torch.norm(task_prev_examples_layer_value, p=2, dim=-1, keepdim=True)
        task_curr_examples_layer_value_norm = torch.norm(task_curr_examples_layer_value, p=2, dim=-1, keepdim=True)

        #normalize
        task_prev_examples_layer_key_normalized = normalize(task_prev_examples_layer_key, p=2, dim=-1)
        task_curr_examples_layer_key_normalized = normalize(task_curr_examples_layer_key, p=2, dim=-1)
        
        task_prev_examples_layer_value_normalized = normalize(task_prev_examples_layer_value, p=2, dim=-1)
        task_curr_examples_layer_value_normalized = normalize(task_curr_examples_layer_value, p=2, dim=-1)        
             
        if task_similarity(task_prev, task_curr, l) > args.task_similarity:

            for i in range(0, len(task_grads_perexample[task_prev])):
                
                class_i = task_labels_perexample[task_prev][i].item()
                        
                for j in range(0, len(task_grads_perexample[task_curr])):
                    
                    class_j = task_labels_perexample[task_curr][j].item()                    
                    
                    if task_prev_examples_layer_key_normalized[i].shape == task_curr_examples_layer_key_normalized[j].shape:
                        
                        jac1 = task_prev_examples_layer_key_normalized[i]
                        jac2 = task_curr_examples_layer_key_normalized[j]
                    
                        result = torch.dot(jac1, jac2)
                        
                        #print("Sim example task curr, task prev: ", result)
                        #print("Percentile low: ", task_ntk_means[task_prev][l][0])
                        #print("Percentile high: ", task_ntk_means[task_prev][l][1])
                    
                        #backward transfer uses a range of mid similarity examples
                        #for backward transfer, it would be like adding a new example
                        if class_i in task_ntk_means[task_prev].keys() and result > 0 and result >= task_ntk_means[task_prev][class_i][l]["perc_low"] and result <= task_ntk_means[task_prev][class_i][l]["perc_high"]:
                            #aligned_keys_layer = gradient_midpoint(task_grads_perexample[task_prev][i, l, 0, task_prev, ...].flatten(), task_grads_perexample[task_curr][j, l, 0, task_curr, ...].flatten())
                            aligned_values_layer = gradient_midpoint(task_prev_examples_layer_value_normalized[i], task_curr_examples_layer_value_normalized[j])
                           
                            #unnormalise
                            #aligned_values_layer = torch.reshape(aligned_values_layer, (5, 6, 64)).unsqueeze(0) / task_prev_examples_layer_value_norm[i]
                            dim_1 = task_grads_perexample[task_prev].shape[-1]
                            dim_2 = task_grads_perexample[task_prev].shape[-2]
                            dim_3 = task_grads_perexample[task_prev].shape[-3]
                            aligned_values_layer = torch.reshape(aligned_values_layer, (dim_3, dim_2, dim_1)).unsqueeze(0) / task_prev_examples_layer_value_norm[i]

                            #then add the new example
                            template_new_examples[0, l, 1, task_prev, :, :, :] = aligned_values_layer
                            #template_old_examples[0, l, 1, task_prev, :, :, :] = torch.reshape(task_prev_examples_layer_value_normalized[i], (5, 6, 64)).unsqueeze(0)
                            template_old_examples[0, l, 1, task_prev, :, :, :] = torch.reshape(task_prev_examples_layer_value_normalized[i], (dim_3, dim_2, dim_1)).unsqueeze(0)
                            
                            #add unnormalised versions to evaluate step [current actual example, new gradient-aligned example]
                            new_aligned_examples_dict[i].append(template_new_examples)
                            
                            if len(old_aligned_examples_dict[i]) == 0:
                                old_aligned_examples_dict[i].append(template_old_examples)
                                                        
                            #print("The NTK between ", str(i), " and ", str(j), " at layer ", str(l), " is: ", result)
                            print("Backward transfer occurred")
                            

        #Evaluate using vectorized GRPO for efficiency
        if args.with_grpo == 1:
            print("STARTING GRPO EVALUATION AT LAYER: ", l)
            current_params = {k: v.detach() for k, v in model_grpo.module.named_parameters() if v.requires_grad}
        
            #new_aligned_examples_dict = evaluate_grpo(model_grpo, optimizer_grpo, checkpoint_model, optimizer_checkpoint, input_target, task_prev, new_aligned_examples_dict, old_aligned_examples_dict, max_norm)
            new_aligned_examples_dict = evaluate_grpo_vectorized(model_grpo, current_params, input_target, task_prev, new_aligned_examples_dict, old_aligned_examples_dict, max_norm)
        
        #all_new_aligned_examples = torch.cat(new_aligned_examples_list, dim = 0)
        #add aligned examples to previous task grads
        if len([item for sublist in new_aligned_examples_dict.values() for item in sublist]) > 0:
            all_new_aligned_examples = torch.cat([item for sublist in new_aligned_examples_dict.values() for item in sublist], dim = 0)
            task_grads_perexample[task_prev] = torch.cat((task_grads_perexample[task_prev], all_new_aligned_examples), dim = 0)
            
    return task_grads_perexample[task_prev]
    
def compute_ntk_forward(task_prev: int, task_curr: int, input_target: [], model_grpo: torch.nn.Module, optimizer_grpo: torch.optim.Optimizer, args = None):#, checkpoint_model, optimizer_checkpoint, max_norm: float, args=None, eps = 1e-8):
    
    #only do the first layer
    for l in range(0, 1): #5 layers
                
        #examples for current task
        new_aligned_examples_dict = {idx: [] for idx in range(0, len(task_grads_perexample[task_curr]))}
        old_aligned_examples_dict = {idx: [] for idx in range(0, len(task_grads_perexample[task_curr]))}
  
        template_new_examples = torch.zeros_like(task_grads_perexample[task_curr][0:1])
        template_old_examples = torch.zeros_like(task_grads_perexample[task_curr][0:1])
        
        task_prev_examples_layer_key = rearrange(task_grads_perexample[task_prev][:, l, 0, task_prev, :, :, :], 'ex len h dh -> ex (len h dh)')
        task_curr_examples_layer_key = rearrange(task_grads_perexample[task_curr][:, l, 0, task_curr, :, :, :], 'ex len h dh -> ex (len h dh)')
        
        task_prev_examples_layer_value = rearrange(task_grads_perexample[task_prev][:, l, 1, task_prev, :, :, :], 'ex len h dh -> ex (len h dh)')
        task_curr_examples_layer_value = rearrange(task_grads_perexample[task_curr][:, l, 1, task_curr, :, :, :], 'ex len h dh -> ex (len h dh)')
        
        #obtain norms
        task_prev_examples_layer_value_norm = torch.norm(task_prev_examples_layer_value, p=2, dim=-1, keepdim=True)
        task_curr_examples_layer_value_norm = torch.norm(task_curr_examples_layer_value, p=2, dim=-1, keepdim=True)

        #normalize
        task_prev_examples_layer_key_normalized = normalize(task_prev_examples_layer_key, p=2, dim=-1)
        task_curr_examples_layer_key_normalized = normalize(task_curr_examples_layer_key, p=2, dim=-1)
        
        task_prev_examples_layer_value_normalized = normalize(task_prev_examples_layer_value, p=2, dim=-1)
        task_curr_examples_layer_value_normalized = normalize(task_curr_examples_layer_value, p=2, dim=-1)        
             
        if task_similarity(task_prev, task_curr, l) > args.task_similarity:

            for i in range(0, len(task_grads_perexample[task_curr])):
                
                class_i = task_labels_perexample[task_curr][i].item()
                
                for j in range(0, len(task_grads_perexample[task_prev])):
                    
                    class_j = task_labels_perexample[task_prev][j].item()                    
                    
                    if task_curr_examples_layer_key_normalized[i].shape == task_prev_examples_layer_key_normalized[j].shape:
                        
                        jac1 = task_curr_examples_layer_key_normalized[i]
                        jac2 = task_prev_examples_layer_key_normalized[j]
                    
                        result = torch.dot(jac1, jac2)
                        
                        #print("Sim example task curr, task prev: ", result)
                        #print("Percentile low: ", task_ntk_means[task_prev][l][0])
                        #print("Percentile high: ", task_ntk_means[task_prev][l][1])
                    
                        #backward transfer uses a range of mid similarity examples
                        #for backward transfer, it would be like adding a new example
                        if class_i in task_ntk_means[task_curr].keys() and result > 0 and result >= task_ntk_means[task_curr][class_i][l]["perc_high"]:
                            #aligned_keys_layer = gradient_midpoint(task_grads_perexample[task_prev][i, l, 0, task_prev, ...].flatten(), task_grads_perexample[task_curr][j, l, 0, task_curr, ...].flatten())
                            aligned_values_layer = gradient_midpoint(task_curr_examples_layer_value_normalized[i], task_prev_examples_layer_value_normalized[j])
                           
                            #unnormalise
                            dim_1 = task_grads_perexample[task_curr].shape[-1]
                            dim_2 = task_grads_perexample[task_curr].shape[-2]
                            dim_3 = task_grads_perexample[task_curr].shape[-3]
                            
                            aligned_values_layer = torch.reshape(aligned_values_layer, (dim_3, dim_2, dim_1)).unsqueeze(0) / task_curr_examples_layer_value_norm[i]
                            #aligned_values_layer = torch.reshape(aligned_values_layer, (5, 6, 64)).unsqueeze(0) / task_curr_examples_layer_value_norm[i]

                            #then add the new example
                            template_new_examples[0, l, 1, task_curr, :, :, :] = aligned_values_layer
                            template_old_examples[0, l, 1, task_curr, :, :, :] = torch.reshape(task_curr_examples_layer_value_normalized[i], (dim_3, dim_2, dim_1)).unsqueeze(0)
                            #template_old_examples[0, l, 1, task_curr, :, :, :] = torch.reshape(task_curr_examples_layer_value_normalized[i], (5, 6, 64)).unsqueeze(0)
                            
                            #add unnormalised versions to evaluate step [current actual example, new gradient-aligned example]
                            new_aligned_examples_dict[i].append(template_new_examples)
                            
                            if len(old_aligned_examples_dict[i]) == 0:
                                old_aligned_examples_dict[i].append(template_old_examples)
                                                        
                            #print("The NTK between ", str(i), " and ", str(j), " at layer ", str(l), " is: ", result)
                            print("Forward transfer occurred")
                                  
        #all_new_aligned_examples = torch.cat(new_aligned_examples_list, dim = 0)
        #add aligned examples to previous task grads
        if len([item for sublist in new_aligned_examples_dict.values() for item in sublist]) > 0:
            all_new_aligned_examples = torch.cat([item for sublist in new_aligned_examples_dict.values() for item in sublist], dim = 0)
            task_grads_perexample[task_curr] = torch.cat((task_grads_perexample[task_curr], all_new_aligned_examples), dim = 0)
            
    return task_grads_perexample[task_curr]    

def evaluate_grpo_vectorized(model_grpo, params, input_target, task_prev, new_aligned_examples_dict, old_aligned_examples_dict, max_norm):
    
    refined_dict = {}
    lr = 0.01 # Set to your optimizer's LR
    
    for actual_example, aligned_grads in new_aligned_examples_dict.items():
        if len(old_aligned_examples_dict[actual_example]) == 0:
            refined_dict[actual_example] = []
            continue

        # 1. Baseline Reward (Actual)
        # Pass a dummy zero-grad or actual example grad
        zero_grad = torch.zeros_like(aligned_grads[0]).unsqueeze(0)
        reward_actual, log_probs_actual = virtual_grad_step_vectorized(
            model_grpo, params, input_target, task_prev, zero_grad, lr, max_norm
        )
        
        # 2. Batch Reward (All Candidates)
        # Convert list of gradient candidates to a single [G, P, D] tensor
        candidate_grads = torch.stack(aligned_grads) 
        
        rewards_aligned, log_probs_aligned = virtual_grad_step_vectorized(
            model_grpo, params, input_target, task_prev, candidate_grads, lr, max_norm
        )

        # 3. GRPO Math (Now on Tensors)
        # raw_reward = actual - candidate
        raw_rewards = reward_actual - rewards_aligned
        
        # Advantage
        adv = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std() + 1e-8)
                
        # Ratio & KL
        # log_probs_actual is [1, C], log_probs_aligned is [G, C]
        ratio = torch.exp(log_probs_aligned - log_probs_actual).mean(dim=(1, 2)) 
        kl = F.kl_div(log_probs_actual.expand_as(log_probs_aligned), 
                      log_probs_aligned, log_target=True, reduction='none').sum(dim=-1).mean(dim=(-1))
        
        # 4. Filter
        epsilon, beta = 0.1, 0.01
        clipped_obj = torch.min(ratio * adv, torch.clamp(ratio, 1-epsilon, 1+epsilon) * adv) - (beta * kl)
        
        top_v, top_i = torch.topk(clipped_obj, min(5, len(clipped_obj)))
        
        # Keep only positive improvement
        final_indices = top_i[top_v > 0]
        refined_dict[actual_example] = [aligned_grads[i] for i in final_indices.tolist()]

    return refined_dict
    
def virtual_grad_step_vectorized(model_grpo, params, input_target, task_prev, candidate_grads, lr, max_norm):
    """
    candidate_grads: Shape [G, P*D] or [G, P, D]
    """
    inner_model = model_grpo.module if hasattr(model_grpo, 'module') else model_grpo
    
    # 1. Prepare Inputs
    input_grpo = input_target[0]
    target_grpo = input_target[1]
    selected = random.sample(range(0, len(target_grpo)), min(len(target_grpo), len(target_grpo)))
    batch_img = input_grpo[selected]
    batch_tgt = target_grpo[selected]
    #batch_img = input_grpo
    #batch_tgt = target_grpo
    
    # 2. Simulate the Step for ALL candidates at once
    # params['e_prompt.prompt'] is [P, D]. 
    # candidate_grads is [G, P, D].
    # virtual_prompts becomes [G, P, D]
    base_prompt = params['e_prompt.prompt']
    
    # Clip grads manually for the batch (simulating clip_grad_norm_)
    #gnorm = candidate_grads.norm(2, dim=(1,2), keepdim=True)
    #clip_coef = max_norm / (gnorm + 1e-6)
    #clip_coef = torch.clamp(clip_coef, max=1.0)
    #clipped_grads = candidate_grads * clip_coef
    
    #dont clip 
    clipped_grads = candidate_grads
    
    virtual_prompts = base_prompt.unsqueeze(0) - lr * clipped_grads

    # 3. Vectorized Forward Pass
    # We use vmap to run the model G times, each time with a different prompt
    def single_eval(p_slice):
        # Create a temporary parameter dict for this specific candidate
        v_params = {k: v for k, v in params.items()}
        v_params['e_prompt.prompt'] = p_slice
        
        predictions = functional_call(inner_model, v_params, args=(batch_img,), 
                                 kwargs={
                                     'task_id': input_target[2], 
                                     'prompt_id': input_target[3], 
                                     'train': input_target[4], 
                                     'prompt_momentum': input_target[5].prompt_momentum
                                 })
        logits = predictions[0]['logits']
        
        # Apply your masking logic
        if input_target[5].train_mask and input_target[6] is not None:
            mask = input_target[6][input_target[2]]
            not_mask = torch.tensor([i for i in range(input_target[5].nb_classes) if i not in mask]).to(logits.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
            
        loss = F.cross_entropy(logits, batch_tgt, reduction='mean')
        log_probs = F.log_softmax(logits, dim=-1)
        return loss, log_probs # Reward is negative loss

    # Parallel execution!
    # Instead of rewards, all_log_probs = vmap(single_eval)(virtual_prompts)
    rewards_list = []
    log_probs_list = []

    for i in range(virtual_prompts.shape[0]):
        # Extract one candidate's prompt [P, D] - no extra dimension!
        p_slice = virtual_prompts[i].squeeze(0)      
        r, lp = single_eval(p_slice)
        rewards_list.append(r)
        log_probs_list.append(lp)

    rewards = torch.stack(rewards_list)
    all_log_probs = torch.stack(log_probs_list)
    
    print("REWARDS POSITIVE")
    print(rewards)
    
    return rewards, all_log_probs    

def evaluate_grpo(model_grpo: torch.nn.Module, optimizer_grpo: torch.optim.Optimizer, checkpoint_model, optimizer_checkpoint, input_target: [], task_prev, new_aligned_examples_dict, old_aligned_examples_dict, max_norm: float):
    #make step on a copy of the model 

    #evaluate loss
        
    #here compare the reward of the original example with the reward of each possible aligned example for that actual example
    for actual_example, aligned_examples in new_aligned_examples_dict.items():
        
        if len(old_aligned_examples_dict[actual_example]) > 0: #there is at least one other example to align to
            
            reward_actual_example, log_probs_actual = virtual_grad_step(model_grpo, optimizer_grpo, input_target, task_prev, old_aligned_examples_dict[actual_example][0], max_norm)

            rewards_aligned_examples = []
            ratio_aligned_examples = []
            kl_aligned_examples = []
                              
            for i in range(0, len(aligned_examples)):
        
                reward_aligned_example_list = []
                log_probs_aligned_list = []
                    
                for j in range(0, 1): #just one for now
                    
                    reward_aligned_example, log_probs_aligned = virtual_grad_step(model_grpo, optimizer_grpo, input_target, task_prev, new_aligned_examples_dict[actual_example][i], max_norm)
                    reward_aligned_example = reward_actual_example - reward_aligned_example #the larger the better
            
                    #only examples with positive reward
                    if reward_aligned_example > 0:
                        reward_aligned_example_list.append(reward_aligned_example)
                        log_probs_aligned_list.append(log_probs_aligned)
                        
                    #restart model to original checkpoint
                    model_grpo.load_state_dict(checkpoint_model, strict=False)
                    optimizer_grpo.load_state_dict(optimizer_checkpoint)
                    #torch.cuda.empty_cache()                        
                                                
                #take mean
                if len(reward_aligned_example_list) > 0:
                    reward_aligned_example = torch.mean(torch.stack(reward_aligned_example_list))
                    log_probs_aligned = torch.mean(torch.stack(log_probs_aligned_list), dim = 0)
                                        
                    rewards_aligned_examples.append(reward_aligned_example)
                    
                    #for GRPO formulation - ratio r_i
                    ratio_aligned_example = torch.mean(log_probs_aligned / log_probs_actual)  #Perhaps change this to exp
                    ratio_aligned_examples.append(ratio_aligned_example)
                    
                    #for GRPO formulation - KL
                    kl = F.kl_div(log_probs_actual, log_probs_aligned, log_target=True, reduction='batchmean')
                    kl_aligned_examples.append(kl)
                
            
            #for GRPO formulation -- A_i
            if len(rewards_aligned_examples) > 0:
                rewards_aligned_examples = torch.tensor(rewards_aligned_examples)
                ratio_aligned_examples = torch.stack(ratio_aligned_examples)
                kl_aligned_examples = torch.stack(kl_aligned_examples)
                
                mean_rewards_aligned = torch.mean(rewards_aligned_examples)
                std_rewards_aligned = torch.std(rewards_aligned_examples)
                
                rewards_aligned_examples = (rewards_aligned_examples - mean_rewards_aligned) / std_rewards_aligned
                
                print("MEAN: ", mean_rewards_aligned)
                print("STD: ", std_rewards_aligned)
                print("KL: ", kl_aligned_examples)

                
                #apply GRPO full formulation - clipping and KL
                epsilon = 0.1
                beta = 0.01
                
                clipped_rewards_aligned = torch.min(ratio_aligned_examples * rewards_aligned_examples, torch.clamp(ratio_aligned_examples, 1 - epsilon, 1 + epsilon) * rewards_aligned_examples) - (beta * kl_aligned_examples)
                
                #rank aligned examples, from the largest gain, and select only those top k in new_aligned_examples
                k = min(5, clipped_rewards_aligned.size(0))
                top_values, top_indices = torch.topk(clipped_rewards_aligned, k, largest=True, sorted=True)
            
                #only select non-negative
                top_values = top_values[top_values > 0]
                top_indices = top_indices[0:len(top_values)]
            
                print("Reward actual example: ", reward_actual_example)
                print("Top rewards aligned examples: ", top_values)
            
                selected_aligned_examples = [new_aligned_examples_dict[actual_example][i] for i in top_indices.tolist()]
            
                new_aligned_examples_dict[actual_example] = selected_aligned_examples
        
    return new_aligned_examples_dict 


def virtual_grad_step(model_grpo: torch.nn.Module, optimizer_grpo: torch.optim.Optimizer, input_target: [], task_prev, example, max_norm: float):

    #only e_prompt will change, head and bias will remain the same
    
    #select a few random examples
    input_grpo = input_target[0]
    target_grpo = input_target[1]
        
    optimizer_grpo.zero_grad()
    model_grpo.module.e_prompt.prompt.grad = torch.cat((task_grads_perexample[task_prev], example)).mean(dim=0) 
            
    torch.nn.utils.clip_grad_norm_(model_grpo.parameters(), max_norm)
    optimizer_grpo.step()
    
    selected = random.sample(range(0, len(target_grpo)), min(5, len(target_grpo)))
    
    output = model_grpo(input_grpo[selected], task_id=input_target[2], prompt_id=input_target[3], train=input_target[4], prompt_momentum=input_target[5].prompt_momentum)
    logits = output[0]['logits']
                
    # here is the trick to mask out classes of non-current tasks
    if input_target[5].train_mask and input_target[6] is not None:
        mask = input_target[6][input_target[2]]
        not_mask = np.setdiff1d(np.arange(input_target[5].nb_classes), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64)
        logits = logits.index_fill(dim=1, index=not_mask.to(logits.device), value=float('-inf'))

    #DBP do not reduce
    criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
                
    loss = criterion(logits, target_grpo[selected])  # base criterion (CrossEntropyLoss) 
    loss = loss.detach().cpu()

    #obtain log probs 
    log_probs = F.log_softmax(logits[torch.isfinite(logits)], dim=-1)
    log_probs = log_probs.detach().cpu()
                                
    return loss.mean(), log_probs

#restart each task
def restart_ntk_means(task_curr: int):
    for i in range(0, task_curr):
        task_ntk_means[task_curr] = []

#is not the mean but the percentile
def compute_ntk_means(task_curr: int, percentile_low = 0.0, percentile_high = 0.0, eps = 1e-8):
    
    #calculate ntk of all replay examples of the same task first? then only target examples close to the mean ntk (i.e. dot product) would be used for transfer
    
    if not task_ntk_means[task_curr]:
        task_ntk_means[task_curr] = {}
    
    num_layers = 5
    
    classes = task_labels_perexample[task_curr].cpu().tolist()
        
    for cl in set(classes):
            
        cl_ind = [i for i, x in enumerate(classes) if x == cl]
          
        for l in range(0, num_layers): #5 layers 
            
            result_all_layer = []

            task_curr_examples_layer_key = rearrange(task_grads_perexample[task_curr][cl_ind, l, 0, task_curr, :, :, :], 'ex len h dh -> ex (len h dh)')
            task_curr_examples_layer_key_norm = normalize(task_curr_examples_layer_key, p=2, dim=-1)

            for i in range(0, len(task_grads_perexample[task_curr][cl_ind])):

                for j in range(i + 1, len(task_grads_perexample[task_curr][cl_ind])):
         # Compute J(x1) @ J(x2).T
         
                    if i != j:
                        #jac1 = task_grads_perexample[task_curr][i, l, 0, task_curr, ...].flatten()[torch.nonzero(task_grads_perexample[task_curr][i, l, 0, task_curr, ...].flatten())]
            
                        #jac2 = task_grads_perexample[task_curr][j, l, 0, task_curr, ...].flatten()[torch.nonzero(task_grads_perexample[task_curr][j, l, 0, task_curr, ...].flatten())]
            
                        #jac1 = jac1[0]
                        #jac1 = jac1 / (jac1.norm() + eps)
                        
                        #jac2 = jac2[0]
                        #jac2 = jac2 / (jac2.norm() + eps)
                        
                        jac1 = task_curr_examples_layer_key_norm[i]
                        jac2 = task_curr_examples_layer_key_norm[j]
                        
                        result = torch.dot(jac1, jac2)
                    
                        result_all_layer.append(result)
                        
            #only append if there is any percentile
            if len(result_all_layer) > 0:
                tau_layer = torch.quantile(torch.tensor(result_all_layer), torch.tensor([percentile_low, percentile_high]))
                tau_layer = tau_layer.tolist()
                
                #must be per task per layer, it's gonna be a list
                #append a new value if not exists, otherwise simply update using exponential moving average for different batches/epochs
                
                if cl in task_ntk_means[task_curr]:
                    if l in task_ntk_means[task_curr][cl]:
                        alpha = 0.9
                        tau_layer[0] = (1 - alpha) * task_ntk_means[task_curr][cl][l]["perc_low"] + alpha * tau_layer[0] #update low percentile of current layer with EMA
                        tau_layer[1] = (1 - alpha) * task_ntk_means[task_curr][cl][l]["perc_high"] + alpha * tau_layer[1] #update high percentile of current layer with EMA
                    
                        task_ntk_means[task_curr][cl][l] = {"perc_low": tau_layer[0], "perc_high": tau_layer[1]}
                    else:
                        task_ntk_means[task_curr][cl][l] = {}
                        task_ntk_means[task_curr][cl][l] = {"perc_low": tau_layer[0], "perc_high": tau_layer[1]}

                else:
                    task_ntk_means[task_curr][cl] = {}
                    task_ntk_means[task_curr][cl][l] = []
                    
                    task_ntk_means[task_curr][cl][l] = {"perc_low": tau_layer[0], "perc_high": tau_layer[1]}

           
def gradient_midpoint(gA, gB, eps=1e-8):
    gA = gA.reshape(-1)
    gB = gB.reshape(-1)

    gA_hat = gA / (gA.norm() + eps)
    gB_hat = gB / (gB.norm() + eps)

    g_mid = gA_hat + gB_hat
    g_mid = g_mid / (g_mid.norm() + eps)

    return g_mid
    
#DBP in each epoch, both previous and current task will be trained... for the last few epochs, we will apply our method
#This is WTP only, TAP is in another function below
def train_one_epoch(model: torch.nn.Module, original_model_list: list, #torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    model.train(set_training_mode)

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'

    #DBP in each epoch, we need to go through each of the tasks separately
    for t, loader in enumerate(data_loader):
        
        original_model_list[t].eval()
        
        for input, target in metric_logger.log_every(loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            with torch.no_grad():
                if original_model_list is not None:
                    output = original_model_list[t](input)
                    logits = output[0]['logits']

                    if args.train_mask and class_mask is not None:
                        mask = []
                        for id in range(t + 1):
                            mask.extend(class_mask[id])
                        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                        logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        prompt_id = torch.max(logits, dim=1)[1]
                        # translate cls to task_id
                        prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(
                            -1)
                    else:
                        prompt_id = None
                else:
                    raise NotImplementedError("original model is None")
                    
            output = model(input, task_id=t, prompt_id=prompt_id, train=set_training_mode,
                        prompt_momentum=args.prompt_momentum)
            logits = output[0]['logits']
            
            # here is the trick to mask out classes of non-current tasks
            if args.train_mask and class_mask is not None:
                mask = class_mask[t]
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)
            # TODO add contrastive loss
            # loss += orth_loss(output[0]['pre_logits'], target, device, args)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            #DBP here we need to incorporate the proposed gradient alignment process between current task and each previous one
            optimizer.zero_grad()
            loss.backward() 
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            torch.cuda.synchronize()
            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
#DBP this whole function changed to accomodate multiple tasks
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
             device, i=-1, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(i + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output 
            if original_model is not None:
                output_ref = original_model(input)
                    
                if isinstance(output_ref, list):
                    logits_ref = [o['logits'] for o in output_ref]
                    logits_all = torch.stack(logits_ref).mean(dim = 0)
                else:
                    logits_all = output_ref['logits']
                        
                if args.train_mask and class_mask is not None:
                    mask = []
                        
                    if isinstance(task_id, (list, tuple)):
                        for tid in task_id:
                            mask.extend(class_mask[tid])
                    else:
                        for tid in range(task_id + 1):
                            mask.extend(class_mask[tid])
                                
                    not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                    logits = logits_all.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        
                prompt_id = torch.max(logits, dim=1)[1]
                # translate cls to task_id
                prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(-1)
                # print(prompt_id)
            else:
                raise NotImplementedError("original model is None")

            output = model(input, task_id=task_id, prompt_id=prompt_id)
            
            if isinstance(output, list):
                logits_list = [o['logits'] for o in output]
                prompt_idx_list = [o.get('prompt_idx', None) for o in output] # tensor B x topk
            else:
                logits_list = [output['logits']]
                prompt_idx_list = [output.get('prompt_idx', None)]
                
            total_loss = 0
            total_acc1 = 0
            total_acc5 = 0
            total_task_acc = 0
            
            for tid, logits in enumerate(logits_list):
                if args.task_inc and class_mask is not None:
                    # adding mask to output logits
                    mask = class_mask[tid if tid < len(class_mask) else i]
                    mask = torch.tensor(mask, dtype=torch.int64).to(device)
                    logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                    logits_mask = logits_mask.index_fill(1, mask, 0.0)
                    logits = logits + logits_mask
                
                loss = criterion(logits, target)

                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                #DBP task inference accuracy
                
                prompt_idx = prompt_idx_list[tid]
                if prompt_idx is not None:
                    task_acc = utils.task_inference_accuracy(prompt_idx, target, target_task_map)
                else:
                    task_acc = torch.tensor(0.0, device=device)
                
                total_loss += loss
                total_acc1 += acc1
                total_acc5 =+ acc5
                total_task_acc =+ task_acc
                
            n_tasks = len(logits_list)
            total_loss /= n_tasks
            total_acc1 /= n_tasks
            total_acc5 /= n_tasks
            total_task_acc /= n_tasks

            metric_logger.meters['Loss'].update(total_loss.item())
            metric_logger.meters['Acc@1'].update(total_acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(total_acc5.item(), n=input.shape[0])
            metric_logger.meters['Acc@task'].update(total_task_acc.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@task {task_acc.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(task_acc=metric_logger.meters['Acc@task'],
                top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
#DBP add evaluate train parameter
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, target_task_map=None, acc_matrix=None, args=None, evaluate_train = False):
    stat_matrix = np.zeros((4, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id + 1):
        #DBP allow to evaluate train
        if evaluate_train:
            test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['train'],
                              device=device, i=i, task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                              args=args)
        else:                        
            test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'],
                              device=device, i=i, task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                              args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        stat_matrix[3, i] = test_stats['Acc@task']

        acc_matrix[i, task_id] = test_stats['Acc@1']
        
    #DBP write acc_matrix
    df_acc_matrix = pd.DataFrame(acc_matrix)
    
    if args.output_dir and utils.is_main_process():
        rep = args.rep
       
        df_acc_matrix.to_csv(os.path.join(args.output_dir, 'hideprompt_accuracies_attask_' + str(task_id) + '_replay' + str(int(args.replay_percentage * 100)) + '_rep' + str(rep) + '.csv'), index = False)
    #DBP end

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@task: {:.4f}\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1,
        avg_stat[3],
        avg_stat[0],
        avg_stat[1],
        avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module,
                       criterion, data_loader: Iterable, data_loader_per_cls: Iterable,
                       optimizer: torch.optim.Optimizer,
                       lr_scheduler,
                       device: torch.device,
                       class_mask=None, target_task_map=None, args=None, ):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    #DBP add acc_matrix_train
    acc_matrix_train = np.zeros((args.num_tasks, args.num_tasks))
    global cls_mean
    global cls_cov
    global task_grads_keys
    global task_grads_values
    global task_grads
    global task_grads_perexample
    global task_labels_perexample
    global task_ntk_means
    cls_mean = {tid: {} for tid in range(0, args.num_tasks)}
    cls_cov = {tid: {} for tid in range(0, args.num_tasks)} 
    task_grads_keys = {tid: {} for tid in range(0, args.num_tasks)}
    task_grads_values = {tid: {} for tid in range(0, args.num_tasks)}
    task_grads = {tid: {} for tid in range(0, args.num_tasks)}
    #DBP per example grads
    task_grads_perexample = {tid: {} for tid in range(0, args.num_tasks)}
    task_labels_perexample = {tid: {} for tid in range(0, args.num_tasks)}
    task_ntk_means = {tid: {} for tid in range(0, args.num_tasks)}



    for task_id in range(args.num_tasks):
        
        #DBP Developing just for first two tasks for now
        
        #if task_id == 3:
            #break
            
        
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            if args.larger_prompt_lr:
                # This is a simple yet effective trick that helps to learn task-specific prompt better.
                base_params = [p for name, p in model_without_ddp.named_parameters() if
                            'prompt' in name and p.requires_grad == True]
                base_fc_params = [p for name, p in model_without_ddp.named_parameters() if
                                'prompt' not in name and p.requires_grad == True]
                base_params = {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
                base_fc_params = {'params': base_fc_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay}
                network_params = [base_params, base_fc_params]
                optimizer = create_optimizer(args, network_params)
            else:
                optimizer = create_optimizer(args, model)
            
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None

        # load original model checkpoint
        # DBP load all previous original model checkpoints
        original_model_list = []
        for t_id in range(0, task_id + 1):
            if args.trained_original_model:
                original_model_list.append(original_model) #DBC copies of models for now
                
                original_checkpoint_path = os.path.join(args.trained_original_model,
                                                    'checkpoint/task{}_checkpoint.pth'.format(t_id + 1))
                if os.path.exists(original_checkpoint_path):
                    print('Loading checkpoint from:', original_checkpoint_path)
                    original_checkpoint = torch.load(original_checkpoint_path, map_location=device, weights_only = False) #DBP added weights_only = False                 
                    original_model_list[t_id].load_state_dict(original_checkpoint['model'])
                else:
                    print('No checkpoint found at:', original_checkpoint_path)
                    return
        # if model already trained
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
        
        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (
                        slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (
                        slice(None), slice(cur_start, cur_end))
                    prev_idx = (
                        slice(None), slice(None),
                        slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (
                        slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            model.module.e_prompt.prompt.grad.zero_()
                            model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]
                            # optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.e_prompt.prompt.grad.zero_()
                            model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                            # optimizer.param_groups[0]['params'] = model.parameters()

        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.e_prompt.prompt_key.grad.zero_()
                        model.module.e_prompt.prompt_key[cur_idx] = model.module.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.e_prompt.prompt_key.grad.zero_()
                        model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()
  
        #DBP train_one_epoch will consider both current and previous tasks
        epoch_number = 1
        align_each_epochs = args.epochs  #only align at the end
        
        for epoch in range(args.epochs): #it will be just one epoch in OCL
            data_loader_prev_cur = get_task_loader(data_loader, task_id, 'train')
                                            
            print("-----TRAINING TASK =:----- ", task_id)
            
            train_stats = train_one_epoch_current_task(model=model, original_model_list=original_model_list, criterion=criterion,
                                            data_loader=data_loader_prev_cur, optimizer=optimizer,
                                            device=device, epoch=epoch, max_norm=args.clip_grad,
                                            set_training_mode=True, task_id=task_id, class_mask=class_mask,
                                            target_task_map=target_task_map, args=args, )

            if task_id > 0:
                print("-----RETRAINING AT TASK <:----- ", task_id)
                
                for epoch_refine in range(0, 1): 
                    train_stats = train_one_epoch_previous_task(model=model, original_model_list=original_model_list, criterion=criterion,
                                                data_loader=data_loader_prev_cur, optimizer=optimizer,
                                                device=device, epoch=epoch, max_norm=args.clip_grad,
                                                set_training_mode=True, task_id=task_id, class_mask=class_mask,
                                                target_task_map=target_task_map, args=args, )

            if lr_scheduler:
                lr_scheduler.step(epoch)
                           
            epoch_number += 1

        if args.prompt_momentum > 0 and task_id > 0:
            if args.use_prefix_tune_for_e_prompt:
                with torch.no_grad():
                    print(model.module.e_prompt.prompt[:, :, task_id].shape)
                    print(
                        model.module.e_prompt.prompt[:, :, 0:task_id].detach().clone().mean(dim=2, keepdim=True).shape)
                    model.module.e_prompt.prompt[:, :, task_id].copy_(
                        (1 - args.prompt_momentum) * model.module.e_prompt.prompt[:, :, task_id].detach().clone()
                        + args.prompt_momentum * model.module.e_prompt.prompt[:, :, 0:task_id].detach().clone().mean(
                            dim=2))

        # compute mean and variance
        _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, task_id=task_id,
                      class_mask=class_mask, args=args)

    #DBP using only first original model from list for now... 
    #DBP add train stats before tap but after wtp
        train_stats_before_tap = evaluate_till_now(model=model, original_model=original_model_list[0], data_loader=data_loader,
                                       device=device,
                                       task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                       acc_matrix=acc_matrix_train, args=args, evaluate_train = True)
                                       
    #DBP add test stats before tap but after wtp                                 
        test_stats_before_tap = evaluate_till_now(model=model, original_model=original_model_list[0], data_loader=data_loader,
                                       device=device,
                                       task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                       acc_matrix=acc_matrix, args=args)
                                       
        if task_id > 0 and not args.not_train_ca:
            pre_ca_test_stats = evaluate_till_now(model=model, original_model=original_model_list[0], data_loader=data_loader,
                                                  device=device,
                                                  task_id=task_id, class_mask=class_mask,
                                                  target_task_map=target_task_map,
                                                  acc_matrix=pre_ca_acc_matrix, args=args)

            #DBP TODO to-do todo
            train_task_adaptive_prediction(model, args, device, class_mask, task_id)

        #DBP restart matrices before evaluating after tap
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        #DBP add acc_matrix_train
        acc_matrix_train = np.zeros((args.num_tasks, args.num_tasks))
    
        #DBP add train stats after tap training 
        train_stats_after_tap = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader,
                                       device=device,
                                       task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                       acc_matrix=acc_matrix_train, args=args, evaluate_train = True)
        #DBP continue                                   
        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader,
                                       device=device,
                                       task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                       acc_matrix=acc_matrix, args=args)

        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_beforetap_{k}': v for k, v in train_stats_before_tap.items()},
                     **{f'test_beforetap_{k}': v for k, v in test_stats_before_tap.items()},
                     **{f'train_aftertap_{k}': v for k, v in train_stats_after_tap.items()},
                     **{f'test_aftertap_{k}': v for k, v in test_stats.items()},
                     }

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir,
                                   '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))),
                      'a') as f:
                f.write(json.dumps(log_stats) + '\n')
                
        #print_task_grads(task_id)


@torch.no_grad()
def _compute_mean(model: torch.nn.Module, data_loader: Iterable, device: torch.device, task_id, class_mask=None,
                  args=None, ):
    model.eval()
    
    for tid in range(0, task_id + 1):
        for cls_id in class_mask[tid]:
            
            data_loader_cls = data_loader[cls_id]['train']
            features_per_cls = []
            
            for i, (inputs, targets) in enumerate(data_loader_cls):
                inputs = inputs.to(device, non_blocking=True)
                
                with torch.no_grad():
                    features = model(inputs, task_id=tid, train=True)[0]['pre_logits']
                    features_per_cls.append(features)
            
            features_per_cls = torch.cat(features_per_cls, dim=0)
            features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]

            dist.barrier()
            dist.all_gather(features_per_cls_list, features_per_cls)

            if args.ca_storage_efficient_method == 'covariance':
                features_per_cls = torch.cat(features_per_cls_list, dim=0)
                # print(features_per_cls.shape)
                cls_mean[cls_id] = features_per_cls.mean(dim=0)
                cls_cov[cls_id] = torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device)
            
            if args.ca_storage_efficient_method == 'variance':
                features_per_cls = torch.cat(features_per_cls_list, dim=0)
                # print(features_per_cls.shape)
                cls_mean[cls_id] = features_per_cls.mean(dim=0)
                cls_cov[cls_id] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device))
                
            if args.ca_storage_efficient_method == 'multi-centroid':
                from sklearn.cluster import KMeans
                n_clusters = args.n_centroids
                features_per_cls = torch.cat(features_per_cls_list, dim=0).cpu().numpy()
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(features_per_cls)
                cluster_lables = kmeans.labels_
                cluster_means = []
                cluster_vars = []
                for i in range(n_clusters):
                   cluster_data = features_per_cls[cluster_lables == i]
                   cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(device)
                   cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(device)
                   cluster_means.append(cluster_mean)
                   cluster_vars.append(cluster_var)
                
                cls_mean[tid][cls_id] = cluster_means
                cls_cov[tid][cls_id] = cluster_vars


def train_task_adaptive_prediction(model: torch.nn.Module, args, device, class_mask=None, task_id=-1):
    model.train()
    run_epochs = args.crct_epochs
    crct_num = 0
    param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'prompt' not in n]
    network_params = [{'params': param_list, 'lr': args.ca_lr, 'weight_decay': args.weight_decay}]
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(network_params, lr=args.ca_lr / 10, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network_params, lr=args.ca_lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for i in range(task_id):
        crct_num += len(class_mask[i])

    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = args.batch_size * 5

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        if args.ca_storage_efficient_method in ['covariance', 'variance']:
            for i in range(task_id + 1):
                for c_id in class_mask[i]:
                    #DBP changed cls_mean[c_id] to cls_mean[i][c_id] and cls_cov[c_id] to cls_cov[i][c_id]
                    mean = torch.tensor(cls_mean[i][c_id], dtype=torch.float64).to(device)
                    cov = cls_cov[i][c_id].to(device)
                    if args.ca_storage_efficient_method == 'variance':
                        cov = torch.diag(cov)
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)

                    sampled_label.extend([c_id] * num_sampled_pcls)

        elif args.ca_storage_efficient_method == 'multi-centroid':
            for i in range(task_id + 1):
               for c_id in class_mask[i]:
                   for cluster in range(len(cls_mean[i][c_id])):
                        #DBP changed cls_mean[c_id] to cls_mean[i][c_id] and cls_cov[c_id] to cls_cov[i][c_id]
                        mean = cls_mean[i][c_id][cluster]
                        var = cls_cov[i][c_id][cluster]
                        if var.mean() == 0:
                            continue
                        m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                        sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                        sampled_data.append(sampled_data_single)
                        sampled_label.extend([c_id] * num_sampled_pcls)
        else:
            raise NotImplementedError


        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
        sampled_label = torch.tensor(sampled_label).long().to(device)
        print(sampled_data.shape)

        inputs = sampled_data
        targets = sampled_label

        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        # print(targets)

        for _iter in range(crct_num):
            inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            outputs = model(inp, fc_only=True)
            logits = outputs[0]['logits']

            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id + 1):
                    mask.extend(class_mask[id])
                # print(mask)
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, tgt)  # base criterion (CrossEntropyLoss)
            acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            #for name, p in model.named_parameters():
            #    if p.requires_grad and p.grad is None:
            #        print(name)
            optimizer.step()
            torch.cuda.synchronize()

            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

            # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        scheduler.step()



def orth_loss(features, targets, device, args):
    if cls_mean:
        # orth loss of this batch
        sample_mean = []
        for k, v in cls_mean.items():
            if isinstance(v, list):
                sample_mean.extend(v)
            else:
                sample_mean.append(v)
        sample_mean = torch.stack(sample_mean, dim=0).to(device, non_blocking=True)
        M = torch.cat([sample_mean, features], dim=0)
        sim = torch.matmul(M, M.t()) / 0.8
        loss = torch.nn.functional.cross_entropy(sim, torch.range(0, sim.shape[0] - 1).long().to(device))
        # print(loss)
        return args.reg * loss
    else:
        sim = torch.matmul(features, features.t()) / 0.8
        loss = torch.nn.functional.cross_entropy(sim, torch.range(0, sim.shape[0] - 1).long().to(device))
        return args.reg * loss
        # return 0.

#DBP combining multiple data loaders
#Here is where we can add a random subset of previous tasks or task ditributions as in HiDe paper (current_task says which is the current)
def get_task_loader(data_loader_dict, task_id, set_type = 'train'):
   
    combined_loader = []
        
    for t in range(0, task_id + 1):
        task_loader = data_loader_dict[t][set_type]

        combined_loader.append(task_loader)
        
    return combined_loader
      
    
    