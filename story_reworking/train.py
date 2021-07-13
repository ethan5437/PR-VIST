'''
This script handling the training process.
'''

import argparse
import math
import time
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
#from revised_loader import revised_Loaders
from Loader_manager import Loaders
from build_story_vocab import Vocabulary
import numpy as np
import copy
from pytorchtools import EarlyStopping
from discriminator import discriminator_model
from datetime import date

                              
def cal_performance(pred, gold, loss_level='sentence', smoothing=False):
#     smoothing = False
    ''' Apply label smoothing if needed '''
    ###LSTM (Concatenation) Modification
    gold_zeros = torch.zeros(gold.size(0),1, device=gold.device, dtype = gold.dtype)
    gold = torch.cat((gold,gold_zeros),1)
            
    ###cal_loss
    loss = cal_loss(pred, gold, loss_level, smoothing)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return loss, n_correct


def cal_loss(pred, gold, loss_level, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    batch_size = gold.size(0)
    seq_size = gold.size(1)
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)  
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1) 
        log_prb = F.log_softmax(pred, dim=1)
        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        if loss_level == 'story':
            loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='none')
            loss = loss.view(batch_size, seq_size)
            loss = torch.sum(loss, 1)
        elif loss_level == 'batch':
            loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')
        else:
            raise ValueError(f'invalid loss_level {loss_level}')

    return loss

def get_reversed_reward(x, reward_rate):
    reversed_reward = -x + 1.5
    return reversed_reward ** reward_rate

def get_reward(pred, gold, discriminator, reward_rate):
    max_seq_len = discriminator.max_seq_len
    
    #mask BOS and EOS
    mask_BOS = (gold != Constants.BOS).long()
    mask_EOS = (gold != Constants.EOS).long()
    mask_PAD = (gold != Constants.PAD).long()
    #mask zeros
    pred = pred * mask_BOS
    pred = pred * mask_EOS
    pred = pred * mask_PAD

    new_pred = torch.empty(0, pred.size(1)).long().to(pred.device)
    for r in pred:
        nz = r.nonzero().squeeze(1)
        z = torch.zeros(r.numel() - nz.numel()).long().to(pred.device)
        z = torch.cat((r[nz], z)).unsqueeze(0)
        new_pred = torch.cat((new_pred, z))
    if new_pred.size(1) < max_seq_len:
        padding = torch.zeros((new_pred.size(0), (max_seq_len-new_pred.size(1)))).long().to(new_pred.device)
        new_pred = torch.cat((new_pred, padding), 1)
    else:
        new_pred = new_pred[:,:200]
    reward = discriminator.batchClassify(new_pred)
    reward = get_reversed_reward(reward, reward_rate)
    return reward

def train_epoch(model, training_data, optimizer, device, smoothing, Dataloader, opt, discriminator):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    count = 0
    
    #for pred dimension, at most 25 words in each sentence
    sentence_max_len = opt.max_token_seq_len-1
    frame_max_len = opt.max_encode_token_seq_len-1

    #testing -- train on one data 
    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):    
        # prepare data
        #LSTM Modification
        frame, frame_pos, frame_sen_pos, frame_gold, targets, targets_pos, targets_sen_pos, targets_gold, previous_targets, story_len = map(lambda x: x.to(device), batch)

        
        """
        testing new model
        """ 
               
        pred = []
        hop = int(opt.hop) 
        length = max(story_len)
        for i in range(length):
            # forward
            optimizer.zero_grad()
                
            gold = targets_gold[i]
            gold_frm = frame_gold[i]

            #first sentence goes with i+hop frame. 
            #For Example. In hop 2, sentence(1+2) goes with frame 3.
#             pred = model(frame[i], frame_pos[i], frame_sen_pos[i], targets[i], targets_pos[i], targets_sen_pos[i], story_len)
            #LSTM Modification
            pred = model(frame[i], frame_pos[i], frame_sen_pos[i], targets[i], targets_pos[i], targets_sen_pos[i], previous_targets[:i+1], story_len)
            
            # backward
            loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
            loss.backward() 
            # update parameters
            optimizer.step_and_update_lr()
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct
            count +=1

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
                
def train_vist_epoch(model, training_data, vist_train_data, optimizer, device, smoothing, Dataloader, opt, discriminator):
    ''' Epoch operation in training phase'''

    model.train()
    roc_iter = iter(training_data)
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    count = 0
    reward_rate = opt.reward_rate
    
    batch_size = opt.batch_size
    sentence_max_len = opt.max_token_seq_len-1
    frame_max_len = opt.max_encode_token_seq_len-1
#     story_max_seq_len = 300
    
    
    hop = opt.hop
    for batch in tqdm(
            vist_train_data, mininterval=2,
            desc='  - (Training)   ', leave=False):
        count +=1
        # prepare vist data
        src_seq, src_pos, src_sen_pos, src_gold, tgt_seq, tgt_pos, tgt_sen_pos, tgt_gold, previous_tgt, story_len = map(lambda x: x.to(device), batch)
        # Adding ROC data
        try:
            roc_batch = next(roc_iter)
        except StopIteration:
            roc_iter = iter(training_data)
            roc_batch = next(roc_iter)

        src_seq_c, src_pos_c, src_sen_pos_c, src_gold_c, tgt_seq_c, tgt_pos_c, tgt_sen_pos_c, tgt_gold_c, previous_tgt_c, story_len_c = map(lambda x: x.to(device), roc_batch)

        #vist 才會有5句以上
        if len(src_seq) > len(src_seq_c):
            diff =  len(src_seq) - len(src_seq_c)
            src_pad_seq_c = torch.zeros(diff, src_seq_c.shape[1], src_seq_c.shape[2]).long().to(device)
            #This is for ground truth(frame) which will be a 0 tensor with story_len * batch size * frame_length
            src_pad_seq_clone_c = src_pad_seq_c.clone()
            #This is for rest of seq, seq_pos, seq_sen_pos to prevend outputing nan. 
            src_pad_seq_c[:,:,0] = Constants.BOS
#             src_pad_seq_c[:,:,1] = Constants.EOS
            
            src_seq_c = torch.cat((src_seq_c, src_pad_seq_c),0)
            src_pos_c = torch.cat((src_pos_c, src_pad_seq_clone_c),0)
            src_sen_pos_c = torch.cat((src_sen_pos_c, src_pad_seq_clone_c),0)
            src_gold_c = torch.cat((src_gold_c, src_pad_seq_clone_c),0)
            
            tgt_pad_seq_c = torch.zeros(diff, tgt_seq_c.shape[1], tgt_seq_c.shape[2]).long().to(device)
            #This is for ground truth(text) which will be a 0 tensor with story_len * batch size * text_length
            tgt_pad_seq_clone_c = tgt_pad_seq_c.clone()
            #This is for ground truth which will be a 0 tensor with story_len * batch size * frame_length
            tgt_pad_seq_c[:,:,0] = Constants.BOS
#             tgt_pad_seq_c[:,:,1] = Constants.EOS
            
            tgt_seq_c = torch.cat((tgt_seq_c, tgt_pad_seq_c),0)
            tgt_pos_c = torch.cat((tgt_pos_c, tgt_pad_seq_clone_c),0)
            tgt_sen_pos_c = torch.cat((tgt_sen_pos_c, tgt_pad_seq_clone_c),0)
            tgt_gold_c = torch.cat((tgt_gold_c, tgt_pad_seq_clone_c),0)
            previous_tgt_c = torch.cat((previous_tgt_c, tgt_pad_seq_clone_c),0)
            
        src_seq = torch.cat((src_seq,src_seq_c), 1)
        src_pos = torch.cat((src_pos,src_pos_c), 1)
        src_sen_pos = torch.cat((src_sen_pos,src_sen_pos_c), 1)
        src_gold = torch.cat((src_gold,src_gold_c), 1)
        
        tgt_seq = torch.cat((tgt_seq,tgt_seq_c), 1)
        tgt_pos = torch.cat((tgt_pos,tgt_pos_c), 1)
        tgt_sen_pos = torch.cat((tgt_sen_pos,tgt_sen_pos_c), 1)
        tgt_gold = torch.cat((tgt_gold,tgt_gold_c), 1)
        previous_tgt = torch.cat((previous_tgt,previous_tgt_c), 1)
        
        story_len = torch.cat((story_len, story_len_c),0)
        length = max(story_len)
        
        # forward
        hop = int(hop)
        story_loss = 0
        for i in range(length):
            # forward
            optimizer.zero_grad()                        
            gold_src = src_gold[i]
            gold = tgt_gold[i]
            #LSTM Modification
            pred = model(src_seq[i], src_pos[i], src_sen_pos[i], tgt_seq[i], tgt_pos[i], tgt_sen_pos[i], previous_tgt[:i+1], story_len)
            
            # backward
            if opt.loss_level == 'sentence' or opt.loss_level == 'hierarchical':
                if opt.is_sen_discriminator:#opt.loss_level == 'sentence' and discriminator != None:
                    loss, n_correct = cal_performance(pred, gold, 'story', smoothing=False)
                    pred = pred.max(1)[1]
                    pred = pred.view(gold.size(0), gold.size(1))
                    reward = get_reward(pred, gold, discriminator, reward_rate)
                    loss = loss * reward
                    loss = loss.sum()
                else:
                    loss, n_correct = cal_performance(pred, gold, 'batch', smoothing=False)

                loss.backward() 
                optimizer.step_and_update_lr()
                total_loss += loss.item()
                non_pad_mask = gold.ne(Constants.PAD)
                n_word = non_pad_mask.sum().item()
                n_word_total += n_word
                n_word_correct += n_correct
            elif opt.loss_level == 'story':                    
                loss, n_correct = cal_performance(pred, gold, 'story', smoothing=False)
                if i < length-1:            
                    story_loss += loss.detach().cpu()#.item()
                elif i == length-1:
                    #addition and move loss to GPU + add require_grad
                    story_loss = loss + story_loss.to(device)#.item()
                    story_loss /= story_len.float()#story level loss / story length 
                    story_loss = story_loss.sum()
                    story_loss.backward() 
                    optimizer.step_and_update_lr()
                    total_loss += story_loss.item()
                    story_loss = 0
                else:
                    raise ValueError('exceed length')
                loss = None    
                non_pad_mask = gold.ne(Constants.PAD)
                n_word = non_pad_mask.sum().item()
                n_word_total += n_word
                n_word_correct += n_correct
              
        if opt.loss_level == 'hierarchical':
            story_loss = 0
            if opt.is_story_discriminator:
                story_pred = torch.empty(tgt_gold.size(1), 0).long().to(device)
                story_gold = torch.empty(tgt_gold.size(1), 0).long().to(device)
            for i in range(length):
                # forward
                optimizer.zero_grad()                        
                gold_src = src_gold[i]
                gold = tgt_gold[i]
                #LSTM Modification
                pred = model(src_seq[i], src_pos[i], src_sen_pos[i], tgt_seq[i], tgt_pos[i], tgt_sen_pos[i], previous_tgt[:i+1], story_len)
                
                # backward
                loss, n_correct = cal_performance(pred, gold, 'story', smoothing=False)
                ###LSTM (Concatenation) Modification
                gold_zeros = torch.zeros(gold.size(0),1, device=gold.device, dtype = gold.dtype)
                gold = torch.cat((gold_zeros, gold),1)
                
                if opt.is_story_discriminator:
                    pred_idx = pred.max(1)[1].view(gold.size(0), gold.size(1))
                    story_pred = torch.cat((story_pred, pred_idx),1)   
                    story_gold = torch.cat((story_gold, gold), 1)   
                if i < length-1:
                    story_loss += loss.detach().cpu()#.item()
                elif i == length-1:
                    #addition and move loss to GPU + add require_grad
                    story_loss = loss + story_loss.to(device)#.item()
                    story_loss /= story_len.float()#story level loss / story length 
                    if opt.is_story_discriminator:
                        reward = get_reward(story_pred, story_gold, discriminator, reward_rate)
                        story_loss = story_loss * reward
                    story_loss = story_loss.sum()

                    story_loss.backward() 
                    optimizer.step_and_update_lr()
                    total_loss += story_loss.item()
                    story_loss = 0
                else:
                    raise ValueError('exceed length')
                loss = None    
                non_pad_mask = gold.ne(Constants.PAD)
                n_word = non_pad_mask.sum().item()
                n_word_total += n_word
                n_word_correct += n_correct
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy    

def pad_all_seqs(src_seq, src_pos, src_sen_pos, src_gold, tgt_seq, tgt_pos, tgt_sen_pos, tgt_gold, story_len, max_length, device):
    if len(src_seq) < max_length:
        diff = max_length - len(src_seq)
        src_pad_seq = torch.zeros(diff, src_seq.shape[1], src_seq.shape[2]).long().to(device)
        #This is for ground truth(frame) which will be a 0 tensor with story_len * batch size * frame_length
        src_pad_seq_clone = src_pad_seq.clone()
        #This is for rest of seq, seq_pos, seq_sen_pos to prevend outputing nan. 
        src_pad_seq[:,:,0] = 2

        src_seq = torch.cat((src_seq, src_pad_seq),0)
        src_pos = torch.cat((src_pos, src_pad_seq_clone),0)
        src_sen_pos = torch.cat((src_sen_pos, src_pad_seq_clone),0)
        src_gold = torch.cat((src_gold, src_pad_seq_clone),0)

        tgt_pad_seq = torch.zeros(diff, tgt_seq.shape[1], tgt_seq.shape[2]).long().to(device)
        #This is for ground truth(text) which will be a 0 tensor with story_len * batch size * text_length
        tgt_pad_seq_clone = tgt_pad_seq.clone()
        #This is for ground truth which will be a 0 tensor with story_len * batch size * frame_length
        tgt_pad_seq[:,:,0] = 2
    #             tgt_pad_seq_c[:,:,1] = 7

        tgt_seq = torch.cat((tgt_seq, tgt_pad_seq),0)
        tgt_pos = torch.cat((tgt_pos, tgt_pad_seq_clone),0)
        tgt_sen_pos = torch.cat((tgt_sen_pos, tgt_pad_seq_clone),0)
        tgt_gold = torch.cat((tgt_gold, tgt_pad_seq_clone),0)
    return src_seq, src_pos, src_sen_pos, src_gold, tgt_seq, tgt_pos, tgt_sen_pos, tgt_gold, story_len
    

def eval_epoch(model, validation_data, device, Dataloader, opt, discriminator):
    ''' Epoch operation in evaluation phase '''
    
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    sentence_max_len = opt.max_token_seq_len-1
    frame_max_len = opt.max_encode_token_seq_len-1
    
    number = 0
    story_loss = 0
    reward_rate = opt.reward_rate
    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            #LSTM Modification
            frame, frame_pos, frame_sen_pos, frame_gold, targets, targets_pos, targets_sen_pos, targets_gold, previous_targets, story_len = map(lambda x: x.to(device), batch) 
    
            
            pred = []
            length = max(story_len)
            for i in range(length):
                # forward
                gold = targets_gold[i]
                gold_frm = frame_gold[i]
                #first sentence goes with i+hop frame. 
                #For Example. In hop 2, sentence(1+2) goes with frame 3.

                #LSTM Modification
                pred = model(frame[i], frame_pos[i], frame_sen_pos[i], targets[i], targets_pos[i], targets_sen_pos[i], previous_targets[:i+1],story_len)
                
                # backward
                if opt.loss_level == 'sentence' or opt.loss_level == 'hierarchical':
                    if opt.is_sen_discriminator:
                        loss, n_correct = cal_performance(pred, gold, 'story', smoothing=False)
                        pred = pred.max(1)[1]
                        pred = pred.view(gold.size(0), gold.size(1))
                        reward = get_reward(pred, gold, discriminator)
                        loss = loss * reward
                        loss = loss.sum()
                    else:
                        loss, n_correct = cal_performance(pred, gold, 'batch', smoothing=False)
                        
                    total_loss += loss.item()
                    non_pad_mask = gold.ne(Constants.PAD)
                    n_word = non_pad_mask.sum().item()
                    n_word_total += n_word
                    n_word_correct += n_correct
                elif opt.loss_level == 'story':
                    loss, n_correct = cal_performance(pred, gold, 'story', smoothing=False)
                    if i < length-1:
                        story_loss += loss.detach().cpu()
                        loss = None
                        non_pad_mask = gold.ne(Constants.PAD)
                        n_word = non_pad_mask.sum().item()
                        n_word_total += n_word
                        n_word_correct += n_correct
                    elif i == length-1:
                        story_loss = loss + story_loss.to(device)
                        story_loss /= story_len.float() 
                        story_loss = story_loss.sum() 
                        total_loss += story_loss.item()
                        loss = None
                        story_loss = 0

                        non_pad_mask = gold.ne(Constants.PAD)
                        n_word = non_pad_mask.sum().item()
                        n_word_total += n_word
                        n_word_correct += n_correct
            if opt.loss_level == 'hierarchical':
                story_loss = 0
                if opt.is_story_discriminator:
                    story_pred = torch.empty(targets_gold.size(1), 0).long().to(device)
                    story_gold = torch.empty(targets_gold.size(1), 0).long().to(device)
                for i in range(length):
                    # forward
                    gold_src = frame_gold[i]
                    gold = targets_gold[i]
                    #LSTM Modification
                    pred = model(frame[i], frame_pos[i], frame_sen_pos[i], targets[i], targets_pos[i], targets_sen_pos[i], previous_targets[:i+1], story_len)

                    # backward
                    loss, n_correct = cal_performance(pred, gold, 'story', smoothing=False)
                    ###LSTM (Concatenation) Modification
                    gold_zeros = torch.zeros(gold.size(0),1, device=gold.device, dtype = gold.dtype)
                    gold = torch.cat((gold_zeros, gold),1)
                    if opt.is_story_discriminator:
                        pred_idx = pred.max(1)[1].view(gold.size(0), gold.size(1))
                        story_pred = torch.cat((story_pred, pred_idx),1)   
                        story_gold = torch.cat((story_gold, gold), 1)   
                    if i < length-1:
                        story_loss += loss.detach().cpu()#.item()
                    elif i == length-1:
                        #addition and move loss to GPU + add require_grad
                        story_loss = loss + story_loss.to(device)#.item()
                        story_loss /= story_len.float()#story level loss / story length 
                        if opt.is_story_discriminator:
                            reward = get_reward(story_pred, story_gold, discriminator, reward_rate)
                            story_loss = story_loss * reward
                        story_loss = story_loss.sum()

                        total_loss += story_loss.item()
                        story_loss = 0
                    else:
                        raise ValueError('exceed length')
                    loss = None    
                    non_pad_mask = gold.ne(Constants.PAD)
                    n_word = non_pad_mask.sum().item()
                    n_word_total += n_word
                    n_word_correct += n_correct
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy



    ''' Start training '''
def train(model, training_data, validation_data, vist_train_data, vist_val_data, optimizer, device, opt, Dataloader, discriminator):
    today = date.today()
    today_time = today.strftime("%b-%d-%Y")
    #early_stopping = EarlyStopping(patience=10, verbose=False) 
    log_train_file = None
    log_valid_file = None
    log_dir = "./log/roc_run_all"
    if opt.model != None: log_dir = log_dir + "_pretrain"
    if opt.vist: log_dir = log_dir + "_vist"

    if opt.log:
        log_train_file = log_dir+ str(opt.hop) + opt.log + 'Frame1' + '.train.log'
        log_valid_file = log_dir+ str(opt.hop) + opt.log + 'Frame1' + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time() 
        if opt.vist:
            train_loss, train_accu = train_vist_epoch(
                model, training_data, vist_train_data, optimizer, device, smoothing=opt.label_smoothing, Dataloader=Dataloader, opt=opt, discriminator=discriminator)
        else:
            train_loss, train_accu = train_epoch(
                model, training_data, optimizer, device, smoothing=opt.label_smoothing, Dataloader=Dataloader, opt=opt, discriminator=discriminator)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        if opt.vist:
            valid_loss, valid_accu = eval_epoch(model, vist_val_data, device, Dataloader, opt, discriminator)
        else:
            valid_loss, valid_accu = eval_epoch(model, validation_data, device, Dataloader, opt, discriminator)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]
        valid_losses += [valid_loss]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}
        
        
        if opt.save_model:
            save_dir = f"./save_model_BIO_{today_time}" + opt.log + str(opt.hop)
            save_dir = save_dir + f'_{opt.loss_level}'
            if opt.is_reverse: save_dir = save_dir + '_reverse'
            if opt.is_story_discriminator: save_dir = save_dir + '_story_dis'
            if opt.is_sen_discriminator: save_dir = save_dir + '_sen_dis'
            save_dir = save_dir + f'_reward_rate_{opt.reward_rate}'
                
            if opt.model != None: save_dir = save_dir + "_pretrain"
            if opt.vist: save_dir = save_dir + "_vist"
            save_dir = save_dir + "/"
            print('save_dir',save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if opt.save_mode == 'all':
                model_name = save_dir + opt.save_model + '_ppl_{ppl: 8.5f}.chkpt'.format(ppl=math.exp(min(valid_loss, 100)))
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = save_dir+opt.save_model + '.chkpt'
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))
                
        #early_stopping(valid_loss)        
        #if early_stopping.early_stop:
        #    print('epoch_i',epoch_i)
        #    break

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    #parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=64)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    #parser.add_argument('-d_model', type=int, default=512)
    #parser.add_argument('-d_inner_hid', type=int, default=2048)
    #parser.add_argument('-d_k', type=int, default=64)
    #parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=2)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-n_warmup_steps', type=int, default=2000)
    #parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-device', type=str, default='0')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-hop', type = float , required = True) 
    parser.add_argument('-vist', default=False, action='store_true')
    parser.add_argument('-vg', default=False, action='store_true')
    parser.add_argument('-loss_level', type=str, choices=['sentence', 'story', 'hierarchical'], default='sentence')
    parser.add_argument('-is_reverse', action='store_true')
    parser.add_argument('-is_sen_discriminator', action='store_true')
    parser.add_argument('-is_story_discriminator', action='store_true')
    parser.add_argument('-reward_rate', type=float, default=0.5)
    
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    print('opt.vg',opt.vg)
    print('opt.loss_level',opt.loss_level)
    print('opt.is_reverse',opt.is_reverse)
    #========= Loading Dataset =========#
    #data = torch.load(opt.data)
    #opt.max_token_seq_len = data['settings'].max_token_seq_len
#     opt.combine_loss = False
#     opt.is_reverse = False
    #1 = 1 hop + 1 Frame only
    #1.5 = 1 hop + all Frames
    #2 = 2 hops + 1 Frame only
    #2.5 = 2 hops + all Frames
    if opt.hop == 1:
        opt.max_encode_token_seq_len = 10*2+1
        opt.max_token_seq_len = 25*2
    elif opt.hop == 1.5:
        opt.max_encode_token_seq_len = 10*int(25)+1
        opt.max_token_seq_len = 25*2
    elif opt.hop == 2:
        opt.max_encode_token_seq_len = 10*2+1
        opt.max_token_seq_len = 25*3
    elif opt.hop == 2.5:
        opt.max_encode_token_seq_len = 10*int(25)+1
        opt.max_token_seq_len = 25*3
    else:
        raise ValueError('opt.hop invalid.')

    torch.manual_seed(1234)
    Dataloader = Loaders(opt)
    Dataloader.get_loaders(opt)
    training_data, validation_data, vist_train_data, vist_val_data,  = Dataloader.loader['train'], Dataloader.loader['val'], Dataloader.loader['vist_train'], Dataloader.loader['vist_val']

    opt.src_vocab_size = len(Dataloader.frame_vocab)
    opt.tgt_vocab_size = len(Dataloader.story_vocab)
    print('opt.src_vocab_size',opt.src_vocab_size)
    print('opt.tgt_vocab_size',opt.tgt_vocab_size)
    print('opt.hop',opt.hop)
    print('opt.is_sen_discriminator', opt.is_sen_discriminator)
    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)
    print('opt.model',opt.model)
    device = torch.device(f'cuda:{opt.device}' if opt.cuda else 'cpu')
    if opt.model:
        print('opt.device',opt.device)
        checkpoint = torch.load(opt.model, map_location=device)
        opt_ = checkpoint['settings']
        opt_.vist = opt.vist
        opt_.vg = opt.vg
        opt_.model=opt.model
        opt_.device=opt.device
        opt_.batch_size=opt.batch_size
        opt_.log=opt.log
        opt_.save_mode = opt.save_mode
        opt_.hop = opt.hop
        opt_.loss_level = opt.loss_level
        opt_.is_reverse = opt.is_reverse
        opt_.is_sen_discriminator = opt.is_sen_discriminator
        opt_.is_story_discriminator = opt.is_story_discriminator
        opt_.reward_rate = opt.reward_rate
        opt = opt_
        print(f"Load pretrain dic")

    
    print('device',device)
    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_encode_token_seq_len,
        opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    if opt.model:
        transformer.load_state_dict(checkpoint['model'])
        n_step = len(training_data)*checkpoint['epoch']
        print(f"N steps {n_step}")
        print(f"Loaded Pretrained Model: {opt.model}")

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09,lr=1e-4),
        opt.d_model, opt.n_warmup_steps)
    
    if opt.is_sen_discriminator or opt.is_story_discriminator:
        GEN_EMBEDDING_DIM = opt.d_model
        GEN_HIDDEN_DIM = opt.d_model
        DIS_EMBEDDING_DIM = opt.d_model
        DIS_HIDDEN_DIM = opt.d_model
        VOCAB_SIZE = opt.tgt_vocab_size
        MAX_SEQ_LEN = 200

        discriminator = discriminator_model.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, device)
        discriminator.load_state_dict(torch.load('discriminator/saved_model/model_minloss-3-b-B-ep100-emb512-1', map_location=device))
        discriminator = discriminator.to(device) 
    else:
        discriminator = None
    print('discriminator',discriminator)

    train(transformer, training_data, validation_data, vist_train_data, vist_val_data, optimizer, device, opt, Dataloader, discriminator)



if __name__ == '__main__':
    main()
