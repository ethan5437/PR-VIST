import torch
import torch.nn as nn
import numpy
import math
#from data_utilities import PerQuestionDataset, random_split, quick_collate, VISTDataset, VIST_TrainDataset
from data_utilities import random_split, quick_collate
from torch.utils.data import DataLoader 
import torch.nn as nn
from torch.optim import lr_scheduler
from utility import save_model, load_model 
import random
from datetime import datetime
import json
import Constants
from Train_Dataset_Path_Search import Train_Dataset_Path_Search
# from Recurrent_Path_Search import Recurrent_Path_Search
import os
import time
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
random.seed(1111)

total_rank = 0
rank_count = 0
total_all = 0

# 
# device = Constants.device

class VIST_Framework():
    def __init__(self, args, word2id, rela2id):
        self.loss_function = nn.MarginRankingLoss(margin=args.margin)
        self.args = args
        self.word2id = word2id
        self.rela2id = rela2id
        self.id2rela = {v:k for k,v in rela2id.items()}
        self.framework = args.framework
        self.dataset = args.dataset.lower()
        self.device = 'cuda:'+str(args.device)
        

    def _set_optimizer(self, model):
        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=self.args.learning_rate, weight_decay=self.args.l2_norm, amsgrad=True)
        if self.args.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params = filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=self.args.learning_rate, weight_decay=self.args.l2_norm)
        if self.args.optimizer == 'adadelta':
            self.optimizer = torch.optim.Adadelta(params = filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=self.args.learning_rate, weight_decay=self.args.l2_norm)
#        return optimizer

    def _eval_metric(self, scores):
#         print('scores',scores)
        pos_scores = scores[0].repeat(len(scores)-1)
        neg_scores = scores[1:]
#         print('neg_scores',neg_scores)
        ones = torch.ones(len(neg_scores)).to(self.device)
        loss = self.loss_function(pos_scores, neg_scores, ones)
        acc = 1 if all([x > y for x, y in zip(pos_scores, neg_scores)]) else 0
#         if torch.isnan(loss).any(): 
#             print('loss',loss)
#             print('scores',scores)
#             print('acc',acc)
#             raise ValueError('stop at _eval_metric')
        return loss, acc

    def _loss_weight(self, current_len, total_len, acc, task):
        hop_weight = self.args.hop_weight**(current_len)
        task_weight = self.args.task_weight if task=='TD' else 1
        acc_weight = self.args.acc_weight if acc==1 else 1
        return acc_weight / (hop_weight * task_weight)

    def _padding_cuda(self, seqs, maxlen, pad_type, padding, seq_position=None, start_position=None):
        pad_seq, mask, position, pad_seq_position = [], [], [], []
        for seq in seqs:
            if pad_type == 'append':
                pad_seq.append(seq + [padding]*(maxlen-len(seq)))
                mask.append([1]*len(seq) + [0]*(maxlen-len(seq)))
                if start_position != None:
                    position.append([i+start_position for i in range(len(seq))] + [0]*(maxlen-len(seq)))
                if seq_position != None:
                    pad_seq_position.append(seq_position+[padding]*(maxlen-len(seq)))

            elif pad_type == 'prepend':
                pad_seq.append([padding]*(maxlen-len(seq)) + seq)
                mask.append([0]*(maxlen-len(seq)) + [1]*len(seq))
                if start_position != None:
                    position.append([0]*(maxlen-len(seq)) + [i+start_position for i in range(len(seq))])
                if seq_position != None:
                    pad_seq_position.append([padding]*(maxlen-len(seq)) + seq_position)
                    
        if start_position == None and seq_position == None:
            return torch.LongTensor(pad_seq).to(self.device), torch.LongTensor(mask).to(self.device)
        if seq_position!=None:
            return torch.LongTensor(pad_seq).to(self.device), torch.LongTensor(mask).to(self.device),\
                    torch.LongTensor(pad_seq_position).to(self.device)
        
        return torch.LongTensor(pad_seq).to(self.device), torch.LongTensor(mask).to(self.device),\
                torch.LongTensor(position).to(self.device)
        
    def _single_UHop_step(self, model, ques, pos_tuples, neg_tuples, step_pos=None, sentence_num=None):
        pos_rela, pos_rela_text, pos_prev, pos_prev_text, _ = zip(*pos_tuples)
        neg_rela, neg_rela_text, neg_prev, neg_prev_text, _ = zip(*neg_tuples)
       
        # input of question
        
        if self.dataset == 'vist' and self.args.q_representation != 'bert':
            ques_pos = ques[1]
            ques = ques[0]
            if self.args.recurrent_UHop:
#                 print('_single_UHop_step -- sentence_num',sentence_num)
                ques_pos = [sentence_num for pos in ques]
#                 print('ques_pos',ques_pos)
            elif self.args.ethan_position:
#                 print('step_pos',step_pos)
#                 print('ques_pos',ques_pos)
                ques_pos = [1 if step_pos == pos else 0 for pos in ques_pos]
#                 print('ques_pos',ques_pos)
#                 error
            elif self.args.abs_position:
                pass
                
            ques, _, ques_pos = self._padding_cuda([ques]*(len(pos_rela)+len(neg_rela)), max(5, len(ques)), 'append', self.word2id['PADDING'], ques_pos)
                                
        elif self.args.q_representation == 'bert':
            ques, ques_mask = self._padding_cuda([ques]*(len(pos_rela)+len(neg_rela)), max(5, len(ques)), 'append', 0)
            ques = torch.cat([ques, ques_mask], dim=-1)
        else:
            ques, _ = self._padding_cuda([ques]*(len(pos_rela)+len(neg_rela)), max(5, len(ques)), 'append', self.word2id['PADDING'])
        # input of relation and previous
        if self.framework == 'baseline' or self.args.dynamic == 'none':
            # concat all previous and relation
            #separate relation and structural relation
            if self.args.is_seperate_structural:
                pos_relas, pos_relas_pos, pos_relas_text, pos_relas_text_pos = [], [], [], []
                for prev, rela, prev_text, rela_text in zip(pos_prev, pos_rela, pos_prev_text, pos_rela_text):
                    sub_rela, sub_rela_pos, sub_rela_text, sub_rela_text_pos = [], [], [], []
                    for p, text in zip(prev, prev_text):
#                         print(p,text)
                        sub_rela.extend(p[:-1])
                        sub_rela_pos.extend([p[-1]]*len(p[:-1]))
                        sub_rela_text.extend(text)
                        sub_rela_text_pos.extend([p[-1]]*len(text))
                    sub_rela.extend(rela[:-1])
                    sub_rela_pos.extend([rela[-1]]*len(rela[:-1]))
                    sub_rela_text.extend(rela_text)
                    sub_rela_text_pos.extend([rela[-1]]*len(rela_text))                
                    pos_relas.append(sub_rela)
                    pos_relas_pos.append(sub_rela_pos)
                    pos_relas_text.append(sub_rela_text)
                    pos_relas_text_pos.append(sub_rela_text_pos)                           
                neg_relas, neg_relas_pos, neg_relas_text, neg_relas_text_pos = [], [], [], []
                for prev, rela, prev_text, rela_text in zip(neg_prev, neg_rela, neg_prev_text, neg_rela_text):
                    sub_rela, sub_rela_pos, sub_rela_text, sub_rela_text_pos = [], [], [], []
                    for p, text in zip(prev, prev_text):
                        sub_rela.extend(p[:-1])
                        sub_rela_pos.extend([p[-1]]*len(p[:-1]))
                        sub_rela_text.extend(text)
                        sub_rela_text_pos.extend([p[-1]]*len(text))
                    sub_rela.extend(rela[:-1])
                    sub_rela_pos.extend([rela[-1]]*len(rela[:-1]))
                    sub_rela_text.extend(rela_text)
                    sub_rela_text_pos.extend([rela[-1]]*len(rela_text))  
                    neg_relas.append(sub_rela)
                    neg_relas_pos.append(sub_rela_pos)
                    neg_relas_text.append(sub_rela_text)
                    neg_relas_text_pos.append(sub_rela_text_pos)
                maxlen = max([len(rela) for rela in pos_relas+neg_relas])
                maxlen = max([len(rela) for rela in pos_relas_text+neg_relas_text])
                relas, _ = self._padding_cuda(pos_relas+neg_relas, maxlen, 'append', self.rela2id['PADDING'])
                relas_pos, _ = self._padding_cuda(pos_relas_pos+neg_relas_pos, maxlen, 'append', self.rela2id['PADDING'])
                relas_text, _ = self._padding_cuda(pos_relas_text+neg_relas_text, maxlen, 'append', self.word2id['PADDING'])
                relas_text_pos, _ = self._padding_cuda(pos_relas_text_pos+neg_relas_text_pos, maxlen, 'append', self.word2id['PADDING'])
                prevs, prevs_text = [], []    
            else:
                pos_relas = [sum(prev+[rela], []) for prev, rela in zip(pos_prev, pos_rela)]
                neg_relas = [sum(prev+[rela], []) for prev, rela in zip(neg_prev, neg_rela)]
                if self.args.constant_padding:
                    maxlen = self.args.rela_size
                    pos_relas = [rela[(len(rela)-maxlen):] for rela in pos_relas]
                    neg_relas = [rela[(len(rela)-maxlen):] for rela in neg_relas]
                else:
                    maxlen = max([len(rela) for rela in pos_relas+neg_relas])
                
                if self.args.prepend:
                    relas, _ = self._padding_cuda(pos_relas+neg_relas, maxlen, 'prepend', self.rela2id['PADDING'])
                else:
                    relas, _ = self._padding_cuda(pos_relas+neg_relas, maxlen, 'append', self.rela2id['PADDING'])
                pos_relas_text = [sum(prev+[rela], []) for prev, rela in zip(pos_prev_text, pos_rela_text)]
                neg_relas_text = [sum(prev+[rela], []) for prev, rela in zip(neg_prev_text, neg_rela_text)]
                if self.args.constant_padding:
                    maxlen = self.args.rela_text_size
                    pos_relas_text = [rela[(len(rela)-maxlen):] for rela in pos_relas_text]
                    neg_relas_text = [rela[(len(rela)-maxlen):] for rela in neg_relas_text]
                else:
                    maxlen = max([len(rela) for rela in pos_relas_text+neg_relas_text])
                    
                if self.args.prepend:
                    relas_text, _ = self._padding_cuda(pos_relas_text+neg_relas_text, maxlen, 'prepend', self.word2id['PADDING'])
                else:
                    relas_text, _ = self._padding_cuda(pos_relas_text+neg_relas_text, maxlen, 'append', self.word2id['PADDING'])
                prevs, prevs_text = [], []
        else:
            maxlen = max([len(rela) for rela in pos_rela+neg_rela])
            relas, _ = self._padding_cuda(pos_rela+neg_rela, maxlen, 'append', self.rela2id['PADDING'])
            maxlen = max([len(rela) for rela in pos_rela_text+neg_rela_text])
            relas_text, _ = self._padding_cuda(pos_rela_text+neg_rela_text, maxlen, 'append', self.word2id['PADDING'])
            if self.args.dynamic == 'flatten':
                # concat all previous
                prevs = [sum(prev, []) for prev in pos_prev+neg_prev]
                maxlen = max([len(prev) for prev in prevs])
                if maxlen > 0:
                    prevs, _ = self._padding_cuda(prevs, maxlen, 'append', self.rela2id['PADDING'])
                    prevs = [prevs]
                else:
                    prevs = []
                prevs_text = [sum(prev, []) for prev in pos_prev+neg_prev]
                maxlen = max([len(prev) for prev in prevs_text])
                if maxlen > 0:
                    prevs_text, _ = self._padding_cuda(prevs_text, maxlen, 'append', self.word2id['PADDING'])
                    prevs_text = [prevs_text]
                else:
                    prevs_text = []
            elif self.args.dynamic == 'recurrent':
                # make every candidates have same steps of previous
                maxlen = max([len(prev) for prev in pos_prev+neg_prev])
                prevs = [prev+[[]]*(maxlen-len(prev)) for prev in pos_prev+neg_prev]
                prevs_text = [prev+[[]]*(maxlen-len(prev)) for prev in pos_prev_text+neg_prev_text]
                # pad every previous respectively
                prevs = [self._padding_cuda(prev, max([len(p) for p in prev]), 'append', self.rela2id['PADDING'])[0] for prev in zip(*prevs)]
                prevs_text = [self._padding_cuda(prev, max([len(p) for p in prev]), 'append', self.word2id['PADDING'])[0] for prev in zip(*prevs_text)]
                
        if self.dataset == 'vist':
            if self.args.is_seperate_structural:
                score = model(ques, relas_text, relas, prevs_text, prevs, ques_pos, relas_text_pos, relas_pos)
            else:
                score = model(ques, relas_text, relas, prevs_text, prevs, ques_pos)
        else:
            score = model(ques, relas_text, relas, prevs_text, prevs)
        loss, acc = self._eval_metric(score)
        
        # readible format for score of all candidates : [(score, [token1, token2 ... ]) ... ]
        #rev_rela = [self.tokenizer.convert_ids_to_tokens(rela) for rela in pos_concat_rela+neg_concat_rela]
        rev_rela = [[self.id2rela[r] for r in rela] for rela in pos_rela+neg_rela]
        rela_score=list(zip(score.detach().cpu().numpy().tolist()[:], rev_rela[:]))
        return loss, acc, rela_score

    def _single_hop_rela_choose(self, model, ques, tuples, next_tuples, movement, sentence_num, step_pos=None):
        if movement == 'continue':
            pos_tuples = [t for t in next_tuples if t[-1] == 1]
            neg_tuples = [t for t in tuples if t[-1] == 1] + [t for t in nex_tuples if t[-1] == 0]
        elif movement == 'terminate':
            pos_tuples = [t for t in tuples if t[-1] == 1]
            neg_tuples = [t for t in next_tuples]
        else:
            raise ValueError(f'Unknown movement:{movement} in UHop._single_hop_rela_choose')
        # special case
        if len(pos_tuples) == 0 or len(neg_tuples) == 0:
            raise ValueError('noNegativeInTD')
#             return 0, 1, 'noNegativeInTD+RC'
        if len(pos_tuples) > 1:
            print('mutiple positive tuples!')
        if len(neg_tuples) > self.args.neg_sample:
            neg_tuples = neg_tuples[:self.args.neg_sample]
        # run model
        loss, acc, rela_score = self._single_UHop_step(model, ques, pos_tuples, neg_tuples, step_pos, sentence_num)
        return loss, acc, rela_score

    def _single_step_rela_choose(self, model, ques, tuples, sample_size, sentence_num, step_pos=None):
        # make +/- pairs
        pos_tuples = [t for t in tuples if t[-1] == 1]
        neg_tuples = [t for t in tuples if t[-1] == 0]
        # special case
        if len(pos_tuples) == 0 or len(neg_tuples) == 0:
            raise ValueError('noNegativeInTD')
        if len(pos_tuples) > 1:
            print('mutiple positive tuples!')
        # run model
        if sample_size <=1:
            sample_size = 1
            
        if sample_size >= len(neg_tuples):
            sample_size = len(neg_tuples)
        neg_tuples = random.sample(neg_tuples, sample_size)
        loss, acc, rela_score = self._single_UHop_step(model, ques, pos_tuples, neg_tuples, step_pos, sentence_num)
        return loss, acc, rela_score, sample_size
        

    def _termination_decision(self, model, ques, tuples, next_tuples, movement, sample_size, sentence_num, step_pos=None):
        # make +/- pairs
        if movement == 'continue':
            pos_tuples = [t for t in next_tuples if t[-1] == 1]
            neg_tuples = [t for t in tuples if t[-1] == 1] 
        elif movement == 'terminate':
            pos_tuples = [t for t in tuples if t[-1] == 1]
            neg_tuples = [t for t in next_tuples if t[-1] == 0]
        else:
            raise ValueError(f'Unknown movement:{movement} in UHop._termination_decision')
        # special case
        if len(pos_tuples) == 0 or len(neg_tuples) == 0:
            raise ValueError('noNegativeInTD')
        if len(pos_tuples) > 1:
            print('mutiple positive tuples!')

        if sample_size >= len(neg_tuples):
            sample_size = len(neg_tuples)
        neg_tuples = random.sample(neg_tuples, sample_size)
        loss, acc, rela_score = self._single_UHop_step(model, ques, pos_tuples, neg_tuples, step_pos, sentence_num)
        return loss, acc, rela_score


    def _execute_UHop(self, model, data, mode):
        index, ques, step_list_ = data
        
        step_list = step_list_[0]
        step_position_list = step_list_[1]
        sentence_num = None
        if self.args.recurrent_UHop:
            sentence_num = step_list_[2]

        loss = torch.tensor(0, dtype=torch.float, requires_grad=True).to(self.device)
        loss_count, step_count = 0, 0
        acc_list = []
        if mode == 'train':
            self.optimizer.zero_grad();model.zero_grad()
        step_count = 0
        rc_acc, rc_count = 0, 0
        td_acc, td_count = 0, 0
        labels, scores = [], []
        
        sample_size_list = []
        sample_iter_list = [sample for sample in range(self.args.termset_sample_size, 0, -20)] + [1]
        
        shrinked = False
                
        for i in range(len(step_list)-1):  
            # TD if not the first nor the last step : continue 
            if i > 0:
                #shrink negative sample for fitting into the GPU
                for sample_iter in sample_iter_list:
                    try:
                        step_loss, acc, score = self._termination_decision(model, ques, step_list[i-1], step_list[i], 'continue', sample_iter, sentence_num, step_position_list[i])
                        if step_loss != 0:
                            step_loss *= self._loss_weight(i, len(step_list)-2, acc, 'TD')
                            if self.args.step_every_step and mode == 'train':
                                step_loss.backward(); self.optimizer.step()
                                self.optimizer.zero_grad();model.zero_grad();
                        break
                    except KeyboardInterrupt: 
                        raise ValueError('KeyboardInterrupt')
                    except RuntimeError as e:
                        torch.cuda.empty_cache()
                        shrinked = True
                        step_loss = 0 #Free the loss backward
                        if sample_iter == sample_iter_list[-2]:
                            print(f'One negative sample is used_1')
                        if sample_iter == sample_iter_list[-1]:
                            raise ValueError('Out of GPU Memory_1')
                        pass

                if step_loss != 0:
                    loss = loss + step_loss.item()
                    step_count += 1
                loss_count += 1
                acc_list.append(acc)
                td_acc += acc; td_count += 1
                labels.append('<C>' if acc else '<T>')
                scores.append(score)
                if self.args.stop_when_err and acc != 1:
                    break
            # single_step RC
            #shrink negative sample for fitting into the GPU
#             t1 = time.time()
            for sample_iter in sample_iter_list:
#                 print('_single_step_rela_choose sample_iter',sample_iter)
                try:
                    step_loss, acc, score, sample_size = self._single_step_rela_choose(model, ques, step_list[i], sample_iter, sentence_num, step_position_list[i])
#                     print('acc_rela_choose',acc)
                    if step_loss != 0:
                        step_loss *= self._loss_weight(i, len(step_list)-2, acc, 'RC')
                        if self.args.step_every_step and mode == 'train':
                            step_loss.backward(); self.optimizer.step()
                            self.optimizer.zero_grad();model.zero_grad();
                    break
                except KeyboardInterrupt: 
                    print('Keyboard Interrupted')
                except RuntimeError as e:
                    torch.cuda.empty_cache()
                    shrinked = True
                    step_loss = 0
                    if sample_iter == sample_iter_list[-2]:
                        print(f'One negative sample is used_2')
                    if sample_iter == sample_iter_list[-1]:
                        raise ValueError('Out of GPU Memory_2')
                    else:
                        pass
#                         print(f'Exception: {e}')
#                         raise ValueError("random sample didn't work")
            sample_size_list.append(sample_size)            
            if step_loss != 0:
                loss = loss + step_loss.item()
                step_count += 1
            loss_count += 1
            acc_list.append(acc)
            rc_acc += acc; rc_count += 1
            labels.append('<CR>' if acc else '<WR>')
            scores.append(score)
            if self.args.stop_when_err and acc != 1:
                break
        # last TD : terminate
        if (not self.args.stop_when_err) or all([x==1 for x in acc_list]):
            #shrink negative sample for fitting into the GPU
            for sample_iter in sample_iter_list:
                try:
                    step_loss, acc, score = self._termination_decision(model, ques, step_list[-2], step_list[-1], 'terminate', sample_iter, sentence_num, step_position_list[i])
                    if step_loss != 0:
                        step_loss *= self._loss_weight(i, len(step_list)-2, acc, 'TD')
                        if self.args.step_every_step and mode == 'train':
                            step_loss.backward(); self.optimizer.step()
                            self.optimizer.zero_grad();model.zero_grad();
                    break
                except KeyboardInterrupt: 
                    print('Keyboard Interrupted')
                except RuntimeError as e:
                    torch.cuda.empty_cache()
                    shrinked = True
                    step_loss = 0
                    if sample_iter == sample_iter_list[-2]:
                        print(f'One negative sample is used_3')
                    if sample_iter == sample_iter_list[-1]:
                        raise ValueError(f'Out of GPU Memory_3')
                        #raise ValueError("random sample didn't work")
                    pass
            if step_loss != 0:
                loss = loss + step_loss.item()
                step_count += 1
            loss_count += 1
            acc_list.append(acc)
            td_acc += acc; td_count += 1
            labels.append('<T>' if acc else '<C>')
            scores.append(score)
        # step if not step_every_step
        acc = 1 if all([x==1 for x in acc_list]) else 0
        if mode == 'train' and not self.args.step_every_step:
            loss /= (step_count if step_count > 0 else 1)
            loss.backward(); self.optimizer.step()
        if len(sample_size_list) == 0:
            print('step_list',step_list)
        return model, (loss, loss_count), acc, scores, '\t'.join(labels), (rc_acc, rc_count), (td_acc, td_count), shrinked, sum(sample_size_list)/len(sample_size_list)

    def train(self, model, args):
        # prepare dataset
        path_train_data = Train_Dataset_Path_Search(self.args, 'train',self.word2id, self.rela2id)
        path_valid_data = Train_Dataset_Path_Search(self.args, 'valid', self.word2id, self.rela2id)
        
        train_dataset = DataLoader(dataset=path_train_data, batch_size=1, shuffle=True, num_workers=0, 
                pin_memory=False, collate_fn=quick_collate)
        valid_dataset = DataLoader(dataset=path_valid_data, batch_size=1, shuffle=True, num_workers=0, 
                pin_memory=False, collate_fn=quick_collate)
        
        self._set_optimizer(model)
        earlystop_counter, min_valid_metric = 0, 100
        # training
        shrinked = 0
        
        data_tmp = ''
        print('\n')
        for epoch in range(0, self.args.epoch_num):
            model = model.train().to(self.device)
            total_loss, total_acc = 0.0, 0.0
            loss_count, acc_count = 0, 0
            total_rc_acc, total_td_acc = 0.0, 0.0
            rc_count, td_count = 0, 0
            avg_sizes = []
            total_count = 0
            for trained_num, path_datas in enumerate(train_dataset):
                if args.recurrent_UHop:
                    for path_data in path_datas:
                        total_count +=1
                        data = path_data#path_train_data._numericalize(path_data, 'UHop', 'vist')
                        model, loss, acc, score, label, rc_acc, td_acc, is_shrinked, avg_size = self._execute_UHop(model, data, 'train')
                        avg_sizes.append(avg_size)
                        avg_avg_size = int(sum(avg_sizes)/len(avg_sizes))
                        if is_shrinked:
                            shrinked+=1
                        total_loss += loss[0].data; loss_count += loss[1]
                        total_acc += acc; acc_count += 1
                        total_rc_acc += rc_acc[0]; rc_count += rc_acc[1]
                        total_td_acc += td_acc[0]; td_count += td_acc[1]
                        print(f'\r{self.args.framework}_{self.args.model}({self.args.dynamic}) {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Epoch {epoch} {total_count}/{data_length} Loss:{total_loss/loss_count:.5f} Acc:{total_acc/acc_count:.4f} RC_Acc:{total_rc_acc/rc_count:.2f} TD_Acc:{total_td_acc/td_count:.2f} Shrinked: {shrinked}, avg_sizes: {avg_avg_size}', end='')
#                     error
                else:
                    total_count +=1
                    path_data = path_datas
                    data = path_data#path_train_data._numericalize(path_data, 'UHop', 'vist')
                    model, loss, acc, score, label, rc_acc, td_acc, is_shrinked,avg_size = self._execute_UHop(model, data, 'train')
                    avg_sizes.append(avg_size)
                    avg_avg_size = int(sum(avg_sizes)/len(avg_sizes))
                    if is_shrinked:
                        shrinked+=1
                    total_loss += loss[0].data; loss_count += loss[1]
                    total_acc += acc; acc_count += 1
                    total_rc_acc += rc_acc[0]; rc_count += rc_acc[1]
                    total_td_acc += td_acc[0]; td_count += td_acc[1]
                    print(f'\r{self.args.framework}_{self.args.model}({self.args.dynamic}) {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Epoch {epoch} {trained_num}/{len(train_dataset)} Loss:{total_loss/loss_count:.5f} Acc:{total_acc/acc_count:.4f} RC_Acc:{total_rc_acc/rc_count:.2f} TD_Acc:{total_td_acc/td_count:.2f} Shrinked: {shrinked}, avg_sizes: {avg_avg_size}', end='')
    
            # validation for examing if early stop
            valid_loss, valid_acc, valid_score, _ = self.evaluate(model, 'valid', valid_dataset, args)
            if valid_loss < min_valid_metric:
                min_valid_metric = valid_loss
                earlystop_counter = 0
                save_model(model, self.args.path)
            else:
                earlystop_counter += 1
            if earlystop_counter > self.args.earlystop_tolerance:
                break
        return model

    def evaluate(self, model, mode, dataset, args, output_result=False):
        if model == None:
            model = self.args.Model(self.args).to(self.device)
            model = load_model(model, self.args)
        model = model.eval().to(self.device)
        if dataset == None:
            path_test_data = Train_Dataset_Path_Search(self.args, mode, self.word2id, self.rela2id)
            dataset = DataLoader(dataset=path_test_data, batch_size=1, shuffle=False, num_workers=0, 
                pin_memory=False, collate_fn=quick_collate)

        total_loss, total_acc, total_rc_acc, total_td_acc = 0.0, 0.0, 0.0, 0.0
        loss_count, acc_count, rc_count, td_count = 0, 0, 0, 0
        labels, scores = [], []
        total_count, passed = 0, 0
        with torch.no_grad():
            model = model.eval()
            acc_list = []
            for num, path_datas in enumerate(dataset):
                total_count +=1                
                if args.recurrent_UHop:
                    for path_data in path_datas:      
                        data = path_data#dataset._numericalize(path_data, 'UHop', 'vist')
                        model, loss, acc, score, label, rc_acc, td_acc, _, _ = self._execute_UHop(model, data, mode)

                        total_loss += loss[0].data; loss_count += loss[1]
                        total_acc += acc; acc_count += 1; acc_list.append(acc)
                        total_rc_acc += rc_acc[0]; rc_count += rc_acc[1]
                        total_td_acc += td_acc[0]; td_count += td_acc[1]
                        labels.append(label)
                        scores.append(score)
                else:
                    path_data = path_datas
                    data = path_data#dataset._numericalize(path_data, 'UHop', 'vist')
                    model, loss, acc, score, label, rc_acc, td_acc, _, _ = self._execute_UHop(model, data, mode)

                    total_loss += loss[0].data; loss_count += loss[1]
                    total_acc += acc; acc_count += 1; acc_list.append(acc)
                    total_rc_acc += rc_acc[0]; rc_count += rc_acc[1]
                    total_td_acc += td_acc[0]; td_count += td_acc[1]
                    labels.append(label)
                    print(f'\r{self.args.framework}_{self.args.model}({self.args.dynamic}) {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {num}/{len(dataset)} Loss:{total_loss/loss_count:.5f} Acc:{total_acc/acc_count:.4f} RC_Acc:{total_rc_acc/rc_count:.2f} TD_Acc:{total_td_acc/td_count:.2f}', end='')
                scores.append(score)
#                 break
        if self.args.framework == 'baseline':
            print(f' Eval {num} Loss:{total_loss/loss_count:.5f} Acc:{total_acc/acc_count:.4f}', end='')
        else:
            print(f' Eval {num} Loss:{total_loss/loss_count:.5f} Acc:{total_acc/acc_count:.4f} RC_Acc:{total_rc_acc/rc_count:.2f} TD_Acc:{total_td_acc/td_count:.2f}', end='')
        print('')
        #dump result
        if output_result:
            with open(f'{self.args.path}/scores_{100*total_acc/acc_count:.2f}.json', 'w') as f:
                json.dump(scores, f)
            with open(f'{self.args.path}/prediction.txt', 'w') as f:
                f.write('\n'.join(labels))
        return total_loss/loss_count, total_acc/acc_count, scores, labels
