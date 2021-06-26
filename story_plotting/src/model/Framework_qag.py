import torch
import torch.nn as nn
import numpy
import math
from scipy.special import softmax
from net import n_gram,n_gram_qg, qg_performance,language_model_scorer,co_qg_Loss
from data_utilities_UHop import PerQuestionDataset, random_split, quick_collate
from torch.utils.data import DataLoader 
import torch.nn as nn
#from torch.optim import lr_scheduler
from UHop_utility import save_model,save_qg_model,load_qg_model, load_model ,list_parameter_require_grad,save_qalm_model,save_qglm_model,load_lm_model
import random
from datetime import datetime
import json
#from pytorch_pretrained_bert import BertTokenizer
#from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from tool import collate_fn
import transformer.Constants as Constants
from transformer.Model import Transformer
from transformer.Question_Generator_both import QuestionGenerator
#from dataset import QGDataset, paired_collate_fn

total_rank = 0
rank_count = 0
total_all = 0
#######QG  tool###########
'''
def paired_collate_fn(insts):
    src_insts, tgt_insts = list(zip(*insts))
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts)
'''
def QG_preprocess_question(data,max_sent_len):
  try:
    qdata=data.lower()
    qdata=qdata.split(' ')
  except:
    qdata=data
  
  if len(qdata) > max_sent_len:
      qdata=qdata[:max_sent_len]
  qdata=[Constants.BOS_WORD] + qdata + [Constants.EOS_WORD]
  return qdata
######################

class Framework():
    def __init__(self, args, word2id, rela2id):

        self.loss_function = nn.MarginRankingLoss(margin= args.margin)
        self.bce = nn.BCELoss()
        self.args = args
        self.word2id = word2id
        self.rela2id = rela2id
        self.id2rela = {v:k for k,v in rela2id.items()}
        #self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert)#for no module
        self.framework = args.framework
        ####################################
        QALM_CONTEXT_SIZE = 2  
        QALM_EMBEDDING_DIM = 300 
        self.qa_lang_model = n_gram(len(word2id), QALM_CONTEXT_SIZE, QALM_EMBEDDING_DIM,True).cuda()
        if args.load_lm_pretrain:
          if args.dataset=='nq_qaqg':
            self.qa_lang_model=load_lm_model(self.qa_lang_model, 'qalm_model.pth','../QA50KE/ABWIM_701')
          else:
            self.qa_lang_model=load_lm_model(self.qa_lang_model, 'qalm_model.pth','../QA50KE/ABWIM_WQ_24')
        self.qa_lang_optimizer = optim.SGD(params = filter(lambda p: p.requires_grad,  self.qa_lang_model.parameters() ), lr=1e-2, weight_decay=1e-9)
        ######################################
        self.cal_qg_performance=qg_performance()
        self.call_qg_Loss=co_qg_Loss()
        self.language_model_score=language_model_scorer()
        ####################################
        if args.dataset=='wq_qaqg':
          with open('../data/WQ_QAG/src_rela_word_2glove_avg_idx_tgt_src_word2idx.json', 'r') as fwp:
            dict_data=json.load(fwp)
          self.src_word2idx=dict_data['src']
          self.tgt_word2idx=dict_data['tgt']
          self.src_relaid2gloveavg=dict_data['src_relaid2gloveavg']
          self.src_2_wordlevel_seq_mid_rel=dict_data['src_2_wordlevel_seq_mid_rel']
          src_2_wordlevel_length=dict_data['src_2_wordlevel_length']
        elif args.dataset=='nq_qaqg':
          with open('../data/NQ_QAG/src_rela_word_2glove_avg_idx_tgt_src_word2idx.json', 'r') as fwp:
            dict_data=json.load(fwp)
          self.src_word2idx=dict_data['src']
          self.tgt_word2idx=dict_data['tgt']
          self.src_relaid2gloveavg=dict_data['src_relaid2gloveavg']
          self.src_2_wordlevel_seq_mid_rel=dict_data['src_2_wordlevel_seq_mid_rel']
          src_2_wordlevel_length=dict_data['src_2_wordlevel_length']
        self.tgt_idx2word = {idx:word for word, idx in self.tgt_word2idx.items()}
        ######################################
        QGLM_CONTEXT_SIZE = 2  
        QGLM_EMBEDDING_DIM = 300
        self.qg_lang_model = n_gram(len(word2id), QGLM_CONTEXT_SIZE, QGLM_EMBEDDING_DIM,True).cuda()
        if args.load_lm_pretrain:
          if args.dataset=='nq_qaqg':
            self.qg_lang_model=load_lm_model(self.qg_lang_model, 'qglm_model.pth','../QA50KE/ABWIM_701')
          else:
            self.qg_lang_model=load_lm_model(self.qg_lang_model, 'qglm_model.pth','../QA50KE/ABWIM_WQ_24')
        #self.qg_lang_model = n_gram_qg(src_2_wordlevel_length, QGLM_CONTEXT_SIZE, QGLM_EMBEDDING_DIM).cuda()
        self.qg_lang_optimizer = optim.SGD(params = filter(lambda p: p.requires_grad,  self.qg_lang_model.parameters() ), lr=1e-2, weight_decay=1e-9)
        # optim.Adam(filter(lambda x: x.requires_grad,self.qg_lang_model.parameters() ),betas=(0.9, 0.98), eps=1e-09,lr=1e-5) 
        self.max_src_word_seq_len=dict_data['max_src_word_seq_len']
        self.max_tgt_word_seq_len=dict_data['max_tgt_word_seq_len']        
        ######################################
        transformer = Transformer(args,
            len(dict_data['src']),
            len(dict_data['tgt']),
            dict_data['max_src_word_seq_len'],
            dict_data['max_tgt_word_seq_len'],
            relaid2gloveavgid=dict_data['src_relaid2gloveavg'],
            tgt_emb_prj_weight_sharing=None,
            emb_src_tgt_weight_sharing=None,
            d_k=args.d_k,
            d_v=args.d_v,
            d_model=args.d_model,
            d_word_vec=args.d_word_vec,
            d_inner=args.d_inner_hid,
            n_layers=args.n_layers,
            n_head=args.n_head,
            dropout=args.qg_dropout_rate).cuda()
        self.qg_model=transformer
         
        if self.args.tanh_basis:
          print('tanh_basis is turned on')
        ######################################
    def get_qg_model(self):
        return self.qg_model
    def convert_seq_to_idx_seq(self, seq, word2idx):
        ''' Mapping seq words to idx sequence. batech size 1'''
        return [[word2idx.get(w, Constants.UNK) for w in seq]]        
        
#################################################
    
    def train_sentence(self,model,sentence):
        #train_loss=torch.zeros(1).cuda()
        
        words, labels = sentence
        #print(words, labels)
        preds = model(words)
        loss=self.cal_qg_performance(preds, labels)
        confidence=self.language_model_score( model, sentence)
        #print('loss=',loss)

        return loss[0],confidence
       


    def trigram(self, test_sentence):
      return [( (test_sentence[i], test_sentence[i+1]), [test_sentence[i+2]]) for i in range(len(test_sentence)-2)]

    
    def Uhop_score_relation_path(self, oscores,tanh_method=False):
        pp=[]
        a=0
        gold_score_list=[]
        #cur_max=0.0
        k=0    
        f=self.args.tanh_power 
        if self.args.tanh_basis:
          basis=(1.0-(np.tanh(f).astype(float)+1)/2)    
        else:
          basis=0.0        
        for i,_scores in enumerate(oscores[:-1]):
          if type(_scores)!= str :
            if _scores.shape[0]!=1:
            
              scores=_scores-torch.min(_scores)                
              gold_score_list.append(scores[0])
            
              pp.append(torch.log(  torch.softmax(scores,-1)[0]+0.000000001))

            
        #neg_highest_tail=0
        #if  oscores[-1]!='noNegativeInTD+RC':
        #    neg_highest_tail=torch.max(oscores[-1])
        for i,p in enumerate(pp):
          if i==0:
            confid=p#*(sm_score_weight[i])
          else:
            if  tanh_method:
              if i%2==1:
                factor=gold_score_list[i]/(gold_score_list[i-1]+gold_score_list[i])                    
                thred=(torch.tanh(2*f*factor-f)+1.0)/2#+basis
                confid+=(p+torch.log(thred)) 
              else:
                confid+=p   
            else:
              confid+=p
          a+=1
        #if  neg_highest_tail!=0:
        #  confid+=torch.log(gold_score_list[-1]/(gold_score_list[-1]+neg_highest_tail) )              
        confid=torch.exp(confid/a)
        if len(pp)==0:
          return torch.ones(1).cuda()
        return confid  
          
    '''
    def Uhop_score_relation_path(self, oscores):
        pp=[]
        a=0
        gold_score_list=[]
        #cur_max=0.0
        k=0    
        f=self.args.tanh_power 
        if self.args.tanh_basis:
          basis=(1.0-(np.tanh(f).astype(float)+1)/2)    
        else:
          basis=0.0        
        for i,_scores in enumerate(oscores[:-1]):
          if type(_scores)!= str :
            if _scores.shape[0]!=1:
            
              scores=_scores-torch.min(_scores)                
              gold_score_list.append(scores[0])
            
              pp.append(torch.log(  torch.softmax(scores,-1)[0]+0.000000001))

            
        #neg_highest_tail=0
        #if  oscores[-1]!='noNegativeInTD+RC':
        #    neg_highest_tail=torch.max(oscores[-1])
        for i,p in enumerate(pp):
          if i==0:
            confid=p#*(sm_score_weight[i])
          else:
            factor=gold_score_list[i]/(gold_score_list[i-1]+gold_score_list[i])  
                  
            thred=(torch.tanh(2*f*factor-f)+1.0)/2#+basis
            confid+=(p+torch.log(thred))   
          a+=1
        #if  neg_highest_tail!=0:
        #  confid+=torch.log(gold_score_list[-1]/(gold_score_list[-1]+neg_highest_tail) )              
        confid=torch.exp(confid/a)
        if len(pp)==0:
          return torch.ones(1).cuda()
        return confid  
    '''        
#################################################        
    def _set_optimizer(self, model):
        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params =  list_parameter_require_grad(model), 
                    lr=self.args.learning_rate, weight_decay=self.args.l2_norm, amsgrad=True)
        if self.args.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params =  list_parameter_require_grad(model) , 
                    lr=self.args.learning_rate, weight_decay=self.args.l2_norm)
        if self.args.optimizer == 'adadelta':
            self.optimizer = torch.optim.Adadelta(params =  list_parameter_require_grad(model), 
                    lr=self.args.learning_rate, weight_decay=self.args.l2_norm)
#        return optimizer

    def _eval_metric(self, scores):
        pos_scores = scores[0].repeat(len(scores)-1)
        neg_scores = scores[1:]
        ones = torch.ones(len(neg_scores)).cuda()
        loss = self.loss_function(pos_scores, neg_scores, ones)
        acc = 1 if all([x > y for x, y in zip(pos_scores, neg_scores)]) else 0
        return loss, acc

    def _loss_weight(self, current_len, total_len, acc, task):
        hop_weight = self.args.hop_weight**(current_len)
        task_weight = self.args.task_weight if task=='TD' else 1
        acc_weight = self.args.acc_weight if acc==1 else 1
        return acc_weight / (hop_weight * task_weight)
    def _padding(self, lists, maxlen, type, padding):
        new_lists = []
        for list in lists:
            if type == 'prepend':
                new_list = [padding] * (maxlen - len(list)) + list
            elif type == 'append':
                new_list = list + [padding] * (maxlen - len(list))
            new_lists.append(new_list)
        return new_lists
    def _padding_cuda(self, seqs, maxlen, pad_type, padding, start_position=None):
        pad_seq, mask, position = [], [], []
        for seq in seqs:
            if pad_type == 'append':
                pad_seq.append(seq + [padding]*(maxlen-len(seq)))
                mask.append([1]*len(seq) + [0]*(maxlen-len(seq)))
                if start_position != None:
                    position.append([i+start_position for i in range(len(seq))] + [0]*(maxlen-len(seq)))
            elif pad_type == 'prepend':
                pad_seq.append([padding]*(maxlen-len(seq)) + seq)
                mask.append([0]*(maxlen-len(seq)) + [1]*len(seq))
                if start_position != None:
                    position.append([0]*(maxlen-len(seq)) + [i+start_position for i in range(len(seq))])
        if start_position == None:
            return torch.LongTensor(pad_seq).cuda(), torch.LongTensor(mask).cuda()
        return torch.LongTensor(pad_seq).cuda(), torch.LongTensor(mask).cuda(), torch.LongTensor(position).cuda()

    def _single_UHop_step(self, model, ques, pos_tuples, neg_tuples):
        if self.args.model=='ABWIM_WQ' and self.args.dynamic == 'none':
          pos_rela, pos_rela_text, _ = zip(*pos_tuples)
          neg_rela, neg_rela_text, _ = zip(*neg_tuples)
        else:
          pos_rela, pos_rela_text, pos_prev, pos_prev_text, _ = zip(*pos_tuples)
          neg_rela, neg_rela_text, neg_prev, neg_prev_text, _ = zip(*neg_tuples)

        # input of question
        if self.args.q_representation == 'bert':
            ques, ques_mask = self._padding_cuda([ques]*(len(pos_rela)+len(neg_rela)), max(5, len(ques)), 'append', 0)
            ques = torch.cat([ques, ques_mask], dim=-1)
        else:
          if self.args.model=='ABWIM_WQ' and self.args.dynamic == 'none':
            pass
          else:            
            ques, _ = self._padding_cuda([ques]*(len(pos_rela)+len(neg_rela)), max(5, len(ques)), 'append', self.word2id['PADDING'])
        # input of relation and previous
        if self.framework == 'baseline' or self.args.dynamic == 'none':
        
          if self.args.model=='ABWIM_WQ' and self.args.dynamic == 'none':
            # MAYBE WRITE A FUNCTION TO CREATE PADDED TENSOR ?
            maxlen = max([len(x) for x in pos_rela+neg_rela])
            pos_rela = self._padding(pos_rela, maxlen, 'prepend', self.rela2id['PADDING'])
            #print(pos_rela)
            neg_rela = self._padding(neg_rela, maxlen, 'prepend', self.rela2id['PADDING'])
            maxlen = max([len(x) for x in pos_rela_text+neg_rela_text])
            pos_rela_text = self._padding(pos_rela_text, maxlen, 'prepend', self.word2id['PADDING'])
            #print(pos_rela_text)
            neg_rela_text = self._padding(neg_rela_text, maxlen, 'prepend', self.word2id['PADDING'])
            ques = torch.LongTensor([ques]*(len(pos_rela)+len(neg_rela))).cuda()
            relas = torch.LongTensor(pos_rela+neg_rela).cuda()
            relas_text = torch.LongTensor(pos_rela_text+neg_rela_text).cuda()
          else:
            # concat all previous and relation
            pos_relas = [sum(prev+[rela], []) for prev, rela in zip(pos_prev, pos_rela)]
            neg_relas = [sum(prev+[rela], []) for prev, rela in zip(neg_prev, neg_rela)]
            maxlen = max([len(rela) for rela in pos_relas+neg_relas])
            relas, _ = self._padding_cuda(pos_relas+neg_relas, maxlen, 'append', self.rela2id['PADDING'])
            pos_relas_text = [sum(prev+[rela], []) for prev, rela in zip(pos_prev_text, pos_rela_text)]
            neg_relas_text = [sum(prev+[rela], []) for prev, rela in zip(neg_prev_text, neg_rela_text)]
            maxlen = max([len(rela) for rela in pos_relas_text+neg_relas_text])
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
        if self.args.model=='ABWIM_WQ' and self.args.dynamic == 'none':
          score = model(ques, relas_text, relas)
          
        else:
          score = model(ques, relas_text, relas, prevs_text, prevs)
        loss, acc = self._eval_metric(score)
        # readible format for score of all candidates : [(score, [token1, token2 ... ]) ... ]
        #rev_rela = [self.tokenizer.convert_ids_to_tokens(rela) for rela in pos_concat_rela+neg_concat_rela]
        rev_rela = [[self.id2rela[r] for r in rela] for rela in pos_rela+neg_rela]
        rela_score=list(zip(score.detach().cpu().numpy().tolist()[:], rev_rela[:]))
        return loss, acc, rela_score,score

    def _single_step_rela_choose(self, model, ques, tuples):
        # make +/- pairs
        pos_tuples = [t for t in tuples if t[-1] == 1]
        neg_tuples = [t for t in tuples if t[-1] == 0]
        # special case
        if len(pos_tuples) == 0 or len(neg_tuples) == 0:
            return 0, 1, 'noNegativeInRC'
        if len(pos_tuples) > 1:
            print('mutiple positive tuples!')
        if len(neg_tuples) > self.args.neg_sample:
          if self.args.model=='ABWIM_WQ' and self.args.dynamic == 'none':
            neg_tuples = neg_tuples[:self.args.neg_sample]
          else:
            neg_tuples = random.sample(neg_tuples, self.args.neg_sample)#neg_tuples[:self.args.neg_sample]
        # run model
        loss, acc, rela_score,score_tensor = self._single_UHop_step(model, ques, pos_tuples, neg_tuples)
        return loss, acc, score_tensor#rela_score

    def _termination_decision(self, model, ques, tuples, next_tuples, movement):
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
            return 0, 1, 'noNegativeInTD'
        if len(pos_tuples) > 1:
            print('mutiple positive tuples!')
        if len(neg_tuples) > self.args.neg_sample:
            neg_tuples = neg_tuples[:self.args.neg_sample]
        # run model
        loss, acc, rela_score,score_tensor = self._single_UHop_step(model, ques, pos_tuples, neg_tuples)
        return loss, acc, score_tensor#rela_score

    def _execute_combined_UHop(self, model, data, mode, every_step_update_mode):
        index, ques, step_list = data
        loss = torch.tensor(0, dtype=torch.float, requires_grad=True).cuda()
        loss_count, step_count = 0, 0
        acc_list = []
        if mode == 'train':
            self.optimizer.zero_grad();model.zero_grad()
        step_count = 0
        rc_acc, rc_count = 0, 0
        td_acc, td_count = 0, 0
        labels, scores = [], []
        for i in range(len(step_list)-1):
            # TD if not the first nor the last step : continue
            if i > 0:
                step_loss, acc, score = self._termination_decision(model, ques, step_list[i-1], step_list[i], 'continue')
                if step_loss != 0:
                    step_loss *= self._loss_weight(i, len(step_list)-2, acc, 'TD')
                    if every_step_update_mode and mode == 'train':
                        step_loss.backward(retain_graph=True); self.optimizer.step()
                        self.optimizer.zero_grad();model.zero_grad();
                    loss = loss + step_loss
                    step_count += 1
                loss_count += 1
                acc_list.append(acc)
                td_acc += acc; td_count += 1
                labels.append('<C>' if acc else '<T>')
                scores.append(score)
                if self.args.stop_when_err and acc != 1:
                    break
            # single_step RC
            step_loss, acc, score = self._single_step_rela_choose(model, ques, step_list[i])
            if step_loss != 0:
                step_loss *= self._loss_weight(i, len(step_list)-2, acc, 'RC')
                if every_step_update_mode and mode == 'train':
                    step_loss.backward(retain_graph=True); self.optimizer.step()
                    self.optimizer.zero_grad();model.zero_grad();
                loss = loss + step_loss
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
            step_loss, acc, score = self._termination_decision(model, ques, step_list[-2], step_list[-1], 'terminate')
            if step_loss != 0:
                step_loss *= self._loss_weight(i, len(step_list)-2, acc, 'TD')
                if every_step_update_mode and mode == 'train':
                    step_loss.backward(retain_graph=True); self.optimizer.step()
                    self.optimizer.zero_grad();model.zero_grad();
                loss = loss + step_loss
                step_count += 1
            loss_count += 1
            acc_list.append(acc)
            td_acc += acc; td_count += 1
            labels.append('<T>' if acc else '<C>')
            scores.append(score)
        # step if not step_every_step
        acc = 1 if all([x==1 for x in acc_list]) else 0
        if mode == 'train' and not every_step_update_mode:
            loss /= (step_count if step_count > 0 else 1)
            #loss.backward(retain_graph=True); self.optimizer.step()
            #self.optimizer.zero_grad();model.zero_grad();
        return model, (loss, loss_count), acc, scores, '\t'.join(labels), (rc_acc, rc_count), (td_acc, td_count)
                    
    def train(self, model, up_path=None):
        #every_step_update_mode=True
        print('model info')
        print(model)
        # prepare dataset
        dataset = PerQuestionDataset(self.args, 'train', self.word2id, self.rela2id)
        if self.args.dataset.lower() == 'wq' or self.args.dataset.lower() == 'wq_train1test2' or self.args.dataset.lower() == 'wq_qaqg':
            train_dataset, valid_dataset = random_split(dataset, 0.9, 0.1)
        else:
            train_dataset = dataset
            valid_dataset = PerQuestionDataset(self.args, 'valid', self.word2id, self.rela2id)
        datas = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=18, 
                pin_memory=False, collate_fn=quick_collate)
        if up_path:
          if self.args.dataset=='nq_qaqg':
            self.qg_model = load_qg_model(self.qg_model, up_path)
          else:
            self.qg_model = load_qg_model(self.qg_model, up_path)
        self.qg_optimizer =  optim.Adam(filter(lambda x: x.requires_grad, self.qg_model.parameters() ),betas=(0.9, 0.98), eps=1e-10,lr=1e-5)#   lr=5e-6
        self._set_optimizer(model)
        max_QG_valid_acc_metric=0.0
        earlystop_counter, min_valid_metric = 0,100
        ########################################################################
        # preprocess for qg and both side lm data:
        self.qg_model.train()
        #print(self.src_2_wordlevel_seq_mid_rel)
        qg_train_data_list=[]   
        qa_lm_train_data_list=[]
        qg_lm_train_data_list=[]
        for trained_num, data in enumerate(datas):
          
          #print(seq)
          if len(data[1])>2:
            seq=self.trigram(data[1])
          else:
            seq=self.trigram([data[1][0]]+[17]+[data[1][1]])
            print(seq)
          words  = torch.LongTensor(np.array([[word ] for word, label in seq])).cuda() 
          labels = torch.LongTensor(np.array([label for word, label in seq])).cuda()
          que_seq_wordlevel=(words,labels.squeeze(-1))        

          qa_lm_train_data_list.append(que_seq_wordlevel)
          #################################
          # this part is for QGLM preprocess for training data (only contain relations)
          '''
          seq=self.trigram(data[5])

          words  = torch.LongTensor(np.array([[word ] for word, label in seq])).cuda() 
          labels = torch.LongTensor(np.array([label for word, label in seq])).cuda()
          src_seq_wordlevel=(words,labels.squeeze(-1))               
          qg_lm_train_data_list.append(src_seq_wordlevel)
          '''
          for sub_id in range(len(data[4])):   
              # this part is for QG_preprocess for training data
              train_tgt_insts = self.convert_seq_to_idx_seq( QG_preprocess_question(data[3],self.max_tgt_word_seq_len), self.tgt_word2idx)                     
              train_src_insts = self.convert_seq_to_idx_seq( QG_preprocess_question( data[4][sub_id],self.max_src_word_seq_len),  self.src_word2idx)
              
              # this part is for QGLM preprocess for training data (contain mid and relations)
              seq=[]   
              for termid in train_src_insts[0]:          
                seq+=self.src_2_wordlevel_seq_mid_rel[str(termid)] 
              seq=self.trigram(seq)

              words  = torch.LongTensor(np.array([[word ] for word, label in seq])).cuda() 
              labels = torch.LongTensor(np.array([label for word, label in seq])).cuda()
              src_seq_wordlevel=(words,labels.squeeze(-1))  
                          
              #############
              src_seq, src_pos=collate_fn(train_src_insts)
              tgt_seq, tgt_pos=collate_fn(train_tgt_insts)
              batch=(src_seq, src_pos, tgt_seq, tgt_pos)
              src_seq, src_pos, tgt_seq, tgt_pos  = map(lambda x: x.cuda(), batch)
              qg_train_data_list.append((src_seq, src_pos, tgt_seq, tgt_pos ,src_seq_wordlevel))
        qg_valid_data_list=[]
        valid_datas = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=18, 
                        pin_memory=False, collate_fn=quick_collate)
        for valid_num, data in enumerate(valid_datas):
          for sub_id in range(len(data[4])): 
              valid_tgt_insts =  self.convert_seq_to_idx_seq( QG_preprocess_question(data[3],self.max_tgt_word_seq_len), self.tgt_word2idx)                     
              valid_src_insts = self.convert_seq_to_idx_seq( QG_preprocess_question( data[4][sub_id],self.max_src_word_seq_len),  self.src_word2idx)           
              #############
              src_seq, src_pos=collate_fn(valid_src_insts)
              tgt_seq, tgt_pos=collate_fn(valid_tgt_insts)
              batch=(src_seq, src_pos, tgt_seq, tgt_pos)
              src_seq, src_pos, tgt_seq, tgt_pos  = map(lambda x: x.cuda(), batch)
              qg_valid_data_list.append((src_seq, src_pos, tgt_seq, tgt_pos))
        ########################################################################
        print('training start')                
        # training
        switch=  self.args.switch
        QG_valid_loss, QG_valid_acc = self.eval_qg( self.qg_model, qg_valid_data_list)
        
        for epoch in range(0, self.args.epoch_num):
            model = model.train().cuda()
            total_loss, total_acc = 0.0, 0.0
            loss_count, acc_count = 0, 0
            total_rc_acc, total_td_acc = 0.0, 0.0
            rc_count, td_count = 0, 0
            qgtotal_loss = 0
            qgtotallm_loss =0
            qatotallm_loss =0
            n_word_total = 0
            total_qa_lm_num_gram=0
            total_qg_lm_num_gram=0
            n_word_correct = 0
            qg_id=0
            total_confidence=0.0
            statistics=[0,0,0,0]
            alpha=0
            for trained_num, data in enumerate(datas):
              for sub_id in range(len(data[4])):       
                if self.args.framework == 'baseline':
                    #ques, tuples = data[:3]
                    self.optimizer.zero_grad(); model.zero_grad(); 
                    loss, acc, score = self._single_step_rela_choose(model, ques, tuples)
                    if loss != 0:
                        loss.backward(); self.optimizer.step()
                    total_loss += (loss.data if loss!=0 else 0); loss_count += 1
                    total_acc += acc; acc_count += 1
                    print(f'\r{self.args.framework}_{self.args.model}({self.args.dynamic}) {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Epoch {epoch}  Sub_source {sub_id}{trained_num}/{len(datas)} Loss:{total_loss/loss_count:.5f} Acc:{total_acc/acc_count:.4f}', end='')
                else:
                    #######################################################
                    src_seq, src_pos, tgt_seq, tgt_pos, src_seq_wordlevel=qg_train_data_list[qg_id]#
                    #src_seq, src_pos, tgt_seq, tgt_pos =qg_train_data_list[qg_id]
                    qg_id+=1
                    gold = tgt_seq[:, 1:]
                    non_pad_mask = gold.ne(Constants.PAD)
                    n_word = non_pad_mask.sum().item()
                    n_word_total += n_word
                    
                    ##########################language model############################

                    
                    if epoch<switch:
                      #with torch.no_grad():
                      #  qglm_loss,qgconfidence=self.train_sentence(self.qg_lang_model,src_seq_wordlevel)#  qg_lm_train_data_list[trained_num]
                      qglm_loss_per_word=0.9487
                      qgconfidence=0.5566
                    else:
                      total_qg_lm_num_gram+=len(src_seq_wordlevel[1])#qg_lm_train_data_list[trained_num][1]
                      self.qg_lang_optimizer.zero_grad();  self.qg_lang_model.zero_grad();
                      self.qa_lang_optimizer.zero_grad();  self.qa_lang_model.zero_grad();
                      qglm_loss,qgconfidence=self.train_sentence(self.qg_lang_model,src_seq_wordlevel)#  qg_lm_train_data_list[trained_num]
                      qglm_loss.backward()
                      self.qg_lang_optimizer.step()
                      qgconfidence_data=qgconfidence.data.item() 
  
                      qgtotallm_loss+=qglm_loss.data
                      qglm_loss_per_word =qgtotallm_loss/total_qg_lm_num_gram
                    #######################################################

                    
                    
                    if epoch<switch:
                      #with torch.no_grad():
                      #  qalm_loss,qaconfidence=self.train_sentence(self.qa_lang_model,qa_lm_train_data_list[trained_num])
                      qalm_loss_per_word=0.9487 
                      qaconfidence=0.5566
                    else:
                      total_qa_lm_num_gram+=len(qa_lm_train_data_list[trained_num][1])
                      self.qg_lang_optimizer.zero_grad();  self.qg_lang_model.zero_grad();
                      self.qa_lang_optimizer.zero_grad();  self.qa_lang_model.zero_grad();
                      qalm_loss,qaconfidence=self.train_sentence(self.qa_lang_model,qa_lm_train_data_list[trained_num])
                      qalm_loss.backward()
                      self.qa_lang_optimizer.step()
                      qaconfidence_data=qaconfidence.data.item()  
  
                      qatotallm_loss+=qalm_loss.data
                      qalm_loss_per_word =qatotallm_loss/total_qa_lm_num_gram
                    
                    #######################################################
                    
                    if epoch<switch: 
                      

                      
                      ########################QA QG#########################
                      
                      self.qg_optimizer.zero_grad()
                      self.qg_model.zero_grad()
                      pred = self.qg_model(src_seq, src_pos, tgt_seq, tgt_pos)
                      qg_loss, n_correct = self.cal_qg_performance(pred, gold)
                      n_word_correct += n_correct
                      qg_loss.backward()
                      self.qg_optimizer.step()
                      qgtotal_loss += qg_loss.data
                      loss_per_word = qgtotal_loss/n_word_total
                      accuracy = n_word_correct/n_word_total 
                      
                      #accuracy=0.9487;loss_per_word =0.9487;
                      #######################################################
                      #if sub_id==0:
                      #with torch.no_grad():#update in the end so we need to freeze grad
                      #  model, loss, acc, score, label, rc_acc, td_acc = self._execute_combined_UHop(model, data[:3], 'train',False)
                      #  Uhop_cond=self.Uhop_score_relation_path(score)
                      #model, loss, acc, score, label, rc_acc, td_acc = self._execute_combined_UHop(model, data[:3], 'train',True)
                      #Uhop_cond=self.Uhop_score_relation_path(score)
                      #Uhop_cond_val=Uhop_cond.data.item()
                      #total_confidence+=Uhop_cond_val
                      ###########
                      Uhop_cond_val=0.9487;acc_count += 1;loss_count += 1;rc_count += 1;td_count += 1;
                      ########################
                      #total_loss += loss[0].data; loss_count += loss[1]
                      #total_acc += acc; acc_count += 1
                      #total_rc_acc += rc_acc[0]; rc_count += rc_acc[1]
                      #total_td_acc += td_acc[0]; td_count += td_acc[1]
                      #######################################################
                      print(f'\r{self.args.model} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Epoch {epoch} Sub_source {sub_id} {trained_num}/{len(datas)} Loss:{total_loss/loss_count:.5f} avg_cond:{total_confidence/(trained_num+1):.4f} cond:{Uhop_cond_val:.2f}Acc:{total_acc/acc_count:.4f} RC_Acc:{total_rc_acc/rc_count:.2f} TD_Acc:{total_td_acc/td_count:.2f} qalm_L:{qalm_loss_per_word:.2f} qalm_c:{qaconfidence:.5f}||qg_Acc:{accuracy:.2f}  qg_L:{loss_per_word:.2f} qglm_L:{qglm_loss_per_word:.2f} qglm_c:{qgconfidence:.5f} ', end='')
                    else:
                      same=0
                      a,b,c,d=0,0,0,0
                      ########################old QA(UHOP+)#########################
                      #self.qg_lang_optimizer.zero_grad();  self.qg_lang_model.zero_grad();
                      #self.qa_lang_optimizer.zero_grad();  self.qa_lang_model.zero_grad();
                      
                      ###main
                      #self.optimizer.zero_grad(); model.zero_grad();
                      
                      #model, loss_, acc_, score_, label_, rc_acc_, td_acc_ = self._execute_combined_UHop(model, data[:3], 'train',True)    
                      #use the score part to update in dual loss
                      #self.optimizer.zero_grad(); model.zero_grad();
                      ##############################################################
                      #if epoch>=switch2:#update in the end so we need to freeze grad
                      if self.args.model=='ABWIM_WQ' and self.args.dynamic == 'none':
                        model, loss, acc, score, label, rc_acc, td_acc = self._execute_combined_UHop(model, data[:3], 'train',False) 
                        loss[0].backward(retain_graph=True); self.optimizer.step()
                        self.optimizer.zero_grad();model.zero_grad();
                      else:
                        model, loss, acc, score, label, rc_acc, td_acc = self._execute_combined_UHop(model, data[:3], 'train',True)                       
                      Uhop_cond=self.Uhop_score_relation_path(score)

                      Uhop_cond_val=Uhop_cond.data.item()
                      total_confidence+=Uhop_cond_val 
                      
                      

                      with torch.no_grad():#update in the end so we need to freeze grad
                        pred = self.qg_model(src_seq, src_pos, tgt_seq, tgt_pos)
                        tokens = [ self.tgt_idx2word[idxtgt] for idxtgt in pred.max(1)[1].cpu().numpy().tolist()[:-1]]
                        pred_words_uid=[self.word2id[x] if x in self.word2id else self.word2id['<unk>'] for x in tokens]#2=<s>

                        #gold_words_uid=[self.word2id[x] if x in self.word2id else self.word2id['<unk>']  for x in  data[1]]
                        if self.args.model=='ABWIM_WQ' and self.args.dynamic == 'none':
                          if len(pred_words_uid) < 5:
                            pred_words_uid = pred_words_uid + [self.word2id['PADDING']] * (5-len(pred_words_uid))        
                        if pred_words_uid[0]==data[1][0]:
                          same=1                         
                        #print( '')
                        #print( 'now_QA','vs','pred_QA')
                        #print( data[1],'vs',pred_words_uid)
                        model, laloss, laacc, lascore, lalabel, larc_acc, latd_acc = self._execute_combined_UHop(model, [data[0],pred_words_uid,data[2]], 'train',False) 
                        laUhop_cond=self.Uhop_score_relation_path(lascore)
                        laUhop_cond_val=laUhop_cond.data.item()
                        if acc and laacc:
                          a=1
                          statistics[0]+=1
                        if acc and not laacc:
                          b=1
                          statistics[1]+=1
                        if not acc and laacc:
                          c=1
                          statistics[2]+=1
                        if not acc and not laacc:
                          d=1
                          statistics[3]+=1
                      #if trained_num%100==0:
                      #  print(statistics)


                        
                        #loss_[0].backward( ); self.optimizer.step()
                        #self.optimizer.zero_grad();model.zero_grad();
                      ###dual revise qa
                      if d==0:#a==1 or c==1:# or d==1:
                        with torch.no_grad():#update in the end so we need to freeze grad
                          pred = self.qg_model(src_seq, src_pos, tgt_seq, tgt_pos)
                          qg_loss, n_correct = self.cal_qg_performance(pred, gold)
                          qg_NLL_loss = self.call_qg_Loss(pred, gold)
                        dual_loss=torch.pow(torch.log(qgconfidence)+qg_NLL_loss-torch.log(qaconfidence)-torch.log(Uhop_cond),2)
                        qad_loss=dual_loss#+loss[0]
                        ###
                        qad_loss.backward(retain_graph=True)
                      ####self.optimizer.step()  
                     
                      total_loss += loss[0].data; loss_count += loss[1]
                      total_acc += acc; acc_count += 1
                      total_rc_acc += rc_acc[0]; rc_count += rc_acc[1]
                      total_td_acc += td_acc[0]; td_count += td_acc[1]
                      
                      ########################QG#############################
                      #self.qg_lang_optimizer.zero_grad();  self.qg_lang_model.zero_grad();
                      #self.qa_lang_optimizer.zero_grad();  self.qa_lang_model.zero_grad();
                      self.qg_optimizer.zero_grad();self.qg_model.zero_grad();
                      ###main
                      pred = self.qg_model(src_seq, src_pos, tgt_seq, tgt_pos)
                      #print(tgt_seq)

                      qg_loss, n_correct = self.cal_qg_performance(pred, gold)
                      qg_NLL_loss = self.call_qg_Loss(pred, gold)
                      n_word_correct += n_correct
                      ###dual revise qg
                      if d==0:#a==1 or b==1:# or d==1:
                        
                        with torch.no_grad():#update in the end so we need to freeze grad
                          model, loss, acc, score, label, rc_acc, td_acc = self._execute_combined_UHop(model, data[:3], 'train',False)
                          Uhop_cond=self.Uhop_score_relation_path(score)
                        dual_loss=torch.pow(torch.log(qgconfidence)+qg_NLL_loss-torch.log(qaconfidence)-torch.log(Uhop_cond),2)
                        qgd_loss=dual_loss+qg_loss
                        ###
                      else:
                        qgd_loss= qg_loss
                      qgd_loss.backward()
                        #print('8787878787878787')
                      
                      #####self.qg_optimizer.step()
                      qgtotal_loss += qg_loss.data
                      loss_per_word = qgtotal_loss/n_word_total
                      accuracy = n_word_correct/n_word_total 
                      #######################################################
                      self.optimizer.step() ;self.qg_optimizer.step();
                      ###########use potential qg high quality que to train qa##############
                      
                      #if (a==1 or c==1) and 
                      if laUhop_cond_val>( Uhop_cond_val) and same==1 and d==0: 
                        
                          if self.args.model=='ABWIM_WQ' and self.args.dynamic == 'none':
                            model, loss, acc, score, label, rc_acc, td_acc = self._execute_combined_UHop(model, [data[0],pred_words_uid,data[2]], 'train',False) 
                            loss[0].backward(); self.optimizer.step()
                            self.optimizer.zero_grad();model.zero_grad();
                          else:
                            model, loss_, acc_, score_, label_, rc_acc_, td_acc_ = self._execute_combined_UHop(model,[data[0],pred_words_uid,data[2]], 'train',True)
                          alpha+=1
                      #######################################################
                      print(f'\r{ self.args.model} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}dept Epoch {epoch} Sub_source {sub_id} {trained_num}/{len(datas)} Loss:{total_loss/loss_count:.5f} avg_cond:{total_confidence/(trained_num+1):.4f} cond:{Uhop_cond_val:.2f}Acc:{total_acc/acc_count:.4f} RC_Acc:{total_rc_acc/rc_count:.2f} TD_Acc:{total_td_acc/td_count:.2f} qalm_L:{qalm_loss_per_word:.2f} qalm_c:{qaconfidence_data:.5f}||qg_Acc:{accuracy:.2f}  qg_L:{loss_per_word:.2f} qglm_L:{qglm_loss_per_word:.2f} qglm_c:{qgconfidence_data:.5f} four p {statistics} rev{alpha}', end='')
            # validation for examing if early stop
            valid_loss, valid_acc, valid_score, _ = self.evaluate(model, 'valid', valid_dataset)
            QG_valid_loss, QG_valid_acc = self.eval_qg(  self.qg_model, qg_valid_data_list)
            #QG_valid_ppl = math.exp(min(QG_valid_loss, 100))
            #print('QG valid perplexity=',QG_valid_ppl)
            if QG_valid_acc > max_QG_valid_acc_metric:
              max_QG_valid_acc_metric=QG_valid_acc
              save_qg_model(self.qg_model,self.args.path)
            if epoch==switch:
              if switch!=0:
                min_valid_metric=100
                earlystop_counter=0
                max_QG_valid_acc_metric=0.0
                save_qalm_model(self.qa_lang_model,self.args.path)
                save_qglm_model(self.qg_lang_model,self.args.path)

            if valid_loss < min_valid_metric:
              min_valid_metric = valid_loss
              earlystop_counter = 0
              save_model(model, self.args.path)
            else:
              if epoch>switch :
                earlystop_counter += 1      
            if earlystop_counter > self.args.earlystop_tolerance:
              break 
            #if epoch==5:
            #  break                   
        return model

    def evaluate(self, model, mode, dataset, output_result=False):
        if model == None:
            model = self.args.Model(self.args).cuda()
            model = load_model(model, self.args.path)
        model = model.eval().cuda()
        if dataset == None:
            dataset = PerQuestionDataset(self.args, mode, self.word2id, self.rela2id)
        datas = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=12, 
                pin_memory=False, collate_fn=quick_collate)
        total_loss, total_acc, total_rc_acc, total_td_acc = 0.0, 0.0, 0.0, 0.0
        loss_count, acc_count, rc_count, td_count = 0, 0, 0, 0
        labels, scores = [], []
        with torch.no_grad():
            model = model.eval()
            acc_list = []
            for num, data in enumerate(datas):
                if self.args.framework == 'baseline':
                    # baseline is equivalent to single step relation choose
                    index, ques, tuples, ques_origin, qg_answer_source_rp,qg_answer_source_rp_only_rel  = data
                    loss, acc, score = self._single_step_rela_choose(model, ques, tuples)
                    total_loss += (loss.data if loss!=0 else 0); loss_count += 1
                    total_acc += acc; acc_count += 1
                    labels.append('<O>' if acc else '<X>')
                else:
                    model, loss, acc, score, label, rc_acc, td_acc = self._execute_combined_UHop(model, data[:3], mode,False)
                    total_loss += loss[0].data; loss_count += loss[1]
                    total_acc += acc; acc_count += 1; acc_list.append(acc)
                    total_rc_acc += rc_acc[0]; rc_count += rc_acc[1]
                    total_td_acc += td_acc[0]; td_count += td_acc[1]
                    labels.append(label)
            
                scores.append([s.detach().cpu().numpy().tolist()[:] if type(s)!= str else 0.0 for s in score])
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

    def eval_qg(self, model, qg_valid_data_list=None,valid_dataset=None):
        if not (qg_valid_data_list):
          if not(validation_data):
              print("w/o no dataset, qg model cannot evaluate its power QAQ")
              return 0
          qg_valid_data_list=[]
          datas = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=18, 
                          pin_memory=False, collate_fn=quick_collate)
          for trained_num, data in enumerate(datas):
            for sub_id in range(len(data[4])):   
                train_tgt_insts = self.convert_seq_to_idx_seq( QG_preprocess_question(data[3],self.max_tgt_word_seq_len), self.tgt_word2idx)                     
                train_src_insts = self.convert_seq_to_idx_seq( QG_preprocess_question( data[4][sub_id],self.max_src_word_seq_len),  self.src_word2idx)        
                #############
                src_seq, src_pos=collate_fn(train_src_insts)
                tgt_seq, tgt_pos=collate_fn(train_tgt_insts)
                batch=(src_seq, src_pos, tgt_seq, tgt_pos)
                src_seq, src_pos, tgt_seq, tgt_pos  = map(lambda x: x.cuda(), batch)
                qg_valid_data_list.append((src_seq, src_pos, tgt_seq, tgt_pos))
        model.eval()
    
        total_loss = 0
        n_word_total = 0
        n_word_correct = 0
    
        with torch.no_grad():
            for data in qg_valid_data_list:
                src_seq, src_pos, tgt_seq, tgt_pos=data
                gold = tgt_seq[:, 1:]
    
                # forward
                pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
                loss, n_correct = self.cal_qg_performance(pred, gold)
    
                # note keeping
                total_loss += loss.item()
    
                non_pad_mask = gold.ne(Constants.PAD)
                n_word = non_pad_mask.sum().item()
                n_word_total += n_word
                n_word_correct += n_correct
    
        loss_per_word = total_loss/n_word_total
        accuracy = n_word_correct/n_word_total
        print(f' QG Eval {len(qg_valid_data_list)} qg_L:{loss_per_word:.2f} qg_Acc:{accuracy:.2f} ')
        return loss_per_word, accuracy
    def test_generate_qg(self, model=None,up_path=None):
      qg_test_data_list=[]
      test_dataset = PerQuestionDataset(self.args, 'test', self.word2id, self.rela2id)
      datas = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=18, 
                      pin_memory=False, collate_fn=quick_collate)
      k=0
      for test_num, data in enumerate(datas):
        for sub_id in range(len(data[4])):   
            test_tgt_insts = self.convert_seq_to_idx_seq( QG_preprocess_question(data[3],self.max_tgt_word_seq_len), self.tgt_word2idx)                     
            test_src_insts = self.convert_seq_to_idx_seq( QG_preprocess_question( data[4][sub_id],self.max_src_word_seq_len),  self.src_word2idx)          
            #############
            src_seq, src_pos=collate_fn(test_src_insts)
            tgt_seq, tgt_pos=collate_fn(test_tgt_insts)
            batch=(src_seq, src_pos, tgt_seq, tgt_pos)
            src_seq, src_pos, tgt_seq, tgt_pos  = map(lambda x: x.cuda(), batch)
            qg_test_data_list.append((src_seq, src_pos, tgt_seq, tgt_pos))
            k+=1
      if model:
        q_gen = QuestionGenerator(model,relas_dict_glove_avg_dataset=self.args.dataset)#,entid2gloveavgid
      else:
        model = load_qg_model(self.qg_model, up_path)      
        q_gen = QuestionGenerator(model,relas_dict_glove_avg_dataset=self.args.dataset)
      tgt_idx2word = {idx:word for word, idx in self.tgt_word2idx.items()}
      import os
      if not self.args.path:
        path = os.path.join(up_path, 'qg_text.txt')
      else:
        path = os.path.join( self.args.path, 'qg_text.txt')
      with open(path, 'w') as f:
          #for batch in qg_test_data_list:
          i=0
          for test_num, data in enumerate(datas):
            for sub_id in range(len(data[4])):
              src_seq, src_pos, tgt_seq, tgt_pos=qg_test_data_list[i]
              all_hyp, all_scores = q_gen.generate_question_batch(src_seq, src_pos)
              for idx_seqs in all_hyp:
                  for idx_seq in idx_seqs:
                      pred_line = ' '.join([tgt_idx2word[idx] for idx in idx_seq])
                      #print(data[3],'===vs',sub_id,'===',pred_line)
                      f.write(pred_line + '\n')
              i+=1                                            
      print('[Info] Finished.'+path)
     
