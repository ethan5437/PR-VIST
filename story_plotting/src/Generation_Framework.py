import torch
import torch.nn as nn
import numpy
import math
from data_utilities import random_split, quick_collate
from torch.utils.data import DataLoader 
import torch.nn as nn
from torch.optim import lr_scheduler
from utility import save_model, load_model 
import random
from datetime import datetime
import json
import Constants
from tqdm import tqdm
from Prediction_Path_Search import Prediction_Path_Search, Vocabulary


total_rank = 0
rank_count = 0
total_all = 0
# device = Constants.device

class Generation_Framework():
    def __init__(self, args, word2id, rela2id):
        self.loss_function = nn.MarginRankingLoss(margin=args.margin)
        self.args = args
        self.word2id = word2id
        self.rela2id = rela2id
        self.id2word = {v:k for k,v in word2id.items()}
        self.id2rela = {v:k for k,v in rela2id.items()}
        self.framework = args.framework
        self.dataset = args.dataset.lower()
        self.device = 'cuda:'+str(args.device)

    def _eval_metric(self, scores):
        pos_scores = scores[-1].repeat(len(scores)-1) #prev
        neg_scores = scores[:-1] #next
        terminate = True if all([x > y for x, y in zip(pos_scores, neg_scores)]) else False
        return terminate
        # return loss, acc

    # def _loss_weight(self, current_len, total_len, acc, task):
    #     hop_weight = self.args.hop_weight**(current_len)
    #     task_weight = self.args.task_weight if task=='TD' else 1
    #     acc_weight = self.args.acc_weight if acc==1 else 1
    #     return acc_weight / (hop_weight * task_weight)

    def _padding_cuda(self, seqs, maxlen, pad_type, padding, seq_position=None, start_position=None):
        pad_seq, mask, position, pad_seq_position = [], [], [], []
        for seq in seqs:
            if pad_type == 'append':
                pad_seq.append(seq + [padding]*(maxlen-len(seq)))
                mask.append([1]*len(seq) + [0]*(maxlen-len(seq)))
                if start_position != None:
                    position.append([i+start_position for i in range(len(seq))] + [0]*(maxlen-len(seq)))
                if seq_position!=None:
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

    def _single_UHop_step(self, model, ques, ques_pos, pos_tuples, neg_tuples, execute_TD, pred_rela_list, pred_rela_text_list, step_pos=None):
        pos_rela, pos_rela_text, pos_prev, pos_prev_text = zip(*pos_tuples)
        # rela, rela_text = zip(*pos_tuples)
        neg_rela, neg_rela_text, neg_prev, neg_prev_text = zip(*neg_tuples)
        # input of question 
        if self.args.q_representation == 'bert':
            ques, ques_mask = self._padding_cuda([ques]*(len(pos_rela)+len(neg_rela)), max(5, len(ques)), 'append', 0)
            ques = torch.cat([ques, ques_mask], dim=-1)
        elif self.dataset == 'vist' and self.args.q_representation != 'bert' and step_pos != None:
            ques, _, ques_pos = self._padding_cuda([ques]*(len(pos_rela)+len(neg_rela)), max(5, len(ques)), 'append', self.word2id['PADDING'], ques_pos)
        else:
            ques, _ = self._padding_cuda([ques]*(len(pos_rela)+len(neg_rela)), max(5, len(ques)), 'append', self.word2id['PADDING'])
        # input of relation and previous
        if self.framework == 'baseline' or self.args.dynamic == 'none':
            # concat all previous and relation
            pos_relas = [sum(prev+[rela], []) for prev, rela in zip(pos_prev, pos_rela)]
            neg_relas = [sum(prev+[rela], []) for prev, rela in zip(neg_prev, neg_rela)]
            maxlen = max([len(rela) for rela in pos_relas+neg_relas])
#             print('_single_UHop_step -- maxlen1',maxlen)
            relas, _ = self._padding_cuda(pos_relas+neg_relas, maxlen, 'append', self.rela2id['PADDING'])
            pos_relas_text = [sum(prev+[rela], []) for prev, rela in zip(pos_prev_text, pos_rela_text)]
            neg_relas_text = [sum(prev+[rela], []) for prev, rela in zip(neg_prev_text, neg_rela_text)]
            maxlen = max([len(rela) for rela in pos_relas_text+neg_relas_text])
#             print('_single_UHop_step -- maxlen2',maxlen)
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
            score = model(ques, relas_text, relas, prevs_text, prevs, ques_pos)
        else:
            score = model(ques, relas_text, relas, prevs_text, prevs)
        

        # readible format for score of all candidates : [(score, [token1, token2 ... ]) ... ]
        #rev_rela = [self.tokenizer.convert_ids_to_tokens(rela) for rela in pos_concat_rela+neg_concat_rela]
        if execute_TD:
            termination = self._eval_metric(score)
            return termination
                
        else:               
            rev_rela = [[self.id2rela[r] for r in rela] for rela in pos_rela+neg_rela]
            rev_rela_text = [[self.id2word[t] for t in text] for text in pos_rela_text+neg_rela_text]
            rela_score_id=list(zip(score.detach().cpu().numpy().tolist()[:], pos_rela+neg_rela, pos_rela_text+neg_rela_text))
            rela_score=list(zip(score.detach().cpu().numpy().tolist()[:], rev_rela[:], rev_rela_text[:]))
            repetitve_penalty = self.args.repetitve_penalty
#             empty_frame_penalty = self.args.empty_frame_penalty
            if self.args.repetitve_penalty != 1.0:
                full_relation_list = []
                if len(pred_rela_list) != 0:
                    for (pred, pred_text) in zip(pred_rela_list, pred_rela_text_list):
                        full_relation_list.append(str(pred) + '+' + str(pred_text))
    #                 print('full_relation_list',full_relation_list[0])
                for i in range(len(score)):
                    prev_rev_rela = rela_score[i][1]
                    prev_rev_rela_text = rela_score[i][2]
                    prev_rev_rela_id = rela_score_id[i][1]
                    prev_rev_rela_text_id = rela_score_id[i][2]
                    
                    full_relation = str(prev_rev_rela_id)+ '+' + str(prev_rev_rela_text_id)
                    rela_counter = full_relation_list.count(full_relation)
                    #from -1 ~ 1, to 0 ~ 2
                    score[i] = (score[i] + 1) * (repetitve_penalty ** rela_counter)

            max_score, max_score_index = torch.max(score,0)
            return rela_score, rela_score_id, max_score_index 

    
    def _single_step_rela_choose(self, model, ques, ques_pos, next_tuples, prev_rela_list, prev_rela_text_list, step_pos=None):
        # make +/- pairs
        pos_tuples = next_tuples[:-1]
        neg_tuples = next_tuples[-1:]
        # special case
        if len(next_tuples) == 0:
            raise ValueError('next_tuples error in _first_step_rela_choose')
            
        only_one_candidate = False
        if len(next_tuples) == 1:
            pos_tuples = next_tuples
            neg_tuples = next_tuples
            only_one_candidate = True
#             return 0, 1, 'noNegativeInTD'
        if len(neg_tuples) > 1:
            print('mutiple positive tuples!')
#         if len(pos_tuples) > self.args.neg_sample:
#             pos_tuples = pos_tuples[:self.args.neg_sample]
        # run model
        execute_TD = False
        rela_score, rela_score_id, pred_rela_index = self._single_UHop_step(model, ques, ques_pos, pos_tuples, neg_tuples, execute_TD, prev_rela_list, prev_rela_text_list, step_pos)
        if only_one_candidate:
            pred_rela_index = 0
        return rela_score, rela_score_id, pred_rela_index
    
    
    def _terminate_decision(self, model, ques, ques_pos, tuples, next_tuples, prev_rela_list, prev_rela_text_list, step_pos=None):
        pos_tuples = next_tuples
        neg_tuples = tuples
        if len(pos_tuples) == 0 or len(neg_tuples) == 0:
            print('tuples',tuples)
            print('next_tuples',next_tuples)
            raise ValueError('next_tuples error in _single_step_rela_choose')
#             return 0, 1, 'noNegativeInTD'
        if len(neg_tuples) > 1:
            print('mutiple negative tuples!')
#         if len(pos_tuples) > self.args.neg_sample:
#             pos_tuples = pos_tuples[:self.args.neg_sample]
        # run model
        execute_TD = True
        termination = self._single_UHop_step(model, ques, ques_pos, pos_tuples, neg_tuples, execute_TD, prev_rela_list, prev_rela_text_list, step_pos)
        return termination
    
    def _execute_UHop(self, model, data, mode, dataset):
        story_id = data[0]
        photo_ids = data[1]
        ques = data[2][0]
        ques_pos = data[2][1]

        labels, scores = [], []
        step_list = []
        pred_enitity_rela_list = []
        pred_enitity_list = []
        prev_rela_list = []
        prev_rela_text_list = []
        prev_pos_list = []
        prev_structural_list = []
        
        
        max_term_step = 20
        steps_len = []
        for i in range(max_term_step):
            
            if i == 0:
                current_noun = '<s>_NOUN'
                current_noun_id = Constants.BOS_ID
                current_pos = 0 #current image position
                sentence_counter = 0
            else:
                current_noun = pred_enitity_list[-1]#choose the highest pred_entity
#                 assert current_noun != '<s>_NOUN', "expect <s1>_NOUN, got current_noun == <s>_NOUN instead"
                current_pos = prev_pos_list[-1]             
                #Then 
                if current_noun in Constants.BOS_LIST:
                    sentence_counter +=1
                
            # TD if not the first nor the last step : continue
            steps, steps_position, candidate_objs = dataset.get_step(photo_ids, current_noun, current_pos, i,\
                                              sentence_counter, prev_rela_list, prev_rela_text_list, self.args)
            
            pred_list = []
            sub_steps_len = []
            for (step, step_pos, candidate_obj) in zip(steps, steps_position, candidate_objs):
                sub_steps_len.append(len(step))

                #Need to be fixed?
                step_pos = [pos+1 for pos in step_pos]
                rela_score, rela_score_id, pred_rela_index = self._single_step_rela_choose(model, ques, ques_pos, step, prev_rela_list, prev_rela_text_list, current_pos+1)
                if (rela_score == None) and (rela_score_id == None) and (pred_rela_index == None):
                    continue
                score, pred_rela, pred_rela_text = rela_score[pred_rela_index]                
                pred_rela_id = rela_score_id[pred_rela_index][1]
                pred_rela_text_id = rela_score_id[pred_rela_index][2]
                pred_rela_id_, pred_rela_text_id_, prev_rela_ids_, prev_rela_text_ids_ = step[pred_rela_index]
                pred_pos = step_pos[pred_rela_index]
                pred_entity = candidate_obj[pred_rela_index]      
                if i == 0:
                    termination = False
                else:
                    prev_step = [[prev_rela_list[-1], prev_rela_text_list[-1], prev_rela_list[:-1], prev_rela_text_list[:-1]]]
                    pred_step = step
                    termination = self._terminate_decision(model, ques, ques_pos, prev_step, pred_step, prev_rela_list, prev_rela_text_list, current_pos+1)
                    
                pred_list.append([score, pred_entity, pred_rela, pred_rela_text, pred_rela_id, pred_rela_text_id, pred_pos,\
                                  termination])
            max_item = max(pred_list, key=lambda item: item[0])
            best_score, best_pred_entity, best_rela, best_rela_text, best_pred_rela_id, best_pred_rela_text_id, best_pos,\
                                                                                    best_termination = max_item
                                
            if self.args.only_five2seven_sentences:
                if pred_enitity_rela_list.count('<s>_NOUN') == 5 or pred_enitity_rela_list.count('<s>_NOUN') == 6:
                    pass
                elif pred_enitity_rela_list.count('<s>_NOUN') >= 7:
                    best_termination = True
                else:
                    best_termination = False
                                            
            if not best_termination:
                pred_enitity_list.append(best_pred_entity)
                prev_structural_list.append(best_pred_entity)
                prev_structural_list.extend(best_rela)
#                 if best_rela[0] != (Constants.FRAME_PAD or Constants.FRAME_START or Constants.FRAME_END):
                if best_rela[0] != Constants.FRAME_PAD and  best_rela[0] != Constants.FRAME_START \
                    and best_rela[0] != Constants.FRAME_END:
                    pred_enitity_rela_list.extend(best_rela[:-1])
                pred_enitity_rela_list.append(best_pred_entity)
                scores.append(best_score)
                
                
            prev_rela_list.append(best_pred_rela_id)
            prev_rela_text_list.append(best_pred_rela_text_id)
            prev_pos_list.append(best_pos-1) #12345 --> 01234

            steps_len.append(sum(sub_steps_len)/len(sub_steps_len))
            if not best_termination and i != max_term_step-1:
                pass
            else:
                break
        return model, scores, story_id, pred_enitity_rela_list, prev_structural_list, steps_len

    
    def generation(self, model, mode, dataset, output_result=False):
        #Read model
        model = self.args.Model(self.args).to(self.device)
        model = load_model(model, self.args)
        model = model.eval().to(self.device)        
        #Read data and KG relations
        dataset = Prediction_Path_Search(self.args, mode, self.word2id, self.rela2id)
        datas = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=12, 
                pin_memory=False, collate_fn=quick_collate)
        
        output = []
        output_rela = []
        output_with_scores = []
        scores = []
        step_len_list = []
        story_id_story_dict = {}
        with torch.no_grad():
            model = model.eval()
            for num, data in enumerate(tqdm(datas)):
                story_id = data[0]
                if story_id not in story_id_story_dict.keys():
                    model, score, story_id, pred_enitities, prev_structural_list, steps_len = self._execute_UHop(model, data, mode, dataset)
                    story_id_story_dict.update({story_id: [score, pred_enitities, prev_structural_list]})
                else:
                    score, pred_enitities, prev_structural_list = story_id_story_dict[story_id]
                if num <= 10:
                    print('pred_enitities',pred_enitities)
                else:
                    if self.args.small:
                        break
                    else:
                        pass

                pred_list = []
                tmp_list = []
                for i,pred_entity in enumerate(pred_enitities):
                    if pred_entity in Constants.BOS_LIST:
                        pred_list.append(tmp_list)
                        tmp_list = []
                    elif i == len(pred_enitities)-1:
                        tmp_list.append(pred_entity)
                        pred_list.append(tmp_list)
#                         tmp_list = []
                    else:
                        tmp_list.append(pred_entity)
                # total_count +=1
                scores.append(score)
                sub_output = []
                for i,pred_entity in enumerate(pred_list):
                    sub_output.append({'story_id':story_id, 'predicted_term_seq':pred_entity, "text": ""})
                sub_output.append({'story_id':story_id, 'scores':scores})   
                
                pred_structural_list = []
                tmp_list = []
                for i,pred_entity in enumerate(prev_structural_list):
                    if pred_entity in Constants.BOS_LIST:
                        pred_structural_list.append(pred_entity)
                        pred_structural_list.append(tmp_list)
                        tmp_list = []
                    elif i == len(prev_structural_list)-1:
                        tmp_list.append(pred_entity)
                        pred_structural_list.append(tmp_list)
#                         tmp_list = []
                    else:
                        tmp_list.append(pred_entity)
                               
                output.append(sub_output[:-1])
                output_with_scores.append(sub_output[-1])
                output_rela.append(pred_structural_list)
                
#                 if num == 1:
#                     raise ValueError('testing: data iteration stop')

        return output, output_with_scores, output_rela
