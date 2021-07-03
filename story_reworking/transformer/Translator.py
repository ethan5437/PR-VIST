''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import Transformer
from transformer.Beam import Beam
from transformer import Constants
import math
import copy
class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(f'cuda:{opt.device}' if opt.cuda else 'cpu')
        checkpoint = torch.load(opt.model, map_location=self.device)
        model_opt = checkpoint['settings']

        self.model_opt = model_opt
        model = Transformer(
            model_opt.src_vocab_size,
            model_opt.tgt_vocab_size,
            model_opt.max_encode_token_seq_len,
            model_opt.max_token_seq_len,
            tgt_emb_prj_weight_sharing=model_opt.proj_share_weight,
            emb_src_tgt_weight_sharing=model_opt.embs_share_weight,
            d_k=model_opt.d_k,
            d_v=model_opt.d_v,
            d_model=model_opt.d_model,
            d_word_vec=model_opt.d_word_vec,
            d_inner=model_opt.d_inner_hid,
            n_layers=model_opt.n_layers,
            n_head=model_opt.n_head,
            dropout=model_opt.dropout).to(self.device)
        
        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        model.word_prob_prj = nn.LogSoftmax(dim=1)
        
        self.model = model
        self.model.eval()

    def translate_batch(self, src_seq, src_pos, src_sen_pos, tgt_seq, tgt_pos, tgt_sen_pos, previous_gt_seq, story_len, pred_seq, pred_seq_pos, pred_seq_sen_pos):
        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)
            
            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(
                inst_dec_beams, len_dec_seq, src_seq, enc_output, tgt_sentence_emb, story_len, inst_idx_to_position_map, n_bm):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_seq_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
#                 print('dec_partial_seq',dec_partial_seq)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_pos_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
#                 print('dec_partial_seq',dec_partial_seq)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_sen_pos(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_sen_pos_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq
            
            #LSTM Modification
            def predict_word(dec_seq, dec_pos, dec_sen_pos, src_seq, enc_output, previous_gt_seq, story_len, n_active_inst, n_bm, pred_seq, pred_seq_pos, pred_seq_sen_pos):
                story_len = story_len.repeat(dec_seq.size(0))
    
                pred_seq = torch.stack(dec_seq.size(0)*[pred_seq])
                pred_seq_pos = torch.stack(dec_seq.size(0)*[pred_seq_pos])
                pred_seq_sen_pos = torch.stack(dec_seq.size(0)*[pred_seq_sen_pos])
                
                #LSTM Modification
                previous_gt_seq = torch.stack(dec_seq.size(0)*[previous_gt_seq]).transpose(0,1)
                
                dec_output, *_ = self.model.decoder(dec_seq, dec_pos, dec_sen_pos, src_seq, enc_output, previous_gt_seq, story_len)
                
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h     
                logits = self.model.tgt_word_prj(dec_output)
                ## word_prob actually
                logits = F.log_softmax(logits, dim=1)
#                 # UNK mask
                logits[:, Constants.UNK] = -1e19
                #rm_set = set(Constants.BOSs+[13])
                rm_set = list(set(Constants.BOSs+[19])) # 19 => "."
                pronouns = [198, 3001, 73, 34, 406, 235]
                rm_set = rm_set + pronouns
                
                #logits[:, 19] -= logits[:, 19].abs()* + 1e-10
                story_dec_seq = torch.cat((pred_seq, dec_seq),1)
                story_dec_pos = torch.cat((pred_seq_pos, dec_pos),1)
                story_dec_sen_pos = torch.cat((pred_seq_sen_pos, dec_sen_pos),1)
                
                for i, (ins, pos, sen_pos) in enumerate(zip(story_dec_seq, story_dec_pos, story_dec_sen_pos)):
                    current_sen_pos = sen_pos[-1]
                    for token, s_pos in zip(ins.flip(0), sen_pos.flip(0)):
                        length_norm = len(ins)
                        if token.item() not in rm_set and s_pos == current_sen_pos:
                            logits[i, token] -= 5+1e-19
                        if s_pos != current_sen_pos:
                            logits[i, token] -= 20/length_norm - 1e-19
                            #break
                word_prob = F.log_softmax(logits, dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
        
                word_prob = logits.view(n_active_inst, n_bm, -1)
                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                i = 0
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    i+=1
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(inst_dec_beams, len_dec_seq)
            dec_sen_pos = prepare_beam_dec_sen_pos(inst_dec_beams, len_dec_seq)
            assert len(dec_seq[0]) == len(dec_pos[0]), str(len(dec_seq[0])) + "vs" + str(len(dec_pos[0]))
            assert len(dec_pos[0]) == len(dec_sen_pos[0]), str(len(dec_pos[0])) + "vs" + str(len(dec_sen_pos[0]))
            assert len(dec_seq[0]) == len(dec_sen_pos[0]), str(len(dec_seq[0])) + "vs" + str(len(dec_sen_pos[0]))
            #dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            #dec_pos, dec_sen_pos = prepare_beam_dec_pos(dec_seq)
            word_prob = predict_word(dec_seq, dec_pos, dec_sen_pos, src_seq, enc_output, previous_gt_seq, story_len, n_active_inst, n_bm, pred_seq, pred_seq_pos, pred_seq_sen_pos)
            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)
            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]
                hyps = [inst_dec_beams[inst_idx].get_seq_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            #-- Encode
            #LSTM Modification
            src_seq, src_pos, src_sen_pos, tgt_seq, previous_gt_seq, story_len = src_seq.to(self.device), src_pos.to(self.device), src_sen_pos.to(self.device), tgt_seq.to(self.device), previous_gt_seq.to(self.device), story_len.to(self.device)
            pred_seq, pred_seq_pos, pred_seq_sen_pos = pred_seq.to(self.device), pred_seq_pos.to(self.device), pred_seq_sen_pos.to(self.device)

            src_enc, *_ = self.model.encoder(src_seq, src_pos, src_sen_pos, story_len)
            #LSTM Modification

            #-- Repeat data for beam search
            n_bm = self.opt.beam_size
            n_inst, len_s, d_h = src_enc.size()
            src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)

            #-- Prepare beams
            #LSTM modification
            inst_dec_beams = [Beam(n_bm, tgt_seq, tgt_pos, tgt_sen_pos,device=self.device) for _ in range(n_inst)]

             #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            #-- Decode
            #for len_dec_seq in range(1, self.model_opt.max_token_seq_len + 1):
            
            #LSTM modification
            tgt_lenth = len([seq for seq in tgt_seq if seq != 0])+1
            for len_dec_seq in range(tgt_lenth, self.opt.max_token_seq_len):
                #LSTM Modification
                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_seq, src_enc, previous_gt_seq, story_len, inst_idx_to_position_map, n_bm)
                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)
        
        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, self.opt.n_best)

        return batch_hyp, batch_scores
