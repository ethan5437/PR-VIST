""" Manage beam search info structure.

    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import torch
import numpy as np
import transformer.Constants as Constants

class Beam():
    ''' Beam search '''

    def __init__(self, size, tgt_seq, tgt_pos, tgt_sen_pos, device=False):
        self.size = size
        self._done = False
        self.device = device

        sentence_length = len([seq for seq in tgt_seq if seq != 0])
        self.tgt_seq = [seq for seq in tgt_seq if seq != 0]
        self.tgt_seq = torch.tensor([self.tgt_seq,self.tgt_seq,self.tgt_seq], device=device)        
        if all(seq == 0 for seq in tgt_pos):  
            self.tgt_pos = [0]*(sentence_length) 
        else:
            self.tgt_pos = [seq for seq in tgt_pos if seq != 0]
        self.tgt_pos = torch.tensor([self.tgt_pos,self.tgt_pos,self.tgt_pos], device=device)
        
        if all(seq == 0 for seq in tgt_sen_pos):  
            self.tgt_sen_pos = [0]*(sentence_length)
        else:
            self.tgt_sen_pos = [seq for seq in tgt_sen_pos if seq != 0]
        self.tgt_sen_pos = torch.tensor([self.tgt_sen_pos,self.tgt_sen_pos,self.tgt_sen_pos], device=device)
        
        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), Constants.BOS, dtype=torch.long, device=device)]

        # The pos at each time-step
        self.next_pos = [torch.full((size,), 1, dtype=torch.long, device=device)]
        
        # The sen pos at each time-step
        self.next_sen_pos = [torch.full((size,), self.tgt_sen_pos[-1][-1]+1, dtype=torch.long, device=device)]

        # BOSs set
        self.BOSs_set = set(Constants.BOSs)
    def get_current_seq_state(self):
        "Get the outputs for the current timestep."
        return self.get_seq_tentative_hypothesis()

    def get_current_pos_state(self):
        "Get the outputs for the current timestep."
        return self.get_pos_tentative_hypothesis()

    def get_current_sen_pos_state(self):
        "Get the outputs for the current timestep."
        return self.get_sen_pos_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 2nd sort

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # Position
        tmp_pos=torch.full((self.size,), Constants.PAD, dtype=torch.long, device=self.device)
        tmp_sen_pos=torch.full((self.size,), Constants.PAD, dtype=torch.long, device=self.device)
        for i, word in enumerate(self.next_ys[-1]):
                tmp_pos[i] = self.next_pos[-1][self.prev_ks[-1][i]]+1
                tmp_sen_pos[i] = self.next_sen_pos[-1][self.prev_ks[-1][i]]
        self.next_pos.append(tmp_pos)
        self.next_sen_pos.append(tmp_sen_pos)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == Constants.EOS:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_seq_tentative_hypothesis(self):
        "Get the decoded (word, pos, sen_pos) sequence for the current timestep."

        if len(self.next_ys) == 1:
            #not sure whether we should us [2,0,0] or [2,2,2]
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_seq_hypothesis(k) for k in keys]
            hyps = [[Constants.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)      
        dec_seq = torch.cat((self.tgt_seq,dec_seq.to(self.device)), 1)
        
        return dec_seq

    def get_pos_tentative_hypothesis(self):
        "Get the decoded (word, pos, sen_pos) sequence for the current timestep."
        if len(self.next_pos) == 1:
            dec_seq = self.next_pos[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_pos_hypothesis(k) for k in keys]
            hyps = [[1] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)
        dec_seq = torch.cat((self.tgt_pos, dec_seq.to(self.device)), 1)
        
        return dec_seq

    def get_sen_pos_tentative_hypothesis(self):
        "Get the decoded (word, pos, sen_pos) sequence for the current timestep."
        if len(self.next_sen_pos) == 1:
            dec_seq = self.next_sen_pos[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_sen_pos_hypothesis(k) for k in keys]
            hyps = [[h[-1]] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)
        dec_seq = torch.cat((self.tgt_sen_pos,dec_seq.to(self.device)), 1)
        
        return dec_seq

    def get_pos_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """

        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_pos[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))

    def get_seq_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]
            
        return list(map(lambda x: x.item(), hyp[::-1]))

    def get_sen_pos_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_sen_pos[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))
