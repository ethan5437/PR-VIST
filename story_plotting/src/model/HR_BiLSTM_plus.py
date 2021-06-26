import torch as th
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np
import math



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.word_embedding = nn.Embedding(args.word_embedding.shape[0], args.word_embedding.shape[1])
        self.word_embedding.weight = nn.Parameter(th.from_numpy(args.word_embedding).float())
        self.word_embedding.weight.requires_grad = False
        self.rela_embedding = nn.Embedding(args.rela_vocab_size, args.word_embedding.shape[1])
        
#         maximum 30 sentences
        if args.abs_position or args.ethan_position:
            self.pos_embedding = nn.Embedding(20, args.word_embedding.shape[1])
        if args.recurrent_UHop: #or args.ethan_position:
            self.sen_pos_embedding = nn.Embedding(30, args.word_embedding.shape[1])
        
        nn.init.xavier_normal_(self.rela_embedding.weight)
        self.rnn = nn.LSTM(input_size=args.emb_size, hidden_size=768//2 if args.q_representation=="bert" else args.hidden_size,
                          num_layers=1, batch_first=False,
                          dropout=args.dropout_rate, bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=args.hidden_size*2, hidden_size=args.hidden_size,
                          num_layers=1, batch_first=False,
                          dropout=args.dropout_rate, bidirectional=True)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.args = args
        self.cos = nn.CosineSimilarity(dim=1)
#         self.tanh = nn.Tanh()
        self.linear = nn.Linear(args.hidden_size*4, args.hidden_size*2, bias=True)
        self.method = args.reduce_method
        self.q_representation = args.q_representation
        self.dataset = args.dataset
        self.args = args
        if args.is_seperate_structural:
             self.rela_pos_embedding = nn.Embedding(6, args.word_embedding.shape[1])
        print('Model -- self.q_representation',self.q_representation)
        print('Model -- self.dataset',self.dataset)

    def rela_encoding(self, rela_text_x, rela_x, ques_hidden_state, ques_h, rela_text_x_pos, rela_x_pos):
        rela_text_x = th.transpose(rela_text_x, 0, 1)
        rela_x = th.transpose(rela_x, 0, 1)
        rela_text_x = self.word_embedding(rela_text_x)
        rela_x = self.rela_embedding(rela_x)
        
        if self.args.is_seperate_structural and rela_text_x_pos is not None and rela_x_pos is not None:
            rela_text_x_pos = th.transpose(rela_text_x_pos, 0, 1)
            rela_x_pos = th.transpose(rela_x_pos, 0, 1)
            rela_text_x_pos = self.rela_pos_embedding(rela_text_x_pos)
            rela_x_pos = self.rela_pos_embedding(rela_x_pos)
            
            rela_text_x = rela_text_x + rela_text_x_pos
            rela_x = rela_x + rela_x_pos
        #rela_text_x = self.dropout(rela_text_x)
        #rela_x = self.dropout(rela_x)
        if(self.q_representation=="bert"):
            rela_hs, hidden_state = self.encode(rela_x)
        else:
            rela_hs, hidden_state = self.encode(rela_x, ques_hidden_state)
        rela_text_hs, hidden_state = self.encode(rela_text_x, hidden_state)
        rela_hs = th.cat([rela_hs, rela_text_hs], 0)
        rela_hs = rela_hs.permute(1, 2, 0)
        rela_h = F.avg_pool1d(rela_hs, kernel_size=rela_hs.shape[2], stride=None)
        rela_h = rela_h.squeeze(2)
        return rela_h

    def forward(self, *inputs):
        if self.dataset == 'vist':
            ques_x, rela_text_x, rela_x, prev_rela_text_x, prev_rela_x, ques_pos = inputs[0], inputs[1], inputs[2],\
                                                                                    inputs[3], inputs[4], inputs[5]
            if self.args.is_seperate_structural:
                rela_x_pos = inputs[6]
                rela_text_x_pos = inputs[7]
#                 print('rela_text_x',rela_text_x)
#                 print('rela_x',rela_x)
#                 print('rela_x_pos',rela_x_pos)
#                 print('rela_text_x_pos',rela_text_x_pos)
#                 error
                
#             print('forward -- ques_x',ques_x.shape)
#             print('forward -- ques_pos',ques_pos.shape)
        else:
            ques_x, rela_text_x, rela_x, prev_rela_text_x, prev_rela_x = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        
#         print('ques_x',ques_x)
#         print('rela_text_x',rela_text_x)
#         print('rela_x',rela_x)
#         print('prev_rela_text_x',prev_rela_text_x)
#         print('prev_rela_x',prev_rela_x)
        
        if(self.q_representation=="bert"):
            question, ques_mask = th.chunk(ques_x, 2, dim=1)
            question = th.squeeze(question, 1)
            ques_mask = th.squeeze(ques_mask, 1)
            ques_segment = th.zeros_like(question, dtype=th.long)
            ques_hs1, pooled_question = self.bert(question, ques_segment, ques_mask, output_all_encoded_layers=False)
            ques_hs1 = self.dropout(ques_hs1)
            ques_hs = th.transpose(ques_hs1, 0, 1)
                        
        else:
            ques_x = th.transpose(ques_x, 0, 1)
            ques_x = self.word_embedding(ques_x)
            if self.args.abs_position or self.args.ethan_position:
                ques_pos = th.transpose(ques_pos, 0, 1) 
                ques_pos = self.pos_embedding(ques_pos)
                ques_x = ques_x+ques_pos
            elif self.args.recurrent_UHop:
                ques_pos = th.transpose(ques_pos, 0, 1) 
                ques_pos = self.sen_pos_embedding(ques_pos)
                ques_x = ques_x+ques_pos
                
            ques_x = self.dropout(ques_x)
            ques_hs1, ques_hidden_state = self.encode(ques_x)
            ques_hs2, _ = self.rnn2(ques_hs1) 
            ques_hs2 = self.dropout(ques_hs2)
            ques_hs = ques_hs1 + ques_hs2
        ques_hs = ques_hs.permute(1, 2, 0)
        ques_h = F.avg_pool1d(ques_hs, kernel_size=ques_hs.shape[2], stride=None)
        ques_h = ques_h.squeeze(2)


        for prev_text, prev_rela in zip(prev_rela_text_x, prev_rela_x):
            if prev_rela.shape[0]>0:
                prev_rela_h = self.rela_encoding(prev_text, prev_rela, ques_hidden_state, ques_h, None, None)
                ques_h = self.linear(th.cat((ques_h, prev_rela_h), dim=-1))
                
        if self.args.is_seperate_structural:
            rela_h = self.rela_encoding(rela_text_x, rela_x, ques_hidden_state, ques_h, rela_text_x_pos, rela_x_pos)
        else:
            rela_h = self.rela_encoding(rela_text_x, rela_x, ques_hidden_state, ques_h, None, None)

        output = self.cos(ques_h, rela_h)
#         if th.isnan(output).any():
#             print('ques_x', inputs[0])
#             print('rela_text_x', inputs[1])
#             print('rela_x', inputs[2])
#             print('prev_rela_text_x', inputs[3])
#             print('prev_rela_x', inputs[4])
#             print('ques_pos', inputs[5])
#             print('ques_h', ques_h)
#             print('rela_h', rela_h)
#             print('output',output)
#             raise ValueError('all 0s...2')
#         print('output',output)
#         error
        return output

    def encode(self, input, hidden_state=None, return_sequence=True):
#         print('encode-- input', input[0][0][:10])
        if hidden_state==None:
            outputs, (h_output, c_output) = self.rnn(input)
        else:
            h_0, c_0 = hidden_state
            outputs, (h_output, c_output) = self.rnn(input, (h_0, c_0))
            
#         print('encode-- outputs[0][0][0:10]', outputs[0][0][0:10])
#         print('encode-- outputs[1][0]', outputs[1][0])
#         if th.isnan(outputs).any():
#             print('encode-- input', input)        
#             print('encode-- outputs', outputs)           
        #print('outputs', outputs)
        outputs = self.dropout(outputs)
#         print('outputs', outputs)
        h_output = self.dropout(h_output)
#         print('h_output', h_output)
        c_output = self.dropout(c_output)
#         print('c_output', c_output)
        if return_sequence == False:
            return outputs[-1], (h_output, c_output)
        else:
            return outputs, (h_output, c_output)
