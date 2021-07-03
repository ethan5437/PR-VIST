from os import listdir
from os.path import isfile, join
import os
import pdb
import csv
import pickle
import numpy
from itertools import chain, repeat, islice
import json
import sys
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data.dataloader
from tqdm import tqdm

import discriminator
import helpers
import generator

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word_count = []

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def text2id_pad(l):
    max_len = 200
    listofzeros = [0] * max_len
    app = []
    for i in range(len(l)):
        app.append(story_vocab(l[i]))
    listofzeros[:len(app)] = app 
    return listofzeros

def reward_lossweight(x):
    weight_loss = -x + 1.5
    return weight_loss

class TermDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):      
        tuple_data = self.data[index]
        return tuple_data
        
    def collate_fn(self, datas):
                
        batch = []
        for pos, neg in datas:
            pre_context = pos
            target_tensor = neg
            batch.append( (pre_context, target_tensor) )
        return batch


          
with open('/home/wei0401/commen-sense-storytelling/user_score/story_vocab.pkl','rb') as f_voc:
    story_vocab = pickle.load(f_voc) 

with open('/home/wei0401/commen-sense-storytelling/user_score/data/pos_neg_story_withimg.json') as f_pos_neg:
    pos_neg = json.load(f_pos_neg)

with open('/home/wei0401/commen-sense-storytelling/user_score/data/pos_neg_story_withoutimg.json') as f_pos_neg_2:
    pos_neg_2 = json.load(f_pos_neg_2)



CUDA = True
POS_NEG_SAMPLES = 200
GEN_EMBEDDING_DIM = 300
GEN_HIDDEN_DIM = 300
DIS_EMBEDDING_DIM = 600
DIS_HIDDEN_DIM = 600
VOCAB_SIZE = len(story_vocab)
MAX_SEQ_LEN = 200

VAL_DATA_SIZE = 200
BATCH_SIZE = 50
MAX_EPOCH = 100
Debug_flag = True
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# load data,  data 1: with images, data 2: with + without images
train_cbow = []
for val in pos_neg:
    pos_story = pos_neg[val]['pos_neg_story'][0]
    neg_story = pos_neg[val]['pos_neg_story'][1]

    pos_story_list = pos_story.split()
    neg_story_list = neg_story.split()

    pos_story_voc = text2id_pad(pos_story_list)
    pos_story_voc_torch = torch.FloatTensor(pos_story_voc).long()
    neg_story_voc = text2id_pad(neg_story_list)
    neg_story_voc_torch = torch.FloatTensor(neg_story_voc).long()

    train_cbow.append((pos_story_voc_torch, neg_story_voc_torch))

for val in pos_neg_2:
    pos_story = pos_neg[val]['pos_neg_story'][0]
    neg_story = pos_neg[val]['pos_neg_story'][1]

    pos_story_list = pos_story.split()
    neg_story_list = neg_story.split()

    pos_story_voc = text2id_pad(pos_story_list)
    pos_story_voc_torch = torch.FloatTensor(pos_story_voc).long()
    neg_story_voc = text2id_pad(neg_story_list)
    neg_story_voc_torch = torch.FloatTensor(neg_story_voc).long()

    train_cbow.append((pos_story_voc_torch, neg_story_voc_torch))

# random seperate data as training and validation
random.shuffle(train_cbow)
validation_data = train_cbow[0:VAL_DATA_SIZE]
training_data = train_cbow[VAL_DATA_SIZE:]
# data loader
train_dataset = TermDataset(training_data)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, collate_fn=train_dataset.collate_fn, shuffle=True)
valid_data = TermDataset(validation_data)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size = BATCH_SIZE, collate_fn=valid_data.collate_fn, shuffle=True)

'''
dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
if CUDA:
    dis = dis.cuda()

# optimizer type
# dis_optimizer = optim.Adagrad(dis.parameters(), lr=1e-5)
dis_optimizer = optim.Adam(dis.parameters(), lr=1e-5)

model_description = '3-b-B-ep100-len200-2'
best_val_loss = None
best_val_acc = None
trained_dis_path = '/home/wei0401/commen-sense-storytelling/user_score/saved_model/'
val_log_path = '/home/wei0401/commen-sense-storytelling/user_score/log/val_log_{}.txt'.format(model_description) 
val_log = open(val_log_path, "a+")
val_log.write("LOSS ACC SAVE?")
val_log.write("\n")

print("......Start training discriminator......")
for epoch in range(MAX_EPOCH):
    LOSS_AVG_TRAIN = 0
    ACC_AVG_TRAIN = 0
    LOSS_AVG_VAL = 0
    ACC_AVG_VAL = 0

    for batch_train in tqdm(train_dataloader, total=len(train_dataloader), desc='Training'):
        POS = []
        NEG = []
        for tupple in batch_train:
            POS.append(tupple[0])
            NEG.append(tupple[1])
        POS_tensor = torch.stack(POS)
        NEG_tensor = torch.stack(NEG)


        inp, target = helpers.prepare_discriminator_data(POS_tensor, NEG_tensor, gpu=CUDA)
        dis_optimizer.zero_grad()
        out = dis.batchClassify(inp)
        loss_fn = nn.BCELoss()
        loss = loss_fn(out, target)
        loss.backward()
        dis_optimizer.step()

        training_loss = loss.data.item()
        training_acc = torch.sum((out > 0.5) == (target > 0.5)).data.item() / len(out)
        LOSS_AVG_TRAIN += training_loss
        ACC_AVG_TRAIN += training_acc

    LOSS_AVG_TRAIN = LOSS_AVG_TRAIN  
    ACC_AVG_TRAIN = ACC_AVG_TRAIN / (len(training_data) / BATCH_SIZE)
    print('Epoch {}, Trainin Loss: {:.4f}, Trainin Accuracy:{:.4f} \n'.format(epoch, LOSS_AVG_TRAIN, ACC_AVG_TRAIN))
    # print("training loss:", LOSS_AVG_TRAIN , ", training accuracy: ", ACC_AVG_TRAIN)

    for batch_val in tqdm(valid_dataloader, total=len(valid_dataloader), desc='Validation'):
        POS_val = []
        NEG_val = []
        for tupple_val in batch_val:
            POS_val.append(tupple_val[0])
            NEG_val.append(tupple_val[1])
        POS_val_tensor = torch.stack(POS_val)
        NEG_val_tensor = torch.stack(NEG_val)

        inp_val, target_val = helpers.prepare_discriminator_data(POS_val_tensor, NEG_val_tensor, gpu=CUDA)
        out_val = dis.batchClassify(inp_val)
        loss_val = loss_fn(out_val, target_val)
        val_loss = loss_val.data.item()
        val_acc = torch.sum((out_val > 0.5) == (target_val > 0.5)).data.item() / len(out_val)
        LOSS_AVG_VAL += val_loss
        ACC_AVG_VAL += val_acc

    LOSS_AVG_VAL = LOSS_AVG_VAL
    ACC_AVG_VAL = ACC_AVG_VAL / (len(validation_data) / BATCH_SIZE)
    print('Epoch {}, Validation Loss: {:.4f}, Validation Accuracy:{:.4f} \n'.format(epoch, LOSS_AVG_VAL, ACC_AVG_VAL))
    # print("val loss: ", val_loss, ", val accuracy: ", val_acc)
    if Debug_flag != True:
        if best_val_loss is None or LOSS_AVG_VAL < best_val_loss:
            print('......Saving model with min loss......')
            best_val_loss = LOSS_AVG_VAL
            best_val_acc = ACC_AVG_VAL
            torch.save(dis.state_dict(), os.path.join(trained_dis_path,'model_minloss-{}'.format( model_description )))
            val_log.write(str(round(LOSS_AVG_VAL, 3)) + " " + str(round(ACC_AVG_VAL, 3)) + " " + "True")
            val_log.write("\n")
        else:
            val_log.write(str(round(LOSS_AVG_VAL, 3)) + " " + str(round(ACC_AVG_VAL, 3)) + " " + "False")
            val_log.write("\n")

print("......Finish discriminator training......")
print("best model loss: {} , acc: {}".format(best_val_loss, best_val_acc))




'''
dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
dis.load_state_dict(torch.load('/home/wei0401/commen-sense-storytelling/user_score/saved_model/model_minloss-3-b-B-ep100'))
# gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
if CUDA:
    # gen = gen.cuda()
    dis = dis.cuda()
for batch_val in tqdm(valid_dataloader, total=len(valid_dataloader), desc='Validation'):
    POS_val = []
    NEG_val = []
    for tupple_val in batch_val:
        POS_val.append(tupple_val[0])
        NEG_val.append(tupple_val[1])
    POS_val_tensor = torch.stack(POS_val)
    NEG_val_tensor = torch.stack(NEG_val)

    target = POS_val_tensor.cuda()
    generated = NEG_val_tensor.cuda()

    rewards_forstory = dis.batchClassify(target) # reward can multiple on loss
    weight_for_loss = reward_lossweight(rewards_forstory)
    # loss_after_weight = torch.mul(your_loss, weight_for_loss)

    # inp2, target2 = helpers.prepare_discriminator_data(target, generated, gpu=CUDA) # inp: generated story, target: ground truth story
    # loss_fn = nn.BCELoss()
    # out1 = dis.batchClassify(inp2)
    # loss1 = loss_fn(out1, target2)
    # pdb.set_trace()


    # rewards = dis.batchClassify(target)
    
    # pg_loss = gen.batchPGLoss(inp, target, rewards)
    # # pg_loss.backward()
    # # gen_opt.step()
    # inp1, target1 = helpers.prepare_discriminator_data(POS_val_tensor, NEG_val_tensor, gpu=CUDA)
    # loss_fn = nn.BCELoss()
    # out = dis.batchClassify(inp1)
    # loss = loss_fn(out, target1)
    # # loss.backward()
    # # gen_opt.step()
