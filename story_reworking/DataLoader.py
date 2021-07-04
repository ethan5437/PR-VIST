import torch
from torch.utils.data.dataset import Dataset
import json
from build_story_vocab import Vocabulary
from transformer import Constants
import spacy
import numpy as np
import copy
nlp = spacy.load("en_core_web_sm", disable=['tagger','parser','ner', 'vector'])
from spacy.symbols import ORTH
nlp.tokenizer.add_special_case(u'[female]', [{ORTH: u'[female]'}])
nlp.tokenizer.add_special_case(u'[male]', [{ORTH: u'[male]'}])
nlp.tokenizer.add_special_case(u'[location]', [{ORTH: u'[location]'}])
nlp.tokenizer.add_special_case(u'[organization]', [{ORTH: u'[organization]'}])

class ROCDataset(Dataset):
    def __init__(self,
                 story_vocab,
                 frame_vocab,
                 text_path, 
                 hop,
                 is_verb=True):
        print(text_path)
        self.dialogs = json.load(open(text_path, 'r'))
        self.max_term_len = 9 #9 for head(Consant BOS[0]) only
        self.max_sentence_len = 23 #24 for head(Consant BOS[0]) only
        self.story_vocab = story_vocab
        self.frame_vocab = frame_vocab
        self.hop = hop

    def __getitem__(self, index):
        frame = []
        story = []
        dialog = self.dialogs[index]
        for i in range(5):
            sentence = []
            tmp_frame = []
            Frame = dialog['coref_mapped_seq'][i]
            description = dialog['ner_story'][i].lower()
            tokens = nlp.tokenizer(description)
            sentence.extend([self.story_vocab(token.text) for token in tokens])
            tmp_frame.extend([self.frame_vocab(F) for F in Frame])

            if len(sentence) > self.max_sentence_len-1:
                sentence = sentence[:self.max_sentence_len-1]
            if len(tmp_frame) > self.max_term_len-1:
                tmp_frame = tmp_frame[:self.max_term_len-1]

            frame.append(tmp_frame)
            story.append(sentence)

        S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos = [], [], [], [], [], []

        sen_len = len(list(zip(story, frame)))+1
        for i, (s, f ) in enumerate(zip(story, frame)):
            Ss, Ss_sen_pos, Ss_word_pos, Fs, Fs_sen_pos, Fs_word_pos = [], [], [], [], [], []
            Ss.append(Constants.BOSs[0])
            Fs.append(Constants.BOSs[0])
            Ss.extend(s)
            Fs.extend(f)
            Ss.append(Constants.EOS)
            #Fs.append(Constants.EOS)
            #print(S)
            #+2 becauae in train_2, I added Constants EOS
            Ss_sen_pos.extend([i+1]*(len(s)+2)) 
            #print(S_sen_pos)     
            Ss_word_pos.extend([i+1 for i in range(len(s)+2)])
            #print(S_word_pos)     
            Fs_sen_pos.extend([i+1]*(len(f)+1))
            Fs_word_pos.extend([i+1 for i in range(len(f)+1)])
            
            S.append(Ss)
            S_sen_pos.append(Ss_sen_pos)
            S_word_pos.append(Ss_word_pos)
            
            F.append(Fs)
            F_sen_pos.append(Fs_sen_pos)
            F_word_pos.append(Fs_word_pos)
            

            #print(len(S), len(S_sen_pos))
            assert len(Ss) == len(Ss_sen_pos)
            assert len(Ss) == len(Ss_word_pos)
            assert len(Fs) == len(Fs_sen_pos)
            assert len(Fs) == len(Fs_word_pos)
        assert len(S) == len(F)
        return S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos, self.hop, len(S)

    def __len__(self):
        return len(self.dialogs)


class ROCAddTermsetDataset(Dataset):
    def __init__(self,
                 story_vocab,
                 frame_vocab,
                 text_path,
                 hop,
                 is_verb=True):
        print(text_path)
        self.dialogs = json.load(open(text_path, 'r'))
        self.max_term_len = 9
        self.max_sentence_len = 23
        self.story_vocab = story_vocab
        self.frame_vocab = frame_vocab
        self.hop = hop

    def __getitem__(self, index):
        dialog = self.dialogs[index]
        wS, wS_sen_pos, wS_word_pos, wF, wF_sen_pos, wF_word_pos = [], [], [], [], [], []
        print('len(dialog)-4',len(dialog)-4)
        for window in range(len(dialog)-4):
            frame = []
            story = []
            for i in range(5):
                sentence = []
                tmp_frame = []
                Frame = dialog[i+window]['predicted_term_seq']
                description = dialog[i+window]['text'].lower()
                tokens = nlp.tokenizer(description)
                sentence.extend([self.story_vocab(token.text) for token in tokens])
                tmp_frame.extend([self.frame_vocab(F) for F in Frame])

                if len(sentence) > self.max_sentence_len-1:
                    sentence = sentence[:self.max_sentence_len-1]
                if len(tmp_frame) > self.max_term_len-1:
                    tmp_frame = tmp_frame[:self.max_term_len-1]

                frame.append(tmp_frame)
                story.append(sentence)

            S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos = [], [], [], [], [], []
            for i, (s, f ) in enumerate(zip(story, frame)):
                Ss, Ss_sen_pos, Ss_word_pos, Fs, Fs_sen_pos, Fs_word_pos = [], [], [], [], [], []
                Ss.append(Constants.BOSs[0])
                Fs.append(Constants.BOSs[0])
                Ss.extend(s)
                Fs.extend(f)
                Ss.append(Constants.EOS)
                #Fs.append(Constants.EOS)
                #print(S)
                #+2 becauae in train_2, I added Constants EOS
                Ss_sen_pos.extend([i+1]*(len(s)+2)) 
                #print(S_sen_pos)     
                Ss_word_pos.extend([i+1 for i in range(len(s)+2)])
                #print(S_word_pos)     
                Fs_sen_pos.extend([i+1]*(len(f)+1))
                Fs_word_pos.extend([i+1 for i in range(len(f)+1)])

                S.append(Ss)
                S_sen_pos.append(Ss_sen_pos)
                S_word_pos.append(Ss_word_pos)

                F.append(Fs)
                F_sen_pos.append(Fs_sen_pos)
                F_word_pos.append(Fs_word_pos)
            #print(len(F), len(F_sen_pos))
                assert len(Ss) == len(Ss_sen_pos)
                assert len(Ss) == len(Ss_word_pos)
                assert len(Fs) == len(Fs_sen_pos)
                assert len(Fs) == len(Fs_word_pos)
                
            wS.append(S)
            wS_sen_pos.append(S_sen_pos)
            wS_word_pos.append(S_word_pos)
            wF.append(F)
            wF_sen_pos.append(F_sen_pos)
            wF_word_pos.append(F_word_pos)
        return wS, wS_sen_pos, wS_word_pos, wF, wF_sen_pos, wF_word_pos, self.hop

    def __len__(self):
        return len(self.dialogs)

class VISTAddDataset(Dataset):
    def __init__(self,
                 story_vocab,
                 frame_vocab,
                 text_path,
                 hop,
                 is_verb=True):
        print(text_path)
        self.dialogs = json.load(open(text_path, 'r'))
        self.max_term_len = 9
        self.max_sentence_len = 23
        self.story_vocab = story_vocab
        self.frame_vocab = frame_vocab
        self.is_verb = is_verb
        self.hop = hop

    def __getitem__(self, index):
        frame = []
        story = []
        dialog = self.dialogs[index]
        #for i in range(len(dialog)):
        for i in range(5):
            sentence = []
            tmp_frame = []
            Frame = dialog[i]['coref_mapped_seq']
            description = dialog[i]['text'].lower()
            tokens = nlp.tokenizer(description)
            sentence.extend([self.story_vocab(token.text) for token in tokens])
            tmp_frame.extend([self.frame_vocab(F) for F in Frame])

            if len(sentence) > self.max_sentence_len-1:
                sentence = sentence[:self.max_sentence_len-1]
            if len(tmp_frame) > self.max_term_len:
                tmp_frame = tmp_frame[:self.max_term_len]

            frame.append(tmp_frame)
            story.append(sentence)

        S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos = [], [], [], [], [], []

        sen_len = len(list(zip(story, frame)))+1
        for i, (s, f ) in enumerate(zip(story, frame)):
            Ss, Ss_sen_pos, Ss_word_pos, Fs, Fs_sen_pos, Fs_word_pos = [], [], [], [], [], []
            Ss.append(Constants.BOSs[0])
            Fs.append(Constants.BOSs[0])
            Ss.extend(s)
            Fs.extend(f)
            Ss.append(Constants.EOS)
            #Fs.append(Constants.EOS)
            #print(S)
            #+2 becauae in train_2, I added Constants EOS
            Ss_sen_pos.extend([i+1]*(len(s)+2)) 
            #print(S_sen_pos)     
            Ss_word_pos.extend([i+1 for i in range(len(s)+2)])
            #print(S_word_pos)     
            Fs_sen_pos.extend([i+1]*(len(f)+1))
            Fs_word_pos.extend([i+1 for i in range(len(f)+1)])
            
            S.append(Ss)
            S_sen_pos.append(Ss_sen_pos)
            S_word_pos.append(Ss_word_pos)
            
            F.append(Fs)
            F_sen_pos.append(Fs_sen_pos)
            F_word_pos.append(Fs_word_pos)
               
            #print(len(S), len(S_sen_pos))
            assert len(Ss) == len(Ss_sen_pos)
            assert len(Ss) == len(Ss_word_pos)
            assert len(Fs) == len(Fs_sen_pos)
            assert len(Fs) == len(Fs_word_pos)
        assert len(S) == len(F)
        return S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos, self.hop, len(S)

    def __len__(self):
        return len(self.dialogs)

class VISTTestDataset(Dataset):
    def __init__(self,
                 story_vocab,
                 frame_vocab,
                 text_path,
                 hop,
                 is_verb=True):
        print(text_path)
        self.dialogs = json.load(open(text_path, 'r'))
        self.max_term_len = 9
        self.max_sentence_len = 23
        self.story_vocab = story_vocab
        self.frame_vocab = frame_vocab
        self.is_verb = is_verb
        self.hop = hop

    def __getitem__(self, index):
        frame = []
        story = []
        dialog = self.dialogs[index]
        for i in range(len(dialog)):
            sentence = []
            tmp_frame = []
            Frame = dialog[i]['predicted_term_seq']
            description = dialog[i]['text'].lower()
            tokens = nlp.tokenizer(description)
            sentence.extend([self.story_vocab(token.text) for token in tokens])
            tmp_frame.extend([self.frame_vocab(F) for F in Frame])

            if len(sentence) > self.max_sentence_len-1:
                sentence = sentence[:self.max_sentence_len-1]
            if len(tmp_frame) > self.max_term_len:
                tmp_frame = tmp_frame[:self.max_term_len]

            frame.append(tmp_frame)
            story.append(sentence)

        S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos = [], [], [], [], [], []

        sen_len = len(list(zip(story, frame)))+1
        for i, (s, f ) in enumerate(zip(story, frame)):
            Ss, Ss_sen_pos, Ss_word_pos, Fs, Fs_sen_pos, Fs_word_pos = [], [], [], [], [], []
            Ss.append(Constants.BOSs[0])
            Fs.append(Constants.BOSs[0])
            Ss.extend(s)
            Fs.extend(f)
            Ss.append(Constants.EOS)
            #Fs.append(Constants.EOS)
            #print(S)
            #+2 becauae in train_2, I added Constants EOS
            Ss_sen_pos.extend([i+1]*(len(s)+2)) 
            #print(S_sen_pos)     
            Ss_word_pos.extend([i+1 for i in range(len(s)+2)])
            #print(S_word_pos)     
            Fs_sen_pos.extend([i+1]*(len(f)+1))
            Fs_word_pos.extend([i+1 for i in range(len(f)+1)])
            
            S.append(Ss)
            S_sen_pos.append(Ss_sen_pos)
            S_word_pos.append(Ss_word_pos)
            
            F.append(Fs)
            F_sen_pos.append(Fs_sen_pos)
            F_word_pos.append(Fs_word_pos)
               
            #print(len(S), len(S_sen_pos))
            assert len(Ss) == len(Ss_sen_pos)
            assert len(Ss) == len(Ss_word_pos)
            assert len(Fs) == len(Fs_sen_pos)
            assert len(Fs) == len(Fs_word_pos)
        if len(F) == 0:
            print('len(F) ==', len(F))
            return [], [], [], [], [], [], self.hop, len(S)
        return S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos, self.hop, len(S)

    def __len__(self):
        return len(self.dialogs)
    
class VISTDataset(Dataset):
    def __init__(self,
                 story_vocab,
                 frame_vocab,
                 text_path,
                 hop):
        print(text_path)
        self.dialogs = json.load(open(text_path, 'r'))
        self.max_term_len = 9
        self.max_sentence_len = 23
        self.story_vocab = story_vocab
        self.frame_vocab = frame_vocab
        self.hop = hop


    def __getitem__(self, index):
        frame = []
        story = []
        for idx in range(index*5,index*5+5):
            dialog = self.dialogs[idx]
            sentence = []
            tmp_frame = []
            Frame = dialog['coref_mapped_seq']
            #Frame = dialog['predicted_term_seq']
            #description = sen['ner_description']
            description = dialog['text'].lower()
            #tokens = description.strip().split()
            tokens = nlp.tokenizer(description)
            sentence.extend([self.story_vocab(token.text) for token in tokens])
            tmp_frame.extend([self.frame_vocab(F) for F in Frame])

            if len(sentence) > self.max_sentence_len-1:
                sentence = sentence[:self.max_sentence_len-1]
            if len(tmp_frame) > self.max_term_len-1:
                tmp_frame = tmp_frame[:self.max_term_len-1]

            frame.append(tmp_frame)
            story.append(sentence)

        S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos = [], [], [], [], [], []
        for i, (s, f ) in enumerate(zip(story, frame)):
            Ss, Ss_sen_pos, Ss_word_pos, Fs, Fs_sen_pos, Fs_word_pos = [], [], [], [], [], []
            Ss.append(Constants.BOSs[0])
            Fs.append(Constants.BOSs[0])
            Ss.extend(s)
            Fs.extend(f)
            Ss.append(Constants.EOS)
            #Fs.append(Constants.EOS)
            #print(S)
            #+2 becauae in train_2, I added Constants EOS
            Ss_sen_pos.extend([i+1]*(len(s)+2))     
            Ss_word_pos.extend([i+1 for i in range(len(s)+2)])  
            Fs_sen_pos.extend([i+1]*(len(f)+1))
            Fs_word_pos.extend([i+1 for i in range(len(f)+1)])
            
            S.append(Ss)
            S_sen_pos.append(Ss_sen_pos)
            S_word_pos.append(Ss_word_pos)
            
            F.append(Fs)
            F_sen_pos.append(Fs_sen_pos)
            F_word_pos.append(Fs_word_pos)
        #print(len(F), len(F_sen_pos))
            assert len(Ss) == len(Ss_sen_pos)
            assert len(Ss) == len(Ss_word_pos)
            assert len(Fs) == len(Fs_sen_pos)
            assert len(Fs) == len(Fs_word_pos)
        return S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos, self.hop


    def __len__(self):
        return len(self.dialogs)//5

class PredictedVISTDataset(Dataset):
    def __init__(self,
                 story_vocab,
                 frame_vocab,
                 text_path,
                 hop):
        print(text_path)
        self.dialogs = json.load(open(text_path, 'r'))
        self.max_term_len = 9
        self.max_sentence_len = 23
        self.story_vocab = story_vocab
        self.frame_vocab = frame_vocab
        self.hop = hop

    def __getitem__(self, index):
        frame = []
        story = []
        for idx in range(index*5,index*5+5):
            dialog = self.dialogs[idx]
            sentence = []
            tmp_frame = []
            Frame = dialog['text_mapped_with_nouns_and_frame']
            Frame = dialog['predicted_term_seq']
            #description = sen['ner_description']
            description = dialog['text'].lower()
            #tokens = description.strip().split()
            tokens = nlp.tokenizer(description)
            sentence.extend([self.story_vocab(token.text) for token in tokens])
            tmp_frame.extend([self.frame_vocab(F) for F in Frame])

            if len(sentence) > self.max_sentence_len-1:
                sentence = sentence[:self.max_sentence_len-1]
            if len(tmp_frame) > self.max_term_len-1:
                tmp_frame = tmp_frame[:self.max_term_len-1]

            frame.append(tmp_frame)
            story.append(sentence)

        S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos = [], [], [], [], [], []
        for i, (s, f ) in enumerate(zip(story, frame)):
            Ss, Ss_sen_pos, Ss_word_pos, Fs, Fs_sen_pos, Fs_word_pos = [], [], [], [], [], []
            Ss.append(Constants.BOSs[0])
            Fs.append(Constants.BOSs[0])
            Ss.extend(s)
            Fs.extend(f)
            Ss.append(Constants.EOS)
            #Fs.append(Constants.EOS)
            #print(S)
            #+2 becauae in train_2, I added Constants EOS
            Ss_sen_pos.extend([i+1]*(len(s)+2))    
            Ss_word_pos.extend([i+1 for i in range(len(s)+2)])  
            Fs_sen_pos.extend([i+1]*(len(f)+1))
            Fs_word_pos.extend([i+1 for i in range(len(f)+1)])
            
            S.append(Ss)
            S_sen_pos.append(Ss_sen_pos)
            S_word_pos.append(Ss_word_pos)
            
            F.append(Fs)
            F_sen_pos.append(Fs_sen_pos)
            F_word_pos.append(Fs_word_pos)
        #print(len(F), len(F_sen_pos))
            assert len(Ss) == len(Ss_sen_pos)
            assert len(Ss) == len(Ss_word_pos)
            assert len(Fs) == len(Fs_sen_pos)
            assert len(Fs) == len(Fs_word_pos)
        return S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos, self.hop, len(S)


    def __len__(self):
        return len(self.dialogs)//5

# MODIFIED DELETE [CONSTANTS.EOS]
def ROC_collate_fn(data):
    
    # This is for VIST where stories have diverse story lengths, so we will need to make empty tensors to fill the gaps. So that every row has the same amount of sentences. 
    def pad_sequence(story, max_story_len, hop_max_seq_len):
        for i,_ in enumerate(story):
            if len(story[i])<max_story_len:
                for _ in range(max_story_len-len(story[i])):
                    story[i].append([Constants.BOS]+[Constants.PAD for _ in range(hop_max_seq_len-1)])
        return story
    
    def pad_zero_sequence(story, max_story_len, hop_max_seq_len):
        for i,_ in enumerate(story):
            if len(story[i])<max_story_len:
                for _ in range(max_story_len-len(story[i])):
                    story[i].append([Constants.PAD]+[Constants.PAD for _ in range(hop_max_seq_len-1)])
        return story
    
    def pad_all_sequence(stories, s_sen_pos, s_word_pos, frames, f_sen_pos, f_word_pos, max_story_len, hop_max_seq_len, hop_max_frame_len):
        
        stories = pad_sequence(stories, max_story_len, hop_max_seq_len)
        story_gold = pad_zero_sequence(stories, max_story_len, hop_max_seq_len)
        s_sen_pos = pad_zero_sequence(s_sen_pos, max_story_len, hop_max_seq_len)
        s_word_pos = pad_zero_sequence(s_word_pos, max_story_len, hop_max_seq_len)
        
        frames = pad_sequence(frames, max_story_len, hop_max_frame_len)
#         frames_gold = pad_zero_sequence(frames, max_story_len, hop_max_frame_len)
        f_sen_pos = pad_zero_sequence(f_sen_pos, max_story_len, hop_max_frame_len)
        f_word_pos = pad_zero_sequence(f_word_pos, max_story_len, hop_max_frame_len)
        
        
        return stories, s_sen_pos, s_word_pos, story_gold, frames, f_sen_pos, f_word_pos
    
    
    #List of sentences and frames [B,]
    stories, s_sen_pos, s_word_pos, frames, f_sen_pos, f_word_pos, hop, story_length = zip(*data)
    hop = hop[0]
    max_frame_seq_len = 10 
    max_seq_len = 25

    for ss in stories:
        for s in ss:
            assert len(s)<=30, f"len(s) {len(s)} >hop_max_seq_len {30}"
            
            
    max_story_len = max(story_length)
    
    if hop == 1:
        hop_max_frame_len = max_frame_seq_len+1
        hop_max_seq_len = max_seq_len*2
        
        stories, s_sen_pos, s_word_pos, story_gold, frames, f_sen_pos, f_word_pos = pad_all_sequence(stories, s_sen_pos, s_word_pos, frames, f_sen_pos, f_word_pos, max_story_len, max_seq_len, max_frame_seq_len)
        frame_length = len(frames[0])
        target_length = len(stories[0]) #5
                 
        pad_stories =  [[[[Constants.BOS]+[Constants.EOS]+ ss[i][:-1]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - 2 - (len(ss[i])-1))],\
                          ss[i-1]+ ss[i][:-1]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - len(ss[i-1])-(len(ss[i])-1))]][i!=0]\
                          for i,s in enumerate(ss)] for ss in stories]
        pad_s_sen_pos= [[[[Constants.PAD]+[Constants.PAD]+ ss[i][:-1]+ \
                          [Constants.PAD for _ in range(hop_max_seq_len - 2 -(len(ss[i])-1))], \
                          ss[i-1]+ ss[i][:-1]+ \
                          [Constants.PAD for _ in range(hop_max_seq_len - len(ss[i-1])-(len(ss[i])-1))]][i!=0] \
                          for i,s in enumerate(ss)] for ss in s_sen_pos]
        stories_pos =  [[[[Constants.PAD]+[Constants.PAD]+ ss[i][:-1]+ \
                          [Constants.PAD for _ in range(hop_max_seq_len - 2 -(len(ss[i])-1))], \
                          ss[i-1]+ ss[i][:-1]+ \
                          [Constants.PAD for _ in range(hop_max_seq_len - len(ss[i-1])-(len(ss[i])-1))]][i!=0] \
                          for i,s in enumerate(ss)] for ss in s_word_pos]
        
        previous_stories = [[[[Constants.BOS]+[Constants.EOS]+[Constants.PAD for _ in range(hop_max_seq_len - 2)],\
                          ss[i-1]+[Constants.PAD for _ in range(hop_max_seq_len - len(ss[i-1]))]][i!=0]\
                          for i,s in enumerate(ss)] for ss in stories]

        story_gold =   [[[[Constants.PAD]+ ss[i]+ \
                          [Constants.PAD for _ in range(hop_max_seq_len - 1 -(len(ss[i])))], \
                          (len(ss[i-1])-1)*[Constants.PAD]+ ss[i]+ \
                          [Constants.PAD for _ in range(hop_max_seq_len - (len(ss[i-1])-1) - len(ss[i]))]][i!=0] \
                          for i,s in enumerate(ss)] for ss in story_gold]
        

        pad_frame = [[ss[i] +[Constants.PAD for _ in range(hop_max_frame_len-len(ss[i]))] for i,s in enumerate(ss)] for ss in frames]
        pad_f_sen_pos = [[[1]*len(ss[i]) + [Constants.PAD for _ in range(hop_max_frame_len-len(ss[i]))] for i,s in enumerate(ss)] for ss in f_sen_pos]
        frame_pos = [[ss[i] + [Constants.PAD for _ in range(hop_max_frame_len-len(ss[i]))] for i,s in enumerate(ss)] for ss in f_word_pos]        
    
    elif hop == 1.5:  
        max_frame_length = 25
        hop_max_frame_len = max_frame_seq_len*max_frame_length+1
        hop_max_seq_len = max_seq_len*2
        stories, s_sen_pos, s_word_pos, story_gold, frames, f_sen_pos, f_word_pos = pad_all_sequence(stories, s_sen_pos, s_word_pos, frames, f_sen_pos, f_word_pos, max_story_len, max_seq_len, max_frame_seq_len)

        frame_length = len(frames[0]) #5
        target_length = len(stories[0]) #5 
        
        pad_stories =  [[[[Constants.BOS]+[Constants.EOS]+ ss[i][:-1]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - 2 - (len(ss[i])-1))],\
                          ss[i-1]+ ss[i][:-1]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - len(ss[i-1])-(len(ss[i])-1))]][i!=0]\
                          for i,s in enumerate(ss)] for ss in stories]
        pad_s_sen_pos =[[[[Constants.PAD]+[Constants.PAD]+ ss[i][:-1]+ \
                          [Constants.PAD for _ in range(hop_max_seq_len - 2 -(len(ss[i])-1))], \
                          ss[i-1]+ ss[i][:-1]+ \
                          [Constants.PAD for _ in range(hop_max_seq_len - len(ss[i-1])-(len(ss[i])-1))]][i!=0] \
                          for i,s in enumerate(ss)] for ss in s_sen_pos]
        stories_pos =  [[[[Constants.PAD]+[Constants.PAD]+ ss[i][:-1]+ \
                          [Constants.PAD for _ in range(hop_max_seq_len - 2 -(len(ss[i])-1))], \
                          ss[i-1]+ ss[i][:-1]+ \
                          [Constants.PAD for _ in range(hop_max_seq_len - len(ss[i-1])-(len(ss[i])-1))]][i!=0] \
                          for i,s in enumerate(ss)] for ss in s_word_pos]

        story_gold =   [[[[Constants.PAD]+ ss[i]+ \
                          [Constants.PAD for _ in range(hop_max_seq_len - 1 -(len(ss[i])))], \
                          (len(ss[i-1])-1)*[Constants.PAD]+ ss[i]+ \
                          [Constants.PAD for _ in range(hop_max_seq_len - (len(ss[i-1])-1) - len(ss[i]))]][i!=0] \
                          for i,s in enumerate(ss)] for ss in story_gold]
        
        previous_stories = [[[[Constants.BOS]+[Constants.EOS]+[Constants.PAD for _ in range(hop_max_seq_len - 2)],\
                               ss[i-1]+[Constants.PAD for _ in range(hop_max_seq_len - len(ss[i-1]))]][i!=0]\
                               for i,s in enumerate(ss)] for ss in stories]
        
        all_frame = []
        for frame in frames:
            temp_frame = []
            for f in frame:
                temp_frame+=f
            all_frame.append([temp_frame]*len(frame))
            
        all_f_sen_pos = []
        for frame in f_sen_pos:
            temp_frame = []
            for f in frame:
                temp_frame+=f
            masked_frame = []
            for i in range(len(frame)):
                frame_copy = copy.deepcopy(temp_frame)
                frame_copy = np.array(frame_copy)
                frame_copy[frame_copy != (i+1)] = 0
                frame_copy[frame_copy == (i+1)] = 1
                masked_frame.append(frame_copy.tolist())
            all_f_sen_pos.append(masked_frame)
            
        all_f_word_pos = []
        for frame in f_word_pos:
            temp_frame = []
            for f in frame:
                temp_frame+=f
#             temp_frame += [temp_frame[-1]+1]
            all_f_word_pos.append([temp_frame]*len(frame))
                
        pad_frame  = [[s + [Constants.PAD for _ in range(hop_max_frame_len - len(s))] for i,s in enumerate(ss)] for ss in all_frame]
        pad_f_sen_pos = [[s + [Constants.PAD for _ in range(hop_max_frame_len- len(s))] for i,s in enumerate(ss)] for ss in all_f_sen_pos]
        frame_pos = [[s + [Constants.PAD for _ in range(hop_max_frame_len - len(s))] for i,s in enumerate(ss)] for ss in all_f_word_pos]
        
    #lengths = torch.LongTensor(lengths).view(-1,1)
    targets = torch.LongTensor(pad_stories).view(-1, target_length, hop_max_seq_len)
    targets_pos = torch.LongTensor(stories_pos).view(-1, target_length, hop_max_seq_len)
    targets_sen_pos = torch.LongTensor(pad_s_sen_pos).view(-1, target_length, hop_max_seq_len)
    targets_gold = torch.LongTensor(story_gold).view(-1, target_length, hop_max_seq_len)
    previous_targets = torch.LongTensor(previous_stories).view(-1, target_length, hop_max_seq_len)

    #frame_lengths = torch.LongTensor(frame_lengths).view(-1,1)
    frame = torch.LongTensor(pad_frame).view(-1,frame_length,hop_max_frame_len)
    frame_pos = torch.LongTensor(frame_pos).view(-1, frame_length, hop_max_frame_len)
    frame_sen_pos = torch.LongTensor(pad_f_sen_pos).view(-1, frame_length, hop_max_frame_len)
    story_length = torch.IntTensor(story_length)

    frame = frame.transpose(0,1)
    frame_pos = frame_pos.transpose(0,1)
    frame_sen_pos = frame_sen_pos.transpose(0,1) 
    frame_gold = frame
    
    targets = targets.transpose(0,1)  
    targets_pos = targets_pos.transpose(0,1)  
    targets_sen_pos = targets_sen_pos.transpose(0,1)
    targets_gold = targets_gold.transpose(0,1)
    previous_targets = previous_targets.transpose(0,1)
    
    return frame, frame_pos, frame_sen_pos, frame_gold, targets, targets_pos, targets_sen_pos, targets_gold, previous_targets, story_length

def ROC_collate_test_fn(data):
    #List of sentences and frames [B,]
    stories, s_sen_pos, s_word_pos, frames, f_sen_pos, f_word_pos, hop, story_length = zip(*data)
#     print('ROC_collate_test_fn--frames',frames)
    if frames == ([],) and f_sen_pos == ([],) and f_word_pos == ([],):
        return [], [], [], [], [], [], [], [], []
     
    hop = hop[0]
    max_seq_len = 25
    max_frame_seq_len = 10 
    #lengths = [len(x)+1 for x in stories]
    if hop == 1:
        frame_length = len(frames[0]) #5
        target_length = len(stories[0]) #5
        hop_max_frame_len = max_frame_seq_len*2+1
        hop_max_seq_len = max_seq_len*2
        
        pad_stories =  [[[[Constants.BOS]+[Constants.EOS]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - 2)],\
                          ss[i-1]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - len(ss[i-1]))]][i!=0] \
                          for i,s in enumerate(ss)] for ss in stories]
        pad_s_sen_pos =[[[[Constants.PAD]+[Constants.PAD]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - 2)],\
                          ss[i-1]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - len(ss[i-1]))]][i!=0] \
                          for i,s in enumerate(ss)] for ss in s_sen_pos]
        stories_pos =  [[[[Constants.PAD]+[Constants.PAD]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - 2)],\
                          ss[i-1]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - len(ss[i-1]))]][i!=0] \
                          for i,s in enumerate(ss)] for ss in s_word_pos]

        story_gold =   [[[[Constants.PAD]+ ss[i]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - 1 -(len(ss[i])))],\
                          (len(ss[i-1])-1)*[Constants.PAD]+ ss[i]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - (len(ss[i-1])-1) - len(ss[i]))]][i!=0] \
                          for i,s in enumerate(ss)] for ss in stories]
        
        pad_frame = [[ss[i] +[Constants.PAD for _ in range(hop_max_frame_len-len(ss[i]))] for i,s in enumerate(ss)] for ss in frames]
        pad_f_sen_pos = [[ss[i] + [Constants.PAD for _ in range(hop_max_frame_len-len(ss[i]))] for i,s in enumerate(ss)] for ss in f_sen_pos]
        frame_pos = [[ss[i] + [Constants.PAD for _ in range(hop_max_frame_len-len(ss[i]))] for i,s in enumerate(ss)] for ss in f_word_pos]        

    
    elif hop == 1.5:  
        max_frame_length = 25
        hop_max_frame_len = max_frame_seq_len*max_frame_length+1
        hop_max_seq_len = max_seq_len*2
        frame_length = len(frames[0]) #5
        target_length = len(stories[0])
        
        pad_stories =  [[[[Constants.BOS]+[Constants.EOS]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - 2)],\
                          ss[i-1]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - len(ss[i-1]))]][i!=0] \
                          for i,s in enumerate(ss)] for ss in stories]
        pad_s_sen_pos =[[[[Constants.PAD]+[Constants.PAD]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - 2)],\
                          ss[i-1]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - len(ss[i-1]))]][i!=0] \
                          for i,s in enumerate(ss)] for ss in s_sen_pos]
        stories_pos =  [[[[Constants.PAD]+[Constants.PAD]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - 2)],\
                          ss[i-1]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - len(ss[i-1]))]][i!=0] \
                          for i,s in enumerate(ss)] for ss in s_word_pos]

        story_gold =   [[[[Constants.PAD]+ ss[i]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - 1 -(len(ss[i])))],\
                          (len(ss[i-1])-1)*[Constants.PAD]+ ss[i]+\
                          [Constants.PAD for _ in range(hop_max_seq_len - (len(ss[i-1])-1) - len(ss[i]))]][i!=0] \
                          for i,s in enumerate(ss)] for ss in stories]
        
        all_frame = []
        for frame in frames:
            temp_frame = []
            for f in frame:
                temp_frame+=f
#             temp_frame += [Constants.EOS]
            all_frame.append([temp_frame]*len(frame))
            
        all_f_sen_pos = []
        for frame in f_sen_pos:
            temp_frame = []
            for f in frame:
                temp_frame+=f
            if len(temp_frame) < 1:
                print('ROC_collate_test_fn -- frames',frames)
                print('ROC_collate_test_fn -- f_sen_pos',f_sen_pos)
                print('ROC_collate_test_fn -- f_word_pos',f_word_pos)
#             temp_frame += [temp_frame[-1]]
            masked_frame = []
            for i in range(len(frame)):
                frame_copy = copy.deepcopy(temp_frame)
                frame_copy = np.array(frame_copy)
                frame_copy[frame_copy != (i+1)] = 0
                frame_copy[frame_copy == (i+1)] = 1
                masked_frame.append(frame_copy.tolist())
            all_f_sen_pos.append(masked_frame)
            
        all_f_word_pos = []
        for frame in f_word_pos:
            temp_frame = []
            for f in frame:
                temp_frame+=f
#             temp_frame += [temp_frame[-1]+1]
            all_f_word_pos.append([temp_frame]*len(frame))
            
                
        pad_frame  = [[s + [Constants.PAD for _ in range(hop_max_frame_len - len(s))] for i,s in enumerate(ss)] for ss in all_frame]
        pad_f_sen_pos = [[s + [Constants.PAD for _ in range(hop_max_frame_len- len(s))] for i,s in enumerate(ss)] for ss in all_f_sen_pos]
        frame_pos = [[s + [Constants.PAD for _ in range(hop_max_frame_len - len(s))] for i,s in enumerate(ss)] for ss in all_f_word_pos]
        
        
    
    #lengths = torch.LongTensor(lengths).view(-1,1)
    targets = torch.LongTensor(pad_stories).view(-1, target_length, hop_max_seq_len)
    targets_pos = torch.LongTensor(stories_pos).view(-1, target_length, hop_max_seq_len)
    targets_sen_pos = torch.LongTensor(pad_s_sen_pos).view(-1, target_length, hop_max_seq_len)
    targets_gold = torch.LongTensor(story_gold).view(-1, target_length, hop_max_seq_len)
    
    #frame_lengths = torch.LongTensor(frame_lengths).view(-1,1)
    frame = torch.LongTensor(pad_frame).view(-1,frame_length,hop_max_frame_len)
    frame_pos = torch.LongTensor(frame_pos).view(-1, frame_length, hop_max_frame_len)
    frame_sen_pos = torch.LongTensor(pad_f_sen_pos).view(-1, frame_length, hop_max_frame_len)

    frame = frame.transpose(0,1)
    frame_pos = frame_pos.transpose(0,1)
    frame_sen_pos = frame_sen_pos.transpose(0,1) 
    frame_gold = frame
    
    targets = targets.transpose(0,1)  
    targets_pos = targets_pos.transpose(0,1)  
    targets_sen_pos = targets_sen_pos.transpose(0,1)
    targets_gold = targets_gold.transpose(0,1)
    
    story_length = torch.IntTensor(story_length)
    
    return frame, frame_pos, frame_sen_pos, frame_gold, targets, targets_pos, targets_sen_pos, targets_gold, story_length


def ROC_added_termset_collate_fn(data):

    #List of sentences and frames [B,]
    stories, s_sen_pos, s_word_pos, frames, f_sen_pos, f_word_pos, hop = zip(*data)
    stories, s_sen_pos, s_word_pos, frames, f_sen_pos, f_word_pos, hop = stories[0], s_sen_pos[0], s_word_pos[0], frames[0], f_sen_pos[0], f_word_pos[0], hop[0]
    
    raise ValueError('should not be using ROC_added_termset_collate_fn in sentence2sentence model')
    
    return frame, frame_pos, frame_sen_pos, frame_gold, targets, targets_pos, targets_sen_pos, targets_gold



def get_ROC_loader(text, roc_vocab, frame_vocab, batch_size, shuffle, num_workers, hop ,fixed_len=False, is_flat = False):
    ROC = ROCDataset(roc_vocab,
                     frame_vocab,
                     text_path=text,
                     hop = hop)
    data_loader = torch.utils.data.DataLoader(dataset=ROC,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=ROC_collate_fn)
    return data_loader

def get_VIST_loader(VIST, roc_vocab, frame_vocab, batch_size, shuffle, num_workers, hop):

    data_loader = torch.utils.data.DataLoader(dataset=VIST,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=ROC_collate_fn)
    return data_loader


def get_VIST_test_loader(VIST, roc_vocab, frame_vocab, batch_size, shuffle, num_workers, hop):

    data_loader = torch.utils.data.DataLoader(dataset=VIST,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=ROC_collate_test_fn)
    return data_loader


def get_window_loader(VIST, roc_vocab, frame_vocab, batch_size, shuffle, num_workers, hop):

    data_loader = torch.utils.data.DataLoader(dataset=VIST,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=ROC_added_termset_collate_fn)
    return data_loader

#     '''
#     toy set
#     '''
#     frames = [[[2, 55],[2, 395],[2,   26],[2, 278],[2, 85]]]
#     f_sen_pos[0].append([7, 7, 7])
#     f_word_pos[0].append([1, 2, 3])


#     '''
#     testing
#     +1
#     '''
#     stories[0].append([2,51,52,53,54,55,7])
#     s_sen_pos[0].append([7,7,7,7,7,7,7])
#     s_word_pos[0].append([1,2,3,4,5,6,7])
#     frames[0].append([2, 55, 56])
#     f_sen_pos[0].append([7, 7, 7])
#     f_word_pos[0].append([1, 2, 3])
#     '''
#     testing
#     +2
#     '''
#     stories[0].append([2,51,52,53,54,55,7])
#     s_sen_pos[0].append([8,8,8,8,8,8,8])
#     s_word_pos[0].append([1,2,3,4,5,6,7])
#     frames[0].append([2, 395, 236])
#     f_sen_pos[0].append([8, 8, 8])
#     f_word_pos[0].append([1, 2, 3])
#     '''
#     testing
#     +3
#     '''
#     stories[0].append([2,51,52,53,54,55,7])
#     s_sen_pos[0].append([9,9,9,9,9,9,9])
#     s_word_pos[0].append([1,2,3,4,5,6,7])
#     frames[0].append([2, 94, 31])
#     f_sen_pos[0].append([9, 9, 9])
#     f_word_pos[0].append([1, 2, 3])
#     '''
#     testing
#     +4
#     '''
#     stories[0].append([2,51,52,53,54,55,7])
#     s_sen_pos[0].append([10,10,10,10,10,10,10])
#     s_word_pos[0].append([1,2,3,4,5,6,7])
#     frames[0].append([2, 26, 1160])
#     f_sen_pos[0].append([10, 10, 10])
#     f_word_pos[0].append([1, 2, 3])           
