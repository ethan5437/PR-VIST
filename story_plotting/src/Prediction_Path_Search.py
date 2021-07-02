import pickle
from collections import defaultdict
import itertools
import operator
from torch.utils.data import DataLoader, Dataset
import torch
import json
from functools import reduce
from itertools import accumulate
import random
import numpy as np
import collections
import Constants
import pickle
from Candidate_Termset_Manager import Candidate_Termset_Manager_Prediction, Vocabulary
# from build_term_vocab import Vocabulary

random.seed(1111)

U_PATH = {}
U_PATH['vist'] = '../data/VIST'
U_PATH['vist_no_pos'] = '../data/VIST'

class Prediction_Path_Search(Dataset):
    def __init__(self, args, mode, word2id, rela2id):
        super(Prediction_Path_Search, self).__init__()
        self.word2id = word2id
        self.rela2id = rela2id
        self.id2word = {v: k for k, v in word2id.items()}
        self.id2rela = {v: k for k, v in rela2id.items()}
        file_path = U_PATH[args.dataset]
        self.args = args
        
        self.remove_set = ['Frame','NOUN','<empty-frame>','inner','inter', '<end-frame>', '<start-frame>']
       
        print('loading origin test data...')
        with open(f'{file_path}/Golden/VIST_coref_nos_mapped_frame_noun_{args.file_type}_list.json') as file:
            self.origin_test_data = json.load(file) 

        obj_num = '5'
        with open(f'{file_path}/External_datasets/VIST_newobj_objs_NOUN_{obj_num}.json') as file:
            self.noun_data = json.load(file)
        
        print('image2term ouptut...')
        with open(f'{file_path}/External_datasets/image2term_new_obj_{args.file_type}.json') as file:
            self.predicted_terms = json.load(file)
        
        print('_get_data...')
        self.data_objs = self._get_data(args, mode, word2id, rela2id)
        print('get_graph_dict...')
        if self.args.graph_choice == 'multi-graph':
            graph_path = f'{file_path}/External_datasets/KG_Multi_Graph_top{args.relation_frequency}.json'
        elif self.args.graph_choice == 'vg-graph':
            graph_path = f'{file_path}/External_datasets/KG_VG_Graph_top{args.relation_frequency}.json'
        elif self.args.graph_choice == 'vist-graph':
            graph_path = f'{file_path}/External_datasets/KG_VIST_Graph_top{args.relation_frequency}.json'
        with open(graph_path) as f:
            self.graph_dict = json.load(f)   
            
        with open(f"../../story_reworking/term2sentence_lstm/term_vocab.pkl",'rb') as f:
            self.frame_vocab = pickle.load(f)
    
    def _get_data(self, args, mode, word2id, rela2id):
        data_objs = []
        print('args.framework',args.framework)
        if args.framework == 'UHop':
            file_path = U_PATH[args.dataset]
            print('file_path',file_path)
            if args.dataset.lower() == 'vist':
                dataset = args.dataset.lower()                
                print('preparing for input test data...')
                input_data = []
                for story_data in self.origin_test_data:
                    story_id = story_data[0]['story_id']
                    photo_ids = [item['photo_flickr_id'] for item in story_data]
#                     obj_sets = [item['coref_mapped_seq'] for photo_id in photo_ids]
                    if args.story_noun_query:
                        obj_sets = [self.predicted_terms[photo_id] for photo_id in photo_ids]
                    else:
                        obj_sets = [self.noun_data[photo_id] for photo_id in photo_ids]

                    noun_set = []
                    noun_pos_set = []
                    for obj_index in range(len(obj_sets)):
                        nouns = obj_sets[obj_index]# + image2term_sets[obj_index]
                        noun_set.extend(nouns)
                        noun_pos_set.extend([obj_index+1]*len(nouns))
                        
                    noun_set = ' '.join(noun_set)
                    noun_set = self._numericalize_str_and_clean_terms(noun_set, word2id, [' '])
                    input_data.append([story_id, photo_ids, [noun_set, noun_pos_set]])
            else:
                raise ValueError('No dataset found')
        print('Data Process Done...')
        print('input_data[0]',input_data[0])
        return input_data

    def _numericalize_str_and_clean_terms(self, string, map2id, dilemeter):        
#         start_token_set = [f'<s{i}>'for i in range(1, 20)]
        if len(dilemeter) == 2:
            string = string.replace(dilemeter[1], dilemeter[0])
        dilemeter = dilemeter[0]
        tokens = string.strip().split(dilemeter)
        tokens = [token for token in tokens if (token not in self.remove_set)]
        tokens = [token.replace('_NOUN','').replace('_Frame','') for token in tokens]  
#         tokens = ['<s>' if token in start_token_set else token for token in tokens]              
        tokens = [token.lower() for token in tokens]
        tokens = [map2id[x] if x in map2id else map2id['<unk>'] for x in tokens]
        return tokens

    def _numericalize_str(self, string, map2id, dilemeter):
        if len(dilemeter) == 2:
            string = string.replace(dilemeter[1], dilemeter[0])
        dilemeter = dilemeter[0]
        tokens = string.strip().split(dilemeter)
        #Remove NOUN and inner/inter
        tokens = [token for token in tokens if token.strip() != '' and (token[-5:]=='Frame' or\
                                                                        token in ['inner','inter', '1','2','3','4','5'])]
        tokens = [map2id[x] if x in map2id else map2id['<unk>'] for x in tokens]
        return tokens
    

    def add_end_token_noun(self, image_objs, current_pos):                
        for i, image_obj in enumerate(image_objs):
            if current_pos == i:
                image_objs[i] = image_objs[i] + [f'<s>_NOUN']
        return image_objs
    
    def get_image_objs(self, photo_id, current_pos):
#         exlcude_empty_frame
        exlcude_start_token = False
        if args.force_one_noun:
            if current_noun in Constants.BOS_LIST:
                exlcude_start_token = True
        if exlcude_start_token:
            image_objs = [self.noun_data[photo_id]+ self.predicted_terms[photo_id]\
                          for photo_id in photo_ids]
        else:
            image_objs = []
            for i, photo_id in enumerate(photo_ids):
                if current_pos == i:
                    if args.is_restrict_vocab:
                        restricted_noun = [noun for noun in self.noun_data[photo_id] if noun in self.frame_vocab.word2idx]
                        image_objs.append(restricted_noun+ self.predicted_terms[photo_id])
                    else:
                        image_objs.append(self.noun_data[photo_id]+ self.predicted_terms[photo_id])
#                     print(i,self.noun_data[photo_id]+ self.predicted_terms[photo_id]+ [f'<s{sentence_counter+1}>_NOUN'])
                else:
                    image_objs.append(self.noun_data[photo_id]+ self.predicted_terms[photo_id])
#                     print(i,self.noun_data[photo_id]+ self.predicted_terms[photo_id])
        if args.is_term_only:
            image_objs = []
            for i, photo_id in enumerate(photo_ids):
                if current_pos == i:
                    image_objs.append(self.predicted_terms[photo_id])
                else:
                    image_objs.append(self.predicted_terms[photo_id])   
#             image_objs = [self.predicted_terms[photo_id]+ [f'<s{sentence_counter+1}>_NOUN'] for photo_id in photo_ids]

        image_objs = self.add_end_token_noun(image_objs, current_pos)
        return image_objs

    def get_hop(self, photo_ids, current_noun, current_pos, item_index, sentence_counter, pre_rela_list, pre_rela_text_list, args):
        random.seed(1111)
        termset_sample_size = args.termset_sample_size
        sample_size = args.sample_size
        sampling = args.is_sampling
        
        #Get photo ids
        photo_ids = [photo_ids[i] for i,photo_id in enumerate(photo_ids)]
        
        image_objs = self.get_image_objs(photo_ids, current_pos)
        manager = Candidate_Termset_Manager_Prediction(image_objs, current_noun, current_pos, self.graph_dict, args)
        candidate_relations, candidate_pos, candidate_nouns = manager.get_candidate_termsets()
        
        #numerizing texts
        numerized_candidates = []           
        for candidate_relation in candidate_relations:
            numerized = self._numericalize_str(candidate_relation, self.rela2id, ['.'])
            numerized_text = self._numericalize_str_and_clean_terms(candidate_relation, self.word2id, ['.', '_'])
            numerized_candidates.append([numerized, numerized_text, pre_rela_list, pre_rela_text_list])
                
        
        if (len(numerized_candidates) > termset_sample_size) and sampling:
            combine_list = list(zip(numerized_candidates, candidate_pos, candidate_nouns))
            numerized_candidates, candidate_pos, candidate_nouns = list(zip(*random.sample(combine_list, termset_sample_size)))
            #one batch
            step = [numerized_candidates]
            step_pos = [candidate_pos]
            candidate_nouns = [candidate_nouns]
            
        elif not sampling:
            #the data will be too large, separate each data to aviod out of memory
            step = [numerized_candidates[i:i+termset_sample_size] for i in range(0, len(numerized_candidates), termset_sample_size)]
            step_pos = [candidate_pos[i:i+termset_sample_size] for i in range(0, len(candidate_pos), termset_sample_size)]
            candidate_nouns = [candidate_nouns[i:i+termset_sample_size] for i in range(0, len(candidate_nouns), termset_sample_size)]
            
        else:
            step = [numerized_candidates]
            step_pos = [candidate_pos]
            candidate_nouns = [candidate_nouns]
        return step, step_pos, candidate_nouns
        
#         return output

    def __len__(self):
        return len(self.data_objs)
    def __getitem__(self, index):
        return self.data_objs[index]

        
if __name__ == '__main__':

    with open('/home/EthanHsu/commen-sense-storytelling/UHop/data/VIST/rela2id_abs.json', 'r') as f:
        rela2id =json.load(f)
    word2id_path = '/home/EthanHsu/commen-sense-storytelling/UHop/data/glove.300d.word2id.json' 
    with open(word2id_path, 'r') as f:
        word2id = json.load(f)
    class ARGS():
        def __init__(self):
            self.dataset = 'vist'
            self.mode = 'generation'
            self.ten_obj = False
            self.term_path = 'new_obj'
            self.graph_choice = 'multi-graph'
            self.framework = 'UHop'
            self.story_noun_query = False
            self.force_one_noun = False
            self.is_restrict_vocab = False
            self.is_term_only = False
            self.termset_sample_size = 1000
            self.sample_size = 40
            self.is_start_end_frame = False
            self.is_image_abs_position = True
            self.is_sampling = False
            self.file_type = 'test'
            self.relation_frequency = 10
        pass
    photo_ids = ['1741642', '1741640', '1741632', '1741622', '1741587']
    current_noun = 'people_NOUN'
    current_pos = 0
    item_index = 0
    sentence_counter = 1
    pre_rela_list = [11]
    pre_rela_text_list = [11]
    
    args = ARGS()
    dataset = Test_Dataset_Path_Search(args, 'generation', word2id, rela2id)
    result = dataset.get_step(photo_ids, current_noun, current_pos, item_index, sentence_counter, pre_rela_list, pre_rela_text_list, args)
    print('result',result)

