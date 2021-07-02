import json
import copy
from tqdm import tqdm
import argparse
import random
import collections
import itertools
from Candidate_Termset_Manager import Candidate_Termset_Manager_Train
# random.seed(1111)
U_PATH = {}
U_PATH['vist'] = '../data/VIST'
U_PATH['vist_no_pos'] = '../data/VIST'

class Train_Dataset_Path_Search():
    '''
    word2id: word dictionary, matching word to a specific id
    rela2id: relation dictionary, matching relation to a specific id
    id2word: word dictionary, matching id to a specific wrod
    id2rela: relation dictionary, matching id to a specific relation
    mode: train, valid, or test
    remove_set: the special tokens
    
    recurrent_UHop: recurrent UHop will not be used in this framework
    q_representation: choosing types of represetation, e.g., bert
    args: args
    file_path: dataset file path
    graph_dict: knowledge graph
    
    golden path: the golden path of VIST stories. 
        Format: [[SVO tuples for sentence 1]], [SVO tuples for sentence 2], ..., [SVO tuples for sentence N]]
        where SVO = noun.verbframe.noun.
        
    #dead kitten: 
    #occurrence = args.occurrence
    #self.start_token_set = [f'<s{i}>'for i in range(0, 30)]
    '''
    def __init__(self, args, mode, word2id, rela2id):
        super(Train_Dataset_Path_Search, self).__init__()
        self.word2id = word2id
        self.rela2id = rela2id
        self.id2word = {v: k for k, v in word2id.items()}
        self.id2rela = {v: k for k, v in rela2id.items()}
        
        self.mode = mode
        self.remove_set = ['Frame','NOUN','<empty-frame>','inner','inter', '<end-frame>', '<start-frame>']
        
        self.recurrent_UHop = args.recurrent_UHop
        self.q_representation = args.q_representation
        self.args = args
        file_path = U_PATH[args.dataset]
        
        print('Path_Search...')
        if self.mode == 'train':
            golden_path = 'golden_path_train'
        elif self.mode == 'valid':    
            golden_path = 'golden_path_val'
        elif self.mode == 'test':    
            golden_path = 'golden_path_test'
            
        if args.is_image_abs_position:
            golden_path = golden_path + '_abs'
        golden_path = golden_path+'.json'
        print(f'loading {golden_path}...')
        with open(f'{file_path}/Golden/{golden_path}') as f:
            self.golden_path_data = json.load(f)

        print('loading relation graph...')
        with open(f'{file_path}/External_datasets/KG_Multi_Graph_top{args.relation_frequency}.json') as f:
            self.graph_dict = json.load(f)
    
    def __getitem__(self, index):   
#         random.seed(1111)
        args = self.args
        golden_items = self.golden_path_data
            
        #get initial_state
        termset_sample_size = args.termset_sample_size
        sample_size = args.sample_size
        is_image_abs_position = args.is_image_abs_position

        golden_item = golden_items[index]
        photo_id, noun_set, initial_state_text, initial_state_pos, golden_path = golden_item
        initial_state = [initial_state_text, initial_state_pos]
        step_list = []
        golden_history = []
        candidate_termset_full_hop = []
        gold_positions_list = []
        story_term_index = 0
        
        #Given golden path (positive samples), creating negative samples. 
        for item_index, path in enumerate(golden_path):
            golden_termset, golden_position_set = path 
            gold_positions_list.extend(golden_position_set)
            
            #Run termset
            for term_index, (term, gold_pos) in enumerate(zip(golden_termset, golden_position_set)): 
                candidate_termsets = []
                golden_relation = list(term.values())[0]
                current_noun = list(term.keys())[0]
                if args.is_beam_search_graph and item_index == 0 and term_index == 0:
                    raise ValueError('not yet is_beam_search_graph')
                else:
                    ###Get one step candidate termsets###
                    candidate_termsets = self.get_one_hop_candidate_termsets(gold_pos=gold_pos, golden_relation=golden_relation, current_noun=current_noun, noun_set=noun_set, term_index=term_index, story_term_index=story_term_index, golden_history=golden_history, last_termset=False)
                    candidate_termset_full_hop.append(candidate_termsets)
                    
                golden_history.append(golden_relation)
                story_term_index+=1

            #Last termset, create termination
            if item_index == len(golden_path)-1:
                candidate_termsets = []
                current_noun = f'<s>_NOUN'
                gold_pos = gold_positions_list[-1]
                gold_positions_list.append(gold_pos)
                golden_relation = None #no golden relation
                
                candidate_termsets = self.get_one_hop_candidate_termsets(gold_pos=gold_pos, golden_relation=golden_relation, current_noun=current_noun, noun_set=noun_set, term_index=term_index, story_term_index=story_term_index, golden_history=golden_history, last_termset=True)
                candidate_termset_full_hop.append(candidate_termsets)   

        ouput_item = [index, initial_state, [candidate_termset_full_hop, gold_positions_list]]
        
        return self._numericalize(ouput_item, 'UHop', 'vist') 
        
        
    def add_end_token_noun(self, noun_set, gold_pos):
        for i, noun in enumerate(noun_set):
            if gold_pos == i: 
                noun_set[i] = noun_set[i]+[f'<s>_NOUN']
        return noun_set
    
    def get_one_hop_candidate_termsets(self, gold_pos, golden_relation, current_noun, noun_set, term_index, story_term_index, golden_history, last_termset=False):
        noun_set = self.add_end_token_noun(noun_set, gold_pos)
        manager = Candidate_Termset_Manager_Train(gold_pos, golden_relation, current_noun, noun_set, term_index, story_term_index, self.graph_dict, golden_history, self.args, last_termset)
        return manager.get_candidate_termsets()
        
    def _numericalize(self, data, framework, dataset=None):
        word2id = self.word2id
        rela2id = self.rela2id
        
        index, ques, step_list = data[0], data[1], data[2]
        if dataset == 'vist':
            ques = data[1][0]
            ques_pos = data[1][1]
            ques_pos = [pos+1 for pos in ques_pos]
            
            step_list = data[2][0]
            gold_pos = data[2][1]
            gold_pos = [pos+1 for pos in gold_pos]
            if self.recurrent_UHop:
                sentence_num = data[2][2]
            
        if self.q_representation == "bert" and dataset != 'vist':
            ques = self._bert_numericalize_str(ques)
        elif dataset == 'vist':
            ques = self._numericalize_str_and_clean_terms(ques, word2id, [' '])
        else:
            ques = self._numericalize_str(ques, word2id, [' '])
            
        if framework == 'baseline':
            tuples = []
            for t in step_list[0]:
                num_rela = self._numericalize_str(t[0], rela2id, ['.'])
                num_rela_text = self._numericalize_str(t[0], word2id, ['.', '_'])
                num_prev = [self._numericalize_str(prev, rela2id, ['.']) for prev in t[1]]
                num_prev_text = [self._numericalize_str(prev, word2id, ['.', '_']) for prev in t[1]]
                tuples.append((num_rela, num_rela_text, num_prev, num_prev_text, t[2])) #t[2] == gold or not
            new_step_list = tuples
        else:
            new_step_list = []
            for step in step_list:
                new_step = []
                for t in step:
                    num_rela = self._numericalize_str(t[0], rela2id, ['.'])            
                    num_rela_text = self._numericalize_str_and_clean_terms(t[0], word2id, ['.', '_'])                    
                    num_prev = [self._numericalize_str(prev, rela2id, ['.']) for prev in t[1]]
                    num_prev_text = [self._numericalize_str_and_clean_terms(prev, word2id, ['.', '_']) for prev in t[1]]
                    new_step.append((num_rela, num_rela_text, num_prev, num_prev_text, t[2]))

                new_step_list.append(new_step)
                
        if dataset == 'vist':
            if self.recurrent_UHop:
                return index, [ques, ques_pos], [new_step_list, gold_pos, sentence_num]
            else:
                return index, [ques, ques_pos], [new_step_list, gold_pos]
        else:
            return index, ques, new_step_list
        
    def _numericalize_str_and_clean_terms(self, string, map2id, dilemeter):
        #print('original str:', string)
        if len(dilemeter) == 2:
            string = string.replace(dilemeter[1], dilemeter[0])
        dilemeter = dilemeter[0]
        tokens = string.strip().split(dilemeter)
        
        tokens = [token for token in tokens if (token not in self.remove_set)]
        tokens = [token.replace('_NOUN','').replace('_Frame','') for token in tokens]  
#         tokens = ['<s>' if token in self.start_token_set else token for token in tokens]              
        tokens = [token.lower() for token in tokens]
        tokens = [map2id[x] if x in map2id else map2id['<unk>'] for x in tokens]
        return tokens

    def _numericalize_str(self, string, map2id, dilemeter):
        #print('original str:', string)
        if len(dilemeter) == 2:
            string = string.replace(dilemeter[1], dilemeter[0])
        dilemeter = dilemeter[0]
        tokens = string.strip().split(dilemeter)
        #Remove NOUN and inner/inter
        tokens = [token for token in tokens if token.strip() != '' and (token[-5:]=='Frame' or token in \
                                                                        ['inner','inter', '1','2','3','4','5'])]

        tokens = [map2id[x] if x in map2id else map2id['<unk>'] for x in tokens]
        
        return tokens
    
    def get_full_length(self):
        length = 0
        for d in self.golden_path_data:
            length += len(d)
        return length
    
    def __len__(self):
        return len(self.golden_path_data)

                    
if __name__ == '__main__':
    with open('../data/VIST/rela2id_abs.json', 'r') as f:
        rela2id =json.load(f)
    word2id_path = '../data/glove.300d.word2id.json' 
    with open(word2id_path, 'r') as f:
        word2id = json.load(f)
    class ARGS():
        def __init__(self):
            self.dataset = 'vist'
            self.mode = 'train'
            self.occurrence = 10
            self.recurrent_UHop = False
            self.q_representation = 'lstm'
#             self.is_seperate_structural = False
            self.is_image_abs_position = True
            self.is_start_end_frame = False
            self.story_noun_query = False
            self.is_story_noun_candidate = False
            self.no_reverse = False
            self.ten_obj = False
            self.is_beam_search_graph = False
            self.termset_sample_size = 150
            self.sample_size = 15
            self.relation_frequency = 10
        pass
    args = ARGS()
    dataset = Train_Dataset_Path_Search(args, 'train', word2id, rela2id)
    print()
    print(0, dataset[100][0])
    print(1, dataset[100][1])
    print(2, dataset[100][2])
    print(len(dataset[100]))
