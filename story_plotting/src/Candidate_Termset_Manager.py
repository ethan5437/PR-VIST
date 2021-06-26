import itertools

###   Candidate_Termset_Manager for Trainining  
class Candidate_Termset_Manager_Train():
    
    '''
    candidate_termsets: list of candidate termsets
    gold_pos: golden golden_relation's image position. (e.g., a noun in image 2, then the position would be 2. )
    golden_relation: golden relation
    current_noun: head entity, the noun you're currently in the story graph. 
    noun_set: nouns in image 1~5, [noun_for_img1, noun_for_img2, noun_for_img3,...,noun_for_img5], where noun_for_imgN is a list of nouns. 
    hop_num: current hop number
    graph_dict: knowledge graph
    golden_history: previous golden path relations
    last_termset: whether is the last hop, will need to set up special candidates for termination.

    Additional notes:
    relation: verb_frame.candidate_noun.structural_relation
    candidate_termset: [relation,self.golden_history[:self.story_term_index],0] 
    '''
    
    def __init__(self, gold_pos, golden_relation, current_noun, noun_set, term_index, story_term_index, graph_dict, golden_history, args, last_termset):
        super(Candidate_Termset_Manager_Train, self).__init__()
        self.candidate_termsets = []
        self.gold_pos = gold_pos
        self.golden_relation = golden_relation
        self.current_noun = current_noun
        self.noun_set = noun_set
#         self.term_index = term_index
        self.hop_num = story_term_index
        self.graph_dict = graph_dict
        self.golden_history = golden_history
        self.args = args
        self.last_termset = last_termset
        
    def get_candidate_termsets(self):
        #<s>_NOUN -- <empty-frame>_Frame -- NOUN inner 
        for noun_set_index, nouns in enumerate(self.noun_set):  
            structural_relation = self.get_structural_relation(noun_set_index)
            for noun in nouns:
                #get noun-empty-frame-none relation
                self.append_empty_frame_candidates(noun, structural_relation)
                #get noun-frame-none relation
                self.append_verb_frame_candidate(noun, structural_relation)
                
        self.candidate_termsets = self.remove_duplicates(self.candidate_termsets)   
        #TODO: check ground-truth
        if not self.last_termset:
            self.candidate_termsets.append([self.golden_relation, self.golden_history[:self.hop_num],1])  
        return self.candidate_termsets
        
    def check_golden_termset(self):
        has_golden = False
        for candidate_term in self.candidate_termsets:
            if candidate_term[-1] == 1:
                has_golden = True
        if not has_golden:
            raise ValueError('candidate_termset has no golden')
            
    def remove_duplicates(self, k):
        k.sort()
        return list(k for k,_ in itertools.groupby(k))
    
    def get_structural_relation(self, noun_set_index):
        if self.args.is_image_abs_position:
            structural_relation = str(noun_set_index+1)
        else:
            if noun_set_index == self.gold_pos:
                structural_relation = 'inner'
            else:
                structural_relation = 'inter'
        return structural_relation
    
    def append_empty_frame_candidates(self, candidate_noun, structural_relation):
        #get noun-empty-frame-none relation
        verb_frame = '<empty-frame>_Frame'
        relation = f'{verb_frame}.{candidate_noun}.{structural_relation}'
        #Appenda nd Remove duplicate relation tuple
        if relation != self.golden_relation:
            candidate_termset = [relation,self.golden_history[:self.hop_num],0]
            self.candidate_termsets.append(candidate_termset)
            
    def append_verb_frame_candidate(self, candidate_noun, structural_relation):
        noun_pair = f'{self.current_noun}+{candidate_noun}'
        if noun_pair in self.graph_dict.keys():
            verb_frames = self.graph_dict[noun_pair]
            for verb_frame in verb_frames:
                relation = f'{verb_frame}.{candidate_noun}.{structural_relation}'
                #Remove duplicate relation tuple
                if relation != self.golden_relation:
                    candidate_termset = [relation,self.golden_history[:self.hop_num],0]
                    self.candidate_termsets.append(candidate_termset)
                    
###   Candidate_Termset_Manager for Prediction                 
class Candidate_Termset_Manager_Prediction():
    '''
    candidate_relations: list of candidate relations, corresponding to the candidate nouns
    candidate_pos: list of position of the candidate relation and noun
    candidate_nouns: list of noun, corresponding to the candidate relation. 

    current_noun: head entity, the noun you're currently in the story graph. 
    current_pos: head entity current image position (e.g., a noun in image 2, then the position would be 2. )
    noun_set: nouns in image 1~5, [noun_for_img1, noun_for_img2, noun_for_img3,...,noun_for_img5], where noun_for_imgN is a list of nouns. 

    Additional notes:
    relation: verb_frame.candidate_noun.structural_relation
    candidate_termset: [relation,self.golden_history[:self.story_term_index],0] 
    '''
        
    def __init__(self, noun_set, current_noun, current_pos, graph_dict, args):
        super(Candidate_Termset_Manager_Prediction, self).__init__()
        self.noun_set = noun_set
        self.current_noun = current_noun
        self.current_pos = current_pos
        self.graph_dict = graph_dict

        self.args = args
        self.candidate_relations = []
        self.candidate_pos = []
        self.candidate_nouns = []
        
    def get_candidate_termsets(self):
        for noun_set_index, nouns in enumerate(self.noun_set):     
            structural_relation = self.get_structural_relation(noun_set_index, self.current_pos)
            for noun in nouns:
                self.append_empty_frame_candidates(noun_set_index, noun, structural_relation)
                self.append_verb_frame_candidate(noun_set_index, noun, structural_relation)
        assert len(self.candidate_nouns) == len(self.candidate_pos), 'len(candidate_nouns) == len(candidate_pos)'
        assert len(self.candidate_nouns) == len(self.candidate_relations), 'len(candidate_nouns) == len(candidate_relations)'

        return self.candidate_relations, self.candidate_pos, self.candidate_nouns

    def get_structural_relation(self, noun_set_index, current_pos):
        if args.is_image_abs_position:
            structural_relation = str(noun_set_index+1)
        else:
            if noun_set_index == current_pos:
                structural_relation = 'inner'
            else:
                structural_relation = 'inter'
        return structural_relation
    
    
    def append_empty_frame_candidates(self, noun_set_index, candidate_noun, structural_relation):
        #get noun-empty-frame-none relation
#       if obj != f'<s{sentence_counter+1}>_NOUN':
        if args.is_start_end_frame:
            if self.current_noun == '<s>_NOUN':
                relation = f'<start-frame>_Frame.{candidate_noun}.{structural_relation}'
            elif obj == '<s>_NOUN':
                relation = f'<end-frame>_Frame.{candidate_noun}.{structural_relation}'
            else:
                relation = f'<empty-frame>_Frame.{candidate_noun}.{structural_relation}'
        else:
            relation = f'<empty-frame>_Frame.{candidate_noun}.{structural_relation}'
        if relation not in self.candidate_relations:
            self.candidate_relations.append(relation)
            self.candidate_pos.append(noun_set_index)
            self.candidate_nouns.append(candidate_noun)
            
    def append_verb_frame_candidate(self, noun_set_index, candidate_noun, structural_relation):
        #GET NOUN --frame-- NOUN inner candidates
        noun_pair = f'{self.current_noun}+{candidate_noun}'
        if noun_pair in self.graph_dict.keys():
            frames = self.graph_dict[noun_pair]
#                     if len(frames) > sample_size and sampling:
#                         frames = random.sample(frames, sample_size)
            for frame in frames:
                relation = f'{frame}.{candidate_noun}.{structural_relation}'
                if relation not in self.candidate_relations:
                    self.candidate_relations.append(relation)
                    self.candidate_pos.append(noun_set_index)
                    self.candidate_nouns.append(candidate_noun)
                    
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
