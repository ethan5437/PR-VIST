import DataLoader
import pickle
from build_term_vocab import Vocabulary
import torch
class Loaders():

    def __init__(self, opt):
        self.loader ={}
        print('term file: /home/joe32140/commen-sense-storytelling/data/term2story_vocabs/term_vocab.pkl')
        with open("/home/joe32140/commen-sense-storytelling/data/term2story_vocabs/term_vocab.pkl",'rb') as f:
            self.frame_vocab = pickle.load(f)
        #with open("../../event-visual-storytelling/data/ROC_Frame_vocab.pkl",'rb') as f:
        with open("/home/joe32140/commen-sense-storytelling/data/term2story_vocabs/story_vocab.pkl",'rb') as f:
#         with open("/home/EthanHsu/commen-sense-storytelling/data/term2story_vocabs/story_vocab.pkl",'rb') as f:
        #with open("../../event-visual-storytelling/data/ROC_Story_vocab.pkl",'rb') as f:
            self.story_vocab = pickle.load(f)
        print(self.story_vocab("."))

    def get_loaders(self, args):
        STORY_TERM_PATH = "/home/EthanHsu/commen-sense-storytelling/data/ROC/ROC_{}.json"

        self.loader['train'] = DataLoader.get_ROC_loader(STORY_TERM_PATH.format('train'),
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         True, 5, args.hop)
        self.loader['val'] = DataLoader.get_ROC_loader(STORY_TERM_PATH.format('valid'),
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size//4+1,
                                                         True, 5, args.hop)
        self.loader['test'] = DataLoader.get_ROC_loader(STORY_TERM_PATH.format('test'),
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         False, 5, args.hop)
        
        
#         STORY_TERM_PATH = "../data/VIST/VIST_coref_nos_mapped_frame_noun_train.json"
#         Dataset = DataLoader.VISTDataset(self.story_vocab,
#                          self.frame_vocab,
#                          text_path=STORY_TERM_PATH, 
#                          hop = args.hop)


        STORY_TERM_PATH = "/home/EthanHsu/commen-sense-storytelling/data/VIST/VIST_replace_coref_mapped_frame_noun_train_diverse_length.json"
        
#         STORY_TERM_PATH = '../data/VIST/VIST_coref_nos_mapped_frame_noun_train_list.json'
        Dataset = DataLoader.VISTAddDataset(self.story_vocab,
                 self.frame_vocab,
                 text_path=STORY_TERM_PATH, 
                 hop = args.hop)
        self.loader['vist_train'] = DataLoader.get_VIST_loader(Dataset,
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size//4+1,
                                                         True, 5, args.hop)
#         STORY_TERM_PATH = "../data/VIST/VIST_coref_nos_mapped_frame_noun_val.json"
#         Dataset = DataLoader.VISTDataset(self.story_vocab,
#                          self.frame_vocab,
#                          text_path=STORY_TERM_PATH,
#                          hop = args.hop)


        STORY_TERM_PATH = "/home/EthanHsu/commen-sense-storytelling/data/VIST/VIST_replace_coref_mapped_frame_noun_val_diverse_length.json"
            
#         STORY_TERM_PATH = '../data/VIST/VIST_coref_nos_mapped_frame_noun_val_list.json'
        
        Dataset = DataLoader.VISTAddDataset(self.story_vocab,
                     self.frame_vocab,
                     text_path=STORY_TERM_PATH, 
                     hop = args.hop)
        self.loader['vist_val'] = DataLoader.get_VIST_loader(Dataset,
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         False, 5, args.hop)
        
        #/home/EthanHsu/commen-sense-storytelling/data/Visual-Genome/visual_genome_paragraph_train.json
        STORY_TERM_PATH = "/home/EthanHsu/commen-sense-storytelling/data/Visual-Genome/visual_genome_paragraph_train_repalce_coref_nos_diverse_length.json"
        Dataset = DataLoader.VISTAddDataset(self.story_vocab,
                 self.frame_vocab,
                 text_path=STORY_TERM_PATH, 
                 hop = args.hop)
        self.loader['vg_train'] = DataLoader.get_VIST_loader(Dataset,
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size//8+1,
                                                         True, 5, args.hop)
        STORY_TERM_PATH = "/home/EthanHsu/commen-sense-storytelling/data/Visual-Genome/visual_genome_paragraph_val_repalce_coref_nos_diverse_length.json"
        Dataset = DataLoader.VISTAddDataset(self.story_vocab,
                     self.frame_vocab,
                     text_path=STORY_TERM_PATH, 
                     hop = args.hop)
        self.loader['vg_val'] = DataLoader.get_VIST_loader(Dataset,
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         False, 5, args.hop)

    def get_test_loaders(self, args):
        #add_termset = 'VIST_replace_coref_mapped_frame_noun_{mode}_diverse_length.json'
        #add_termset = '../data/test_roc_add_term.json'
        #add_termset = '../data/generated_terms_ACL/VIST_VG_3terms_test_self_output_diverse.json'
#         add_termset = '../data/added_path_terms_ACL/best_path_vg_random_5terms_story_bounded_set.json'
        add_termset = args.term_path#'../data/added_path_terms_ACL/5_path_vist_5terms_story.json'

#         add_termset = '/home/wei0401/commen-sense-storytelling/image2term/test/demo_term_list.json'
        #add_termset = '../data/added_path_terms_ACL/6_path_vg_5terms_story.json'
        #add_termset = '../data/added_path_stories/VIST_test_self_output_diverse_added_path_highest_noun2_coor.json'
        #add_termset = '../data/added_path_stories/language_model_terms_add_lowest.json'
        #print(f"test data name:{add_termset}")
        Dataset = DataLoader.VISTTestDataset(self.story_vocab,
                         self.frame_vocab,
                         text_path=add_termset, 
                         hop = args.hop)
        self.loader['add_window_termset'] = DataLoader.get_VIST_test_loader(Dataset,
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         False, 
                                                         6, 
                                                         args.hop)


        STORY_TERM_PATH = "/home/EthanHsu/commen-sense-storytelling/data/generated_terms/VIST_test_self_output_diverse.json"
        Dataset = DataLoader.PredictedVISTDataset(self.story_vocab,
                         self.frame_vocab,
                         text_path=STORY_TERM_PATH,
                         hop = args.hop)
        self.loader['vist_term'] = DataLoader.get_VIST_test_loader(Dataset,
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         False, 
                                                         5, 
                                                         args.hop)
