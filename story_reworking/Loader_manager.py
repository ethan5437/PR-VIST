import DataLoader
import pickle
from build_term_vocab import Vocabulary
import torch
class Loaders():

    def __init__(self, opt):
        self.loader ={}
        with open("data/term2story_vocabs/term_vocab.pkl",'rb') as f:
            self.frame_vocab = pickle.load(f)
        with open("data/term2story_vocabs/story_vocab.pkl",'rb') as f:
            self.story_vocab = pickle.load(f)
        print(self.story_vocab("."))

    def get_loaders(self, args):
        STORY_TERM_PATH = "data/ROC/ROC_{}.json"

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

        STORY_TERM_PATH = "data/VIST/VIST_replace_coref_mapped_frame_noun_train_diverse_length.json"
        
        Dataset = DataLoader.VISTAddDataset(self.story_vocab,
                 self.frame_vocab,
                 text_path=STORY_TERM_PATH, 
                 hop = args.hop)
        self.loader['vist_train'] = DataLoader.get_VIST_loader(Dataset,
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size//4+1,
                                                         True, 5, args.hop)

        STORY_TERM_PATH = "data/VIST/VIST_replace_coref_mapped_frame_noun_val_diverse_length.json"        
        Dataset = DataLoader.VISTAddDataset(self.story_vocab,
                     self.frame_vocab,
                     text_path=STORY_TERM_PATH, 
                     hop = args.hop)
        self.loader['vist_val'] = DataLoader.get_VIST_loader(Dataset,
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         False, 5, args.hop)
        
    def get_test_loaders(self, args):
        add_termset = args.term_path
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


        STORY_TERM_PATH = "data/generated_terms/VIST_test_self_output_diverse.json"
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
