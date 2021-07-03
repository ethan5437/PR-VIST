''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import json
from transformer.Translator import Translator
from Loader_manager import Loaders
from build_story_vocab import Vocabulary
from transformer import Constants

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=3,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-device', type=int, required=True)
    parser.add_argument('-hop', required=True, type=float)
    parser.add_argument('-term_path', required=True, type=str)
#     parser.add_argument('-length', required=True, type=int)
    parser.add_argument('-add_term', type=str2bool, nargs='?',const=True, default=False, help="noise in data T/F")#
    parser.add_argument('-combine_loss', action='store_true')
    parser.add_argument('-is_reverse', action='store_true')

    
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    print('opt.add_term',opt.add_term)
    print('combine_loss',opt.combine_loss)
    if 'reverse' in opt.model:
        opt.is_reverse = True
    print('is_reverse',opt.is_reverse)
    # Prepare DataLoader
#     opt.max_token_seq_len = 50
    if opt.hop == 1:
        opt.max_encode_token_seq_len = 10*2+1
        opt.max_token_seq_len = 25*2
    elif opt.hop == 1.5:
        opt.max_encode_token_seq_len = 10*int(25)+1
        opt.max_token_seq_len = 25*2
    elif opt.hop == 2:
        opt.max_encode_token_seq_len = 10*2+1
        opt.max_token_seq_len = 25*3
    elif opt.hop == 2.5:
        opt.max_encode_token_seq_len = 10*int(25)+1
        opt.max_token_seq_len = 25*3
        
    Dataloader = Loaders(opt)
    Dataloader.get_test_loaders(opt)
    if opt.add_term:
        test_loader = Dataloader.loader['add_window_termset']
    else:
        test_loader = Dataloader.loader['vist_term']

    opt.src_vocab_size = len(Dataloader.frame_vocab)
    opt.tgt_vocab_size = len(Dataloader.story_vocab)
    
#     output = json.load(open('../data/VIST/VIST_replace_coref_mapped_frame_noun_test_diverse_length.json'))
    output = json.load(open('../data/generated_terms/VIST_test_self_output_diverse.json'))
    #output = json.load(open('../data/added_path_terms_ACL/best_path_vg_5terms_story.json'))

    term_path = opt.term_path.split('/')[-1]
    term_path = '.'.join(term_path.split('.')[:-1])
    model_path = opt.model.split('/')[0]
    output_filename = f'../data/generated_story_ACL2021/TransLSTM{str(opt.hop)}_{model_path}_term_{term_path}.json'
#     '../data/generated_story_IJCAI/VIST_VG_'+'hop_'+str(opt.hop)+'_best_path_random_5terms_bounded_BIO_set'+'.json'
    print(f'output filename: {output_filename}')
            
    #output = json.load(open('../../commen-sense-storytelling/data/remove_bus_test.json'))
    count=0
    BOS_set = set([2,3,4,5,6,7])
    translator = Translator(opt)

    
    #print('tombstone_NOUN',Dataloader.frame_vocab.word2idx['tombstone_NOUN'])
    #print('I_NOUN',Dataloader.frame_vocab.word2idx['I_NOUN']) 
    #print('i_NOUN',Dataloader.frame_vocab.word2idx['i_NOUN']) 
    print('.',Dataloader.story_vocab.word2idx['.'])
    print('!',Dataloader.story_vocab.word2idx['!'])

    story_count = 0
#     gt_index = int(opt.hop)-1
    hop = opt.hop
    bigger_than_10 = 0
    
    frame_story_dict = {}
    frame_story_dict.update({'':''})
    
    with open("./test/hop"+str(opt.hop)+opt.output, 'w', buffering=1) as f_pred,\
            open("./test/hop"+str(opt.hop)+'gt.txt', 'w', buffering=1) as f_gt,\
            open("./test/hop"+str(opt.hop)+'show.txt', 'w', buffering=1) as f:
        for frame, frame_pos, frame_sen_pos, _, gt_seqs, gt_poss, gt_sen_poss, _, story_len in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            #filtering out empty set in dataset
            if type(frame) is list:
                f_line = ''
            else:
                f_line = ' '.join([Dataloader.frame_vocab.idx2word[idx.item()] for idx in frame[0][0] if idx!=Constants.PAD])
            if f_line in frame_story_dict.keys():
                if type(output[0]) is list:
                    for sentence_index in range(len(output[count])):
                        output[count][sentence_index]['predicted_story']= frame_story_dict[f_line]
                        if sentence_index < len(print_line_list):
                            pass
                        else:
                            pass
                    count+=1
                else:
                    for sentence_index in range(5):
                        output[count]['predicted_story']=frame_story_dict[f_line]
                        count+=1
                continue
                
            if type(frame) is list and type(frame_pos) is list and type(frame_sen_pos) is list:
                print('empty set')
                for sentence_index in range(5):
                    output[count]['predicted_story']=''
                    if sentence_index < len(print_pred_list):
                        output[count]['predicted_sentence'] = ''          
                    else:
                        output[count]['predicted_sentence'] = ''
                    count+=1 
                continue
                    
            if frame.shape[1] == 1 and (frame[0][0][:10].numpy() == [[[2]+[0]*9]]):
                print('empty story')
                for sentence_index in range(5):
                    output[count]['predicted_story']=''
                    output[count]['predicted_sentence'] = ''             
                    count+=1
                continue                    
            pred_sentence_list = []
            pred_sentence_pos_list = []
            pred_sentence_sen_pos_list = []
            pred_seq = torch.tensor([]).long()
            pred_seq_pos = torch.tensor([]).long()
            pred_seq_sen_pos = torch.tensor([]).long()
            print_line_list = []
            #for lstm input
            gt_seq_history = []
            
            for i,(f, f_pos, f_sen_pos, gt_seq, gt_pos, gt_sen_pos) in enumerate(zip(frame, frame_pos, frame_sen_pos, gt_seqs, gt_poss, gt_sen_poss)):
                
                gt_seq_ = gt_seq[0]
                gt_pos_ = gt_pos[0]
                gt_sen_pos_ = gt_sen_pos[0]
             
                if hop == 1 or hop == 1.5:
                    if i == 0:
                        gt_seq_ = gt_seq_
                        gt_pos_ = gt_pos_
                        gt_sen_pos_ = gt_sen_pos_
                        gt_seq_history.append(gt_seq_)
                    if i != 0:
                        pred_sentence_list.append(pred_sentence)
                        pred_sentence_pos_list.append(list(range(1, len(pred_sentence)+1)))
                        pred_sentence_sen_pos_list.append([i]*len(pred_sentence))
                        hop_max_seq_len = len(gt_seq[0])
                        if len(pred_sentence) > 0:
                            pred_sentence = [seq for seq in pred_sentence if seq != Constants.PAD]
                            if pred_sentence[0] != 2: #not equal to BOS
                                pred_sentence = [2] + pred_sentence
                            if pred_sentence[-1] != 7: #not equal to EOS
                                pred_sentence = pred_sentence + [7]                            
                        gt_seq_ = torch.tensor([pred_sentence +[Constants.PAD for _ in range(hop_max_seq_len - len(pred_sentence))]])[0]
                        gt_pos_ = torch.tensor([list(range(1, len(pred_sentence)+1)) +[Constants.PAD for _ in range(hop_max_seq_len - len(pred_sentence))]])[0]
                        gt_sen_pos_ = torch.tensor([[i]*len(pred_sentence) +[Constants.PAD for _ in range(hop_max_seq_len - len(pred_sentence))]])[0]
                        
                        if i >= 2:
                            pred_sentence_flat_list = [item for sublist in pred_sentence_list for item in sublist]
                            pred_sentence_pos_flat_list = [item for sublist in pred_sentence_pos_list for item in sublist]
                            pred_sentence_sen_pos_flat_list = [item for sublist in pred_sentence_sen_pos_list for item in sublist]
                            pred_seq = torch.tensor(pred_sentence_flat_list)
                            pred_seq_pos = torch.tensor(pred_sentence_pos_flat_list)
                            pred_seq_sen_pos = torch.tensor(pred_sentence_sen_pos_flat_list)
                        gt_seq_history.append(gt_seq_)

                f_line = ' '.join([Dataloader.frame_vocab.idx2word[idx.item()] for idx in f[0] if idx!=Constants.PAD])
                #LSTM_modification
                previous_gt_seq_ = torch.stack(gt_seq_history)
                #LSTM_modification
                all_hyp, all_scores = translator.translate_batch(f, f_pos, f_sen_pos, gt_seq_, gt_pos_, gt_sen_pos_, previous_gt_seq_, story_len, pred_seq, pred_seq_pos, pred_seq_sen_pos)
                
                assert len(all_hyp) == 1,(len(all_hyp) == 1)
                idx_seq = all_hyp[0][0]
                index = [i for i, seq in enumerate(idx_seq) if seq == Constants.EOS]
                if index != []:
                    pred_sentence = idx_seq[:index[0]+1]
                else:
                    index = [i for i, seq in enumerate(idx_seq) if seq == 19 or seq == 43]
                    if index != []: 
                        pred_sentence = idx_seq[:index[0]+1] + [7]
                    else:
                        pred_sentence = idx_seq
                print_line = ' '.join([Dataloader.story_vocab.idx2word[idx] for idx in pred_sentence if idx not in BOS_set])
                print_line_list.append(print_line)

            frame_story_dict.update({f_line:' '.join(print_line_list)})
            if type(output[0]) is list:
                for sentence_index in range(len(output[count])):
                    output[count][sentence_index]['predicted_story']=' '.join(print_line_list)
                    if sentence_index < len(print_line_list):
                        pass
                    else:
                        pass
                count+=1
            else:
                for sentence_index in range(5):
                    output[count]['predicted_story']=' '.join(print_line_list)
#                     if sentence_index < len(print_line_list):
#                         output[count]['predicted_sentence'] = print_line_list[sentence_index]                
#                     else:
#                         output[count]['predicted_sentence'] = ''
                    count+=1
                
              
            
            if story_count < 10:
                print('print_line_list',print_line_list)
            story_count+=1
        
    print('[Info] Finished.')
    json.dump(output, open(output_filename,'w'), indent=4)
    
    
if __name__ == "__main__":
    main()
    
    '''
        First input will be a frame of a story with torch.Size([1, 51]), 
        which means 1 story with 5 frames.
        e.g.tensor([[   2,   55,   56,    2,  395,  236,    2,   72, 1851,    2,  278,   96,
            2,   85,   79,    7,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0]])
        The output of translator.translate_batch(f, f_pos, f_sen_pos, gt_seq)'s all_hyp gives you story: 
        e.g. [[[198, 118, 31, 1528, 104, 19, 2, 340, 24, 31, 710, 446, 45, 19, 2, 198, 218, 230, 3076, 19, 2, 20, 527, 175, 38, 390, 19, 2, 198, 736, 230, 142, 19, 7]]]
        which equals to 'i visited a small town . there was a table set up . i found some books . the man looked at me . i played some games .' 
    '''
