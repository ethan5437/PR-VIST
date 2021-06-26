import argparse
import os
#from UHop import UHop
#from Baseline import Baseline
# from Framework import Framework
from VIST_Framework import VIST_Framework
from Generation_Framework import Generation_Framework
from utility import load_model, checkpoint_exist, find_save_dir
import json
import numpy as np
import torch
import Constants
import os
from os import path as os_path
# from build_term_vocab import Vocabulary

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.manual_seed(1111)

parser = argparse.ArgumentParser()
parser.add_argument('--model', action='store', type=str, default=None) # HR-BiLSTM, ABWIM, MVM
parser.add_argument('--framework', action='store', type=str, default='UHop') # UHop, baseline

train_parser = parser.add_mutually_exclusive_group(required=True)   # train + test | test only
train_parser.add_argument('--train', action='store_true')
train_parser.add_argument('--test', action='store_true')
train_parser.add_argument('--generation', action='store_true')

parser.add_argument('--emb_size', action='store', type=int)
parser.add_argument('--path', action='store', type=str, default=None) # for test mode, specify model path
parser.add_argument('--epoch_num', action='store', type=int, default=1000)
parser.add_argument('--hidden_size', action='store', type=int, default=256)
parser.add_argument('--num_filters', action='store', type=int, default=150)
parser.add_argument('--neg_sample', action='store', type=int, default=200)
parser.add_argument('--dropout_rate', action='store', type=float, default=0.0)
parser.add_argument('--learning_rate', action='store', type=float, default=0.0001)
parser.add_argument('--optimizer', action='store', type=str, default='adam')
parser.add_argument('--l2_norm', action='store', type=float, default=0.0)
parser.add_argument('--earlystop_tolerance', action='store', type=int, default=10)
parser.add_argument('--margin', action='store', type=float, default=0.5)
parser.add_argument('--train_step_1_only', action='store', type=bool, default=False)
parser.add_argument('--train_rela_choose_only', action='store', type=bool, default=False)
parser.add_argument('--show_result', action='store', type=bool, default=False)
parser.add_argument('--train_embedding', action='store', type=bool, default=False)
parser.add_argument('--log_result', action='store', type=bool, default=False)
parser.add_argument('--dataset', action='store', type=str) #sq, wq, wq_train1_test2
parser.add_argument('--saved_dir', action='store', type=str, default='saved_model')
parser.add_argument('--hop_weight', action='store', type=float, default=1)
parser.add_argument('--task_weight', action='store', type=float, default=1)
parser.add_argument('--acc_weight', action='store', type=float, default=1)
parser.add_argument('--stop_when_err', action='store_true')
parser.add_argument('--step_every_step', action='store_true')
parser.add_argument('--dynamic', action='store', type=str, default='flatten')
parser.add_argument('--only_one_hop', action='store_true')
parser.add_argument('--reduce_method', action='store', type=str, default='dense')
parser.add_argument('--pretrained_bert', action='store', type=str, default='bert-base-multilingual-cased')
parser.add_argument('--q_representation', action='store', type=str, default='lstm')
parser.add_argument('--device', action='store', type=int, default=1)
parser.add_argument('--occurrence', action='store', type=int, default=2)
parser.add_argument('--path_search', action='store_true')
parser.add_argument('--termset_sample_size', action='store', type=int, default=150)
parser.add_argument('--sample_size', action='store', type=int, default=15)
parser.add_argument('--abs_position', action='store_true')
parser.add_argument('--ethan_position', action='store_true')
parser.add_argument('--is_sampling', action='store_true')
parser.add_argument('--recurrent_UHop', action='store_true')
# parser.add_argument('--term_path', action='store', type=str, default='AAAI')
parser.add_argument('--term_path', choices=['AAAI', 'new_obj', 'noun_only'])
parser.add_argument('--force_one_noun', action='store_true')
parser.add_argument('--story_noun_query', action='store_true')
parser.add_argument('--is_image_abs_position', action='store_true')
parser.add_argument('--is_seperate_structural', action='store_true')
parser.add_argument('--is_term_only', action='store_true')
parser.add_argument('--file_type', action='store', type=str, default='test') #train, test, val
parser.add_argument('--is_story_noun_candidate', action='store_true')
parser.add_argument('--constant_padding', action='store_true')
parser.add_argument('--rela_size', action='store', type=int, default=60)
parser.add_argument('--rela_text_size', action='store', type=int, default=80)
parser.add_argument('--prepend', action='store_true')
parser.add_argument('--small', action='store_true')
parser.add_argument('--only_five_sentences', action='store_true')
parser.add_argument('--only_four2six_sentences', action='store_true')
parser.add_argument('--repetitve_penalty', action='store',type=float, default=1.0)
parser.add_argument('--ten_obj', action='store_true')
parser.add_argument('--is_KG_only', action='store_true')
parser.add_argument('--no_reverse', action='store_true')
parser.add_argument('--is_restrict_vocab', action='store_true')
parser.add_argument('--empty_frame_penalty', action='store',type=float, default=1.0)
parser.add_argument('--is_start_end_frame', action='store_true')
# parser.add_argument('--is_top_30_graph', action='store_true')
parser.add_argument('--relation_frequency', choices=['10', '20', '30'])
parser.add_argument('--is_beam_search_graph', action='store_true')
parser.add_argument('--only_five2seven_sentences', action='store_true')
parser.add_argument('--only_six2seven_sentences', action='store_true')
parser.add_argument('--over5', action='store_true')
parser.add_argument('--graph_choice', choices=['multi-graph', 'vg-graph', 'vist-graph'], default='multi-graph')


args = parser.parse_args()
print(f'args: {args}')

FILE_PATH = '/home/EthanHsu/commen-sense-storytelling/UHop/'

# if args.is_image_abs_position:
#     if not  args.is_seperate_structural:
#         raise ValueError('is_seperate_structural must be true if is_image_abs_position is true')
    
import_model_str = 'from model.{} import Model as Model'.format(args.model)
exec(import_model_str)
if args.train == True:
    args.path = find_save_dir(args.saved_dir, args.model) if args.path == None else args.path
    with open(os.path.join(args.path, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)


#baseline_path, UHop_path = path_finder.path_finder()
#wpq_path = path_finder.WPQ_PATH()
print('Model',Model)
args.Model = Model
if args.framework == 'baseline':
    if args.dataset.lower() == 'wq':
        with open('../data/baseline/KBQA_RE_data/webqsp_relations/rela2id.json', 'r') as f:
            rela2id = json.load(f)
        with open('../data/WQ/main_exp/rela2id.json', 'r') as f:
            rela_token2id =json.load(f)
    elif args.dataset.lower() == 'sq':
        with open('../data/baseline/KBQA_RE_data/sq_relations/rela2id.json', 'r') as f:
            rela2id = json.load(f)
        with open('../data/SQ/rela2id.json', 'r') as f:
            rela_token2id =json.load(f)
    elif args.dataset.lower() == 'exp4':
        with open('../data/PQ/exp4/rela2id.json', 'r') as f:
            rela_token2id = json.load(f)
        with open('../data/PQ/exp4/concat_rela2id.json', 'r') as f:
            rela2id = json.load(f)
    elif 'pq' in args.dataset.lower():
        with open(f'../data/PQ/baseline/{args.dataset.upper()}/concat_rela2id.json', 'r') as f:
            rela2id = json.load(f)
        with open(f'../data/PQ/baseline/{args.dataset.upper()}/rela2id.json', 'r') as f:
            rela_token2id =json.load(f)
    else:
        raise ValueError('Unknown dataset')
elif args.framework == 'UHop':
    if args.dataset == 'wq' or args.dataset == 'WQ':
        with open('../data/WQ/main_exp/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset == 'sq' or args.dataset == 'SQ':
        with open('../data/SQ/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'wq_train1test2':
        with open('../data/WQ/train1test2_exp/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'exp4':
        with open('../data/PQ/exp4/rela2id.json', 'r') as f:
            rela2id = json.load(f)
    elif 'wpq' in args.dataset.lower():
        with open('../data/PQ/exp3/UHop/rela2id.json', 'r') as f:
            rela2id = json.load(f)
    elif args.dataset.lower() == 'pq':
        with open('../data/PQ/PQ/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'pq1':
        with open('../data/PQ/PQ1/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'pq2':
        with open('../data/PQ/PQ2/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'pq3':
        with open('../data/PQ/PQ3/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'pql':
        with open('../data/PQ/PQL/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'pql1':
        with open('../data/PQ/PQL1/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'pql2':
        with open('../data/PQ/PQL2/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'pql3':
        with open('../data/PQ/PQL3/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'vist':
        if args.is_image_abs_position:
            if args.is_start_end_frame:
                with open(f'{FILE_PATH}/data/VIST/rela2id_abs_start_end_frame.json', 'r') as f:
                    rela2id =json.load(f)   
            else:
                with open(f'{FILE_PATH}/data/VIST/rela2id_abs.json', 'r') as f:
                    rela2id =json.load(f)
        else:
            with open(f'{FILE_PATH}/data/VIST/rela2id.json', 'r') as f:
                rela2id =json.load(f)
    elif args.dataset.lower() == 'vist_no_pos':
        with open(f'{FILE_PATH}/data/VIST/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    else:
        raise ValueError('Unknown dataset.')

#print(rela2id)
#print(rela2id['scientist'])
#exit()
word2id_path = f'{FILE_PATH}/data/glove.300d.word2id.json' if args.emb_size == 300 else f'{FILE_PATH}/data/glove.50d.word2id.json' 
word_emb_path = f'{FILE_PATH}/data/glove.300d.word_emb.npy' if args.emb_size == 300 else f'{FILE_PATH}/data/glove.50d.word_emb.npy'
with open(word2id_path, 'r') as f:
    word2id = json.load(f)
word_emb = np.load(word_emb_path)
args.word_embedding = word_emb

if args.framework == 'UHop': 
    args.rela_vocab_size = len(rela2id)
if args.framework == 'baseline':
    args.rela_vocab_size = len(rela_token2id)

# Should introduce UHop here!
device = 'cuda:'+str(args.device)
print('device',device)
if args.framework == 'UHop':
    if args.path_search:
        uhop = VIST_Framework(args, word2id, rela2id)
    else:
        uhop = Framework(args, word2id, rela2id)
    
    
    model = Model(args)
    model = model.to(device)
    print('args.path',args.path)

#     torch.cuda.empty_cache()
    if args.train == True:
        if checkpoint_exist(args.path):
            model = load_model(model, args)
#             print('model.device',next(model.parameters()).is_cuda)
        
        if args.path_search:
            model = uhop.train(model, args)
        else:
            model = uhop.train(model)
        loss, acc, scores, labels = uhop.evaluate(model=None, mode='test', dataset=None, args=args,\
                                                  output_result=True)
    if args.test == True:
        loss, acc, scores, labels = uhop.evaluate(model=None, mode='test', dataset=None, args=args,\
                                                  output_result=True)
    if args.generation == True:
#         print('repetitve_penalty',args.repetitve_penalty)
        filename = args.path.split('/')[-1]
        if args.is_sampling:
            filename = filename + '_is_sampling'
        if args.force_one_noun:
            filename = filename + '_force_one_noun'
        if args.abs_position:
            filename = filename + '_abs_position'
        if args.ethan_position:
            filename = filename + '_ethan_position'
        if args.recurrent_UHop:
            filename = filename + '_recurrent_UHop'
        if args.story_noun_query:
            filename = filename + '_story_noun_query'
        if args.is_image_abs_position:
            filename = filename + '_is_image_abs_position'
        if args.is_seperate_structural:
            filename = filename + '_is_seperate_structural'
        if args.is_term_only:
            filename = filename + '_is_term_only'
        if args.is_story_noun_candidate:
            filename = filename + '_is_story_noun_candidate'
        if args.only_five_sentences:
            filename = filename + '_only_five_sentences'
        if args.only_four2six_sentences:
            filename = filename + '_only_four2six_sentences'
        if args.only_five2seven_sentences:
            filename = filename + '_only_5to7'
        if args.only_six2seven_sentences:
            filename = filename + '_only_6to7'
        if args.over5:
            filename = filename + '_over5'
        if args.is_start_end_frame:
            filename = filename + '_is_start_end_frame'
        if args.is_beam_search_graph:
            filename = filename + '_is_beam_search_graph'

        filename = filename + f'_repetitve_penalty_{args.repetitve_penalty}'
        if args.is_KG_only:
            filename = filename + '_is_KG_only'
        if args.no_reverse:
            filename = filename + '_no_reverse'
        if args.is_restrict_vocab:
            filename = filename + '_is_restrict_vocab'
            
        filename = filename + "_" + args.file_type
        if args.small:
            filename = filename + '_small'
        term_path = args.term_path
        output_file_name = f'../data/generated_terms/pred_terms_{filename}_{term_path}_{args.file_type}.json'
#         output_file_scores_name = f'../data/generated_terms/pred_scores_{filename}_{term_path}.json'
        output_file_rela_name = f'../data/generated_terms/pred_relations_{filename}_{term_path}_{args.file_type}.json'
        print('output_file_name',output_file_name)
        uhop = Generation_Framework(args, word2id, rela2id)
        output, output_with_scores, output_rela = uhop.generation(model=None, mode='generation', dataset=None, output_result=True)
                
        print(f'outputing... {output_file_name}')
        with open(output_file_name,'w') as jsonfile:
            json.dump(output, jsonfile, indent=4)
           
        print(f'outputing... {output_file_rela_name}')
        with open(output_file_rela_name,'w') as jsonfile:
            json.dump(output_rela, jsonfile, indent=4)
            
#         print(f'outputing... {output_file_scores_name}')    
#         with open(output_file_scores_name,'w') as jsonfile:
#             json.dump(output_with_scores, jsonfile, indent=4)
            
        print(f'Done!')

elif args.framework == 'baseline':
    baseline = Framework(args, word2id, rela_token2id)
    model = Model(args).to(device)
    if args.train == True:
        model = baseline.train(model)
        baseline.evaluate(model=None, mode='test', dataset=None, output_result=True)
    if args.test == True:
        baseline.evaluate(model=None, mode='test', dataset=None)
