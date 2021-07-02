import torch
import os
from os import path as os_path

def log_result(num, ques, relas, rela_texts, scores, acc, path, word2id, rela2id):
    id2word = {v: k for k, v in word2id.items()}
    id2rela = {v: k for k, v in rela2id.items()}
    with open(os.path.join(path, 'error.log'), 'a') as f:
        f.write(f'\n{num} ==============================\n')
        q = [id2word[x] for x in ques[0].data.cpu().numpy()]
        f.write(' '.join(q)+'\n') 
        f.write('Correct:\n')
        t = [id2word[x] for x in rela_texts[0].data.cpu().numpy()]
        f.write(' '.join(t)+'\n') 
        r = [id2rela[x] for x in relas[0].data.cpu().numpy()]
        f.write(' '.join(r)+'\n') 
        c_s = scores[0]
        f.write(str(c_s.data.cpu().numpy())+'\n') 
        if acc == 1:
            f.write('Result:Correct!\n')
        else:
            f.write('Result:Incorrect! ====================\n\n')
            for q, r, t, s in zip(ques[1:], relas[1:], rela_texts[1:], scores[1:]):
                if s > c_s:
                    t = [id2word[x] for x in t.data.cpu().numpy()]
                    f.write(' '.join(t)+'\n') 
                    r = [id2rela[x] for x in r.data.cpu().numpy()]
                    f.write(' '.join(r)+'\n') 
                    f.write(str(s.data.cpu().numpy())+'\n') 
        f.write('\n====================================\n')

def find_save_dir(parent_dir, model_name):
    counter = 0
    save_dir = f'../{parent_dir}/{model_name}_{counter}'
    while os.path.exists(save_dir):
        counter += 1
        save_dir = f'../{parent_dir}/{model_name}_{counter}'
    os.mkdir(save_dir)
    print(f'save_dir is {save_dir}')
    return save_dir

def save_model(model, path):
    path = os.path.join(path, 'model.pth')
    torch.save(model.state_dict(), path)
    print(f'save model at {path}')

def save_model_with_result(model, loss, acc, rc, td, td_rc, path):
    path = os.path.join(path, f'model_{loss:.4f}_{acc:.4f}_{rc:.2f}_{td:.2f}_{td_rc:.2f}.pth')
    torch.save(model.state_dict(), path)
    print(f'save model at {path}')

def load_model(model, args):
    path = args.path
    device = f'cuda:{args.device}'
    path = os.path.join(path, 'model.pth')
    print(f'load model from: {path}')
#     print('torch.load(path)',torch.load(path))
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def checkpoint_exist(path):
    if os_path.isfile(path+'/model.pth'):
        return True
    else:
        return False

def get_output_name(filename, args):
#         print('repetitve_penalty',args.repetitve_penalty)
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
    return filename
