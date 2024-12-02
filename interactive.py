import os
import sys
import argparse
import gc
from nltk import sent_tokenize

os.environ['TRANSFORMERS_CACHE'] = '/home/people/20202939/scratch/trans_cache/'
parser = argparse.ArgumentParser()
parser.add_argument("--model", default='/home/people/20202939/scratch/chatbot/ParlAI/data/models/blenderbot2/blenderbot2_3B/norm/model', required=False, type=str)
parser.add_argument("--log", default='', required=False, type=str)
parser.add_argument("--start", default=0, required=False, type=int)
parser.add_argument("--end", default=50, required=False, type=int)
parser.add_argument('--cls', action='store_true')
parser.add_argument('--contrastive', action='store_true')
parser.add_argument('--baseline', action='store_true')

args = parser.parse_args()
NUM_DOCS_MEMORY = 30

from utils import set_seed, generateNgram, chunk_memory
set_seed(42)

sys.path.append('./Redundant_Classifier/')
from redudant_classifier import Redundant_Classifier
redudant_model = Redundant_Classifier('roberta-large', './Redundant_Classifier/ckpt')

from blender_opt import blender_opt_3B, blender_opt_400M
blender_opt = blender_opt_3B
blender_opt['override']['n_docs'] = NUM_DOCS_MEMORY
blender_opt['override']['model_file'] = args.model
blender_opt['model_file'] = args.model

import os,sys
from parlai.core.agents import create_agent
import yaml
import uuid
from nltk import sent_tokenize
import gc
import random
import time
import json
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
import os, sys
import uuid
import random
from tqdm import tqdm
from utils import extract_questions
import torch

BLOCK_LIST = []
FAILED_STS = []
count = 0

def get_context_representations(model, contexts):
    concat_rep = []
    for i in range(len(contexts),1,-1):
        dialog = contexts[:i]
        with torch.no_grad():
            model.bot_agent.reset()
            model.bot_agent.model.long_term_memory.active_memory_slots = []
            model.bot_agent.model.long_term_memory.n_docs = NUM_DOCS_MEMORY

            obs = model.bot_agent.observe({'id': 'localHuman', 'episode_done': True, 
                                        'label_candidates': None, 'text': '\n'.join(dialog[:-1]),
                                        'labels': [dialog[-1]],
                                        })

            batch = model.bot_agent.batchify([obs])
            batch = batch.to('cuda') if torch.cuda.is_available() else batch

            foward_model = model.bot_agent.model
            xs = model.bot_agent._model_input(batch)
            ys = batch.label_vec

            foward_model.longest_label = max(foward_model.longest_label, ys.size(1))
            encoder_states = foward_model.encoder(*xs)

            bsz = ys.size(0)
            seqlen = ys.size(1)
            inputs = ys.narrow(1, 0, seqlen - 1)
            inputs = foward_model._get_initial_forced_decoder_input(bsz, inputs)
            enc_out, enc_mask, input_turns_cnt, docs, doc_scores = encoder_states
            dec_out, new_incr_state = foward_model.seq2seq_decoder(
                inputs, (enc_out, enc_mask), None)
            concat_rep.append(dec_out[0][1:])
    
    return torch.cat(concat_rep,dim=0)

class Blender:
    def __init__(self):
        opt = blender_opt
        self.opt = opt
        self.bot_agent = create_agent(opt, requireModelExists=True)
    
    def predict(self, contexts, beam_min_length = 20, beam_block_ngram = 3, beam_context_block_ngram = 3, 
                beam_delay = 10, temperature = 0.5, inference = 'beam', beam_length_penalty = 0.65,
                topk = 3, topp = 0.5, beam_size = 10, block_list = BLOCK_LIST, sts_similar = False,
                n_docs_memory = NUM_DOCS_MEMORY, personas = None,
                blocking = False, delimeter = '\n',
                add_first_response = None):
        
        self.bot_agent.reset()
        self.bot_agent.model.long_term_memory.active_memory_slots = []
        self.bot_agent.model.long_term_memory.n_docs = n_docs_memory
        self.bot_agent.opt['beam_length_penalty'] = self.bot_agent.beam_length_penalty = beam_length_penalty
        self.bot_agent.opt['beam_block_ngram'] = self.bot_agent.beam_block_ngram = beam_block_ngram
        
        global FAILED_STS
        
        self.bot_agent.opt['beam_min_length'] = self.bot_agent.beam_min_length = beam_min_length
        self.bot_agent.opt['inference'] = self.bot_agent.inference = inference
        self.bot_agent.opt['beam_size'] = self.bot_agent.beam_size = beam_size
        if inference == 'beam':
            temperature = 1.0
        else:
            self.bot_agent.opt['temperature'] = self.bot_agent.temperature = temperature
            self.bot_agent.opt['topk'] = self.bot_agent.topk = topk
            self.bot_agent.opt['topp'] = self.bot_agent.topp = topp

        past_questions = []
        for i in range(len(contexts) - 2, -1, -2):
            past_questions += extract_questions(contexts[i], valid_check = False)
        
        #--- N-gram question blocking ---#
        if blocking:
            questions = []
            for i in range(len(contexts) - 2, -1, -2):
                questions += extract_questions(contexts[i], valid_check = False)

            block_ngrams = questions.copy()
            for i in range(0,len(questions)):
                words = questions[i].split()[1:]
                if len(words) >= 4:
                    block_ngrams += generateNgram(words, ngram = 4)

            self.bot_agent.beam_block_list.clear()
            for ngram in block_ngrams:
                self.bot_agent.beam_block_list.add(ngram)
        #--- N-gram blocking ---#
        
        if personas is not None:
            self.bot_agent.observe({'id': 'localHuman',
                                    'episode_done': True,
                                    'label_candidates': None,
                                    'personas': personas,
                                    'text': delimeter.join(contexts)})
        else:
            self.bot_agent.observe({'id': 'localHuman',
                                    'episode_done': True,
                                    'label_candidates': None,
                                    'text': delimeter.join(contexts)})
        
        responses = self.bot_agent.batch_act([self.bot_agent.observation])[0]
        responses = responses['beam_texts']
        if add_first_response != None:
            responses.insert(0,(add_first_response, -5))
        first_response = responses[0][0]
        
        if sts_similar:
            part_personas = [p.replace("partner's persona: ",'') for p in personas if "partner's persona:" in p]
            part_personas = [sent_tokenize(line) for line in part_personas]
            part_personas = [j for k in part_personas for j in k]
            
            for i in range(0,len(responses)):
                
                new_question = extract_questions(responses[i][0], valid_check = False)
                if len(new_question) == 0:
                    responses[i] = None
                    continue
                
                cur_question = new_question[0]
                recent_dialog = contexts[-1] + '\n' + responses[i][0].replace(cur_question, '</s></s>' + cur_question)
                recent_dialog = recent_dialog[:recent_dialog.rfind('?') + 1]
                
                preds = redudant_model.predict([recent_dialog] * len(part_personas), part_personas)
                for pred in preds:
                    if pred['Redundant'] > 0.5:
                        responses[i] = None
                        break
                
                if responses[i] != None:
                    break

            responses = [r for r in responses if r != None]
            if len(responses) == 0:
                FAILED_STS.append(first_response)
                responses = [(first_response,0)]
        
        responses = [list(z) for z in responses]
        
        return responses
    
    def generate(self, contexts, inference = 'beam', question = "norm", beam_min_length = 20, **kwargs):    
        if len(contexts) >= 2 and '?' not in contexts[-2] + contexts[-1]:
            question = 'force'
        if question == 'force':
            beam_length_penalty = 0.648
        elif question == 'high':
            beam_length_penalty = 0.647
        elif question == 'contrastive':
            global count
            count += 1
            hidden_states = get_context_representations(self, contexts)
            torch.save(hidden_states, './logs/hidden_states/context_' + str(count).zfill(4) + '.pt')
            beam_length_penalty = 0.6501
        elif question == 'disable':
            beam_length_penalty = 0.649
            beam_min_length = 15
        else:
            beam_length_penalty = 0.650
        
        return self.predict(contexts, beam_min_length = beam_min_length,
                            beam_length_penalty = beam_length_penalty, **kwargs)

model = Blender()

LOG_PATH = args.log
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

dialog_seeds = []
for i in range(args.start,args.end):
    dialog_seeds.append(open('dialog_seeds/' + str(i) + '.txt').read().splitlines())

if args.baseline:
    for dialog_idx in tqdm(range(args.start,args.end)):
        dialogs = dialog_seeds[dialog_idx]
        max_turn = len(dialogs) + 40
        q_logs = []

        while True:
            reponse  = model.generate(dialogs, beam_size = 10, question = 'high',
                                      sts_similar = args.cls, blocking = True, delimeter = '\n')

            dialogs.append(reponse[0][0])
            print(dialogs[-1])

            if len(dialogs) >= max_turn:
                print('end session...\n')
                print('\n............\n')
                file_name = 'bot_' + str(dialog_idx)

                save_dialogs = dialogs.copy()
                with open(os.path.join(LOG_PATH,file_name + '.txt'),'w') as f:
                    f.write('\n'.join(save_dialogs))
                break
    
    # -------------------- PERSONA EXTRACTION -------------------------------
    from persona_model import PersonaModel, MemoryDecoder
    from utils import extract_questions
    model_persona = MemoryDecoder()
    
    base_convs = []
    for i in range(args.start,args.end):
        base_convs.append(os.path.join(LOG_PATH,'bot_' + str(i) + '.txt'))
    base_convs = [open(p).read().splitlines() for p in base_convs]

    persona_map = []
    for conv in tqdm(base_convs):
        decode_texts = []
        for j in range(0,len(conv)):
            last_question = extract_questions(conv[j-1], valid_check = False) if j >= 1 else []
            decode_text = last_question[0] + '\n' + conv[j] if len(last_question) >= 1 else conv[j]
            decode_texts.append(decode_text)
        persona_map.append(model_persona.generate_memories(decode_texts))

    import pickle
    pickle.dump(persona_map, open(os.path.join(LOG_PATH,'personas.pkl'),'wb'))

else:
    import pickle
    persona_map = pickle.load(open('self_chat_conv/baseline/personas.pkl','rb'))

    base_convs = []
    for i in range(args.start,args.end):
        base_convs.append('self_chat_conv/baseline/bot_' + str(i) + '.txt')
    base_convs = [open(p).read().splitlines() for p in base_convs]

    for i in tqdm(range(args.start,args.end)):
        prev_dialog = dialog_seeds[i].copy()
        new_dialog  = base_convs[i][len(prev_dialog):].copy()
        full_dialog = base_convs[i].copy()

        for j in range(0, len(new_dialog)):
            if '?' in new_dialog[j]:
                cur_idx = len(prev_dialog) + j
                prev_personas = list(reversed(persona_map[i][:cur_idx]))
                part_personas = [k for z in prev_personas[0::2] for k in z]
                self_personas = [k for z in prev_personas[1::2] for k in z]

                part_personas = chunk_memory(part_personas, "partner's persona:")
                self_personas = chunk_memory(self_personas, 'your persona:')

                if args.contrastive:
                    reponse  = model.generate(prev_dialog + new_dialog[:j], question = 'contrastive',
                                              personas = part_personas + self_personas,
                                              blocking = True, sts_similar = args.cls, delimeter = '\n')[0][0]
                else:
                    reponse  = model.generate(prev_dialog + new_dialog[:j], question = 'force',
                                              personas = part_personas + self_personas,
                                              blocking = True, sts_similar = args.cls, delimeter = '\n')[0][0]

                print("Old: ", full_dialog[cur_idx])
                full_dialog[cur_idx] = reponse
                print("New: ", full_dialog[cur_idx])
                print()

    #     print('end session...\n')
    #     print('\n............\n')
        file_name = 'bot_' + str(i)

        with open(os.path.join(LOG_PATH,file_name + '.txt'),'w') as f:
            f.write('\n'.join(full_dialog))
