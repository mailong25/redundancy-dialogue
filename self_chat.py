import os
import argparse
import torch
from sentence_transformers import CrossEncoder
import gc
os.environ['TRANSFORMERS_CACHE'] = '/home/people/20202939/scratch/trans_cache/'
parser = argparse.ArgumentParser()
parser.add_argument("--model", default='/home/people/20202939/scratch/chatbot/ParlAI/data/models/blenderbot2/blenderbot2_3B/norm/model', required=False, type=str)
parser.add_argument("--log", default='', required=False, type=str)
parser.add_argument("--start", default=0, required=False, type=int)
parser.add_argument("--end", default=100, required=False, type=int)
parser.add_argument('--sts', action='store_true')
parser.add_argument('--blocking', action='store_true')
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--others', action='store_true')

args = parser.parse_args()
NUM_DOCS_MEMORY = 30

def set_seed(seed: int):
    from torch import manual_seed
    from torch.cuda import manual_seed as cuda_manual_seed, manual_seed_all
    from numpy.random import seed as np_seed
    from random import seed as r_seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    manual_seed(seed)
    cuda_manual_seed(seed)
    manual_seed_all(seed)
    np_seed(seed)
    r_seed(seed)
set_seed(42)

def generateNgram(words, ngram = 4, deli = ' '):    
    ngrams = []
    for i in range(0,len(words) - ngram + 1):
        block = words[i:i + ngram]
        ngrams.append(deli.join(block))
    return ngrams


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_sts = CrossEncoder('cross-encoder/quora-roberta-large')
model2 = model_sts.model.to(device)
del model_sts.model
model_sts.model = model2
gc.collect()

blender_opt = {'init_opt': None, 'allow_missing_init_opts': False, 'task': 'interactive', 'download_path': None, 
 'loglevel': 'info', 'datatype': 'train', 'image_mode': 'raw', 'hide_labels': False, 'multitask_weights': [1],
 'batchsize': 1, 'dynamic_batching': None, 'verbose': False, 'is_debug': False, 'datapath': '../data/',
 'model': None, 'model_file': args.model, 'init_model': None,
 'dict_class': 'parlai.core.dict:DictionaryAgent', 'display_examples': False, 'display_prettify': False,
 'display_add_fields': '', 'interactive_task': True, 'outfile': '', 'save_format': 'conversations',
 'local_human_candidates_file': None, 'single_turn': False, 'log_keep_fields': 'all', 'image_size': 256,
 'image_cropsize': 224, 'candidates': 'inline', 'eval_candidates': 'inline', 'interactive_candidates': 'fixed',
 'repeat_blocking_heuristic': True, 'fixed_candidates_path': None, 'fixed_candidate_vecs': 'reuse',
 'encode_candidate_vecs': True, 'encode_candidate_vecs_batchsize': 256, 'train_predict': False,
 'cap_num_predictions': 100, 'ignore_bad_candidates': False, 'rank_top_k': -1, 'return_cand_scores': False,
 'use_memories': False, 'wrap_memory_encoder': False, 'memory_attention': 'sqrt', 'normalize_sent_emb': False,
 'share_encoders': True, 'learn_embeddings': True, 'data_parallel': False, 'reduction_type': 'mean',
 'polyencoder_type': 'codes', 'poly_n_codes': 64, 'poly_attention_type': 'basic', 'poly_attention_num_heads': 4,
 'codes_attention_type': 'basic', 'codes_attention_num_heads': 4, 'embedding_size': 300, 'n_layers': 2,
 'ffn_size': 300, 'dropout': 0.0, 'attention_dropout': 0.0, 'relu_dropout': 0.0, 'n_heads': 2,
 'learn_positional_embeddings': False, 'embeddings_scale': True, 'n_positions': None, 'n_segments': 0,
 'variant': 'aiayn', 'activation': 'relu', 'output_scaling': 1.0, 'share_word_embeddings': True,
 'n_encoder_layers': -1, 'n_decoder_layers': -1, 'model_parallel': False, 'checkpoint_activations': False,
 'generation_model': 'bart', 'query_model': 'bert', 'rag_model_type': 'token', 'thorough': False,
 'n_extra_positions': 0, 'gold_knowledge_passage_key': 'checked_sentence', 'gold_knowledge_title_key': 'title',
 'rag_retriever_query': 'full_history', 'rag_retriever_type': 'dpr', 'retriever_debug_index': None, 'n_docs': 5,
 'min_doc_token_length': 64, 'max_doc_token_length': 256, 'rag_query_truncate': 512, 'print_docs': False,
 'path_to_index': 'zoo:hallucination/wiki_index_compressed/compressed_pq', 'path_to_dense_embeddings': None,
 'dpr_model_file': 'zoo:hallucination/multiset_dpr/hf_bert_base.cp',
 'path_to_dpr_passages': 'zoo:hallucination/wiki_passages/psgs_w100.tsv', 'retriever_embedding_size': 768,
 'tfidf_max_doc_paragraphs': -1, 'tfidf_model_path': 'zoo:wikipedia_full/tfidf_retriever/model',
 'dpr_num_docs': 25, 'poly_score_initial_lambda': 0.5, 'polyencoder_init_model': 'wikito',
 'poly_faiss_model_file': None, 'regret': False, 'regret_intermediate_maxlen': 32, 'regret_model_file': None,
 'regret_dict_file': None, 'regret_override_index': False, 'indexer_type': 'compressed',
 'indexer_buffer_size': 65536, 'compressed_indexer_factory': 'IVF4096_HNSW128,PQ128', 
 'compressed_indexer_gpu_train': False, 'compressed_indexer_nprobe': 64, 'hnsw_indexer_store_n': 128,
 'hnsw_ef_search': 128, 'hnsw_ef_construction': 200, 'rag_turn_n_turns': 2,
 'rag_turn_marginalize': 'doc_then_turn', 'rag_turn_discount_factor': 1.0, 'beam_size': 1,
 'beam_min_length': 1, 'beam_context_block_ngram': -1, 'beam_block_ngram': -1, 'beam_block_full_context': False,
 'beam_length_penalty': 0.65, 'skip_generation': False, 'inference': 'greedy', 'topk': 10, 'topp': 0.9,
 'beam_delay': 30, 'beam_block_list_filename': None, 'temperature': 1.0, 'compute_tokenized_bleu': False,
 'interactive_mode': True, 'embedding_type': 'random', 'embedding_projection': 'random', 'fp16': False,
 'fp16_impl': 'safe', 'force_fp16_tokens': False, 'optimizer': 'adamax', 'learningrate': 0.0001,
 'gradient_clip': 0.1, 'adam_eps': 1e-08, 'adafactor_eps': (1e-30, 0.001), 'momentum': 0, 'nesterov': True, 
 'nus': (0.7,), 'betas': (0.9, 0.999), 'weight_decay': None, 'rank_candidates': False, 'truncate': 1024,
 'text_truncate': None, 'label_truncate': None, 'history_reversed': False, 'history_size': -1,
 'person_tokens': False, 'split_lines': False, 'use_reply': 'label', 'add_p1_after_newln': False, 
 'delimiter': '\n', 'history_add_global_end_token': None, 'special_tok_lst': None, 'gpu': -1, 'no_cuda': False,
 'dict_file': None, 'dict_initpath': None, 'dict_language': 'english', 'dict_max_ngram_size': -1,
 'dict_minfreq': 0, 'dict_maxtokens': -1, 'dict_nulltoken': '__null__', 'dict_starttoken': '__start__',
 'dict_endtoken': '__end__', 'dict_unktoken': '__unk__', 'dict_tokenizer': 're', 'dict_lower': False,
 'bpe_debug': False, 'dict_textfields': 'text,labels', 'bpe_vocab': None, 'bpe_merge': None,
 'bpe_add_prefix_space': None, 'bpe_dropout': None, 'lr_scheduler': 'reduceonplateau',
 'lr_scheduler_patience': 3, 'lr_scheduler_decay': 0.5, 'invsqrt_lr_decay_gamma': -1, 'warmup_updates': -1,
 'warmup_rate': 0.0001, 'update_freq': 1, 't5_model_arch': 't5-base', 't5_model_parallel': False,
 't5_dropout': 0.0, 't5_generation_config': None, 'search_query_generator_model_file': None,
 'search_query_generator_inference': 'greedy', 'search_query_generator_beam_min_length':1,
 'search_query_generator_beam_size': 1, 'search_query_generator_text_truncate': 512, 
 'splitted_chunk_length': 256, 'doc_chunk_split_mode': 'word', 'n_ranked_doc_chunks': 1, 
 'doc_chunks_ranker': 'head', 'woi_doc_chunk_size': 500, 'search_server': 'None',
 'knowledge_access_method': 'memory_only', 'memory_key': 'full_text', 'query_generator_key': 'full_text', 
 'gold_document_key': '__selected-docs__', 'gold_sentence_key': '__selected-sentences__', 
 'gold_document_titles_key': '__select-docs-titles__', 'skip_search_key': 'skip_search', 'insert_gold_docs': False,
 'memory_extractor_phrase': 'persona:', 'retriever_ignore_phrase': 'persona:',
 'query_generator_ignore_phrase': 'persona:', 'query_generator_model_file': 'zoo:blenderbot2/query_generator/model',
 'query_generator_delimiter': '\n', 'query_generator_inference': 'beam', 'query_generator_beam_size': 1, 
 'query_generator_beam_min_length': 2, 'query_generator_truncate': -1, 'memory_retriever_truncate': -1, 
 'retriever_delimiter': '\n', 'share_search_and_memory_query_encoder': False, 'memory_reader_model': None, 
 'memory_doc_title_delimiter': ' / ', 'memory_writer_model': 'bert', 
 'memory_writer_model_file': 'zoo:hallucination/multiset_dpr/hf_bert_base.cp', 
 'add_cleaned_reply_to_history': False, 'memory_decoder_key': 'full_text',
 'memory_decoder_ignore_phrase': 'persona:', 'memory_decoder_model_file': 'zoo:blenderbot2/memory_decoder/model', 
 'memory_decoder_delimiter': '\n', 'memory_decoder_beam_size': 3, 'memory_decoder_beam_min_length': 10, 
 'memory_decoder_truncate': -1, 'memory_decoder_one_line_memories': False, 
 'parlai_home': '/scratch/20202939/chatbot/ParlAI-1.6.0', 
 'override': {'knowledge_access_method': 'memory_only', 
              'n_docs': NUM_DOCS_MEMORY, 
              'parlai_home': '/home/people/20202939/scratch/chatbot/ParlAI',
              'datapath': '/home/people/20202939/scratch/chatbot/ParlAI/data',
              'model_file': args.model,
              'beam_block_ngram': 3,
              'beam_block_full_context': False,
              'memory_decoder_beam_min_length': 3,
              'memory_decoder_model_file': 'zoo:blenderbot2/memory_decoder/model',
              'search_server': 'None'},
              'starttime': 'Oct10_00-22'}

import os,sys
from parlai.core.agents import create_agent
import yaml
import uuid
# from nltk import sent_tokenize
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
FAILED_STS = 0

class Blender:
    def __init__(self):
        opt = blender_opt
        self.opt = opt
        self.bot_agent = create_agent(opt, requireModelExists=True)

    def predict(self, contexts, beam_min_length = 20, beam_block_ngram = 3, beam_context_block_ngram = 3, 
                beam_delay = 10, temperature = 0.5, inference = 'beam', beam_length_penalty = 0.65,
                topk = 3, topp = 0.5, beam_size = 10, block_list = BLOCK_LIST, sts_similar = False,
                n_docs_memory = NUM_DOCS_MEMORY,
                blocking = False, delimeter = '\n',
                add_first_response = None):
        
        self.bot_agent.reset()
        self.bot_agent.model.long_term_memory.active_memory_slots = []
        self.bot_agent.model.long_term_memory.n_docs = n_docs_memory
        self.bot_agent.opt['beam_length_penalty'] = self.bot_agent.beam_length_penalty = beam_length_penalty
        self.bot_agent.opt['beam_block_ngram'] = self.bot_agent.beam_block_ngram = beam_block_ngram
        
        global FAILED_STS
        
        past_questions = []
        for i in range(len(contexts) - 2, -1, -2):
            past_questions += extract_questions(contexts[i], ignore_context_dependent = False)
        
        #--- N-gram question blocking ---#
        if blocking:
            questions = []
            for i in range(len(contexts) - 2, -1, -2):
                questions += extract_questions(contexts[i], ignore_context_dependent = False)

            block_ngrams = questions.copy()
            for i in range(0,len(questions)):
                words = questions[i].split()[1:]
                if len(words) >= 4:
                    block_ngrams += generateNgram(words, ngram = 4)

            self.bot_agent.beam_block_list.clear()
            for ngram in block_ngrams:
                self.bot_agent.beam_block_list.add(ngram)
        #--- N-gram blocking ---#
        
        self.bot_agent.observe({'id': 'localHuman',
                                'episode_done': True,
                                'label_candidates': None,
                                'text': delimeter.join(contexts)})
        
        responses = self.bot_agent.batch_act([self.bot_agent.observation])[0]
        responses = responses['beam_texts']
        if add_first_response != None:
            responses.insert(0,add_first_response)
        first_response = responses[0][0]
        
        if len(past_questions) != 0 and sts_similar:
            repetitive = []
            for i in range(0,len(responses)):
                new_question = extract_questions(responses[i][0], valid_check = False)
                for q in new_question:
                    if q in repetitive:
                        responses[i] = None
                        break
                    else:
                        pair_candidate_and_existing = list(zip([q] * len(past_questions), past_questions))
                        scores = model_sts.predict(pair_candidate_and_existing)
                        if max(scores) > 0.8:
                            print("Filtered:",q)
                            repetitive.append(q)
                            responses[i] = None
                            break

                if responses[i] != None:
                    break

            responses = [r for r in responses if r != None]
            if len(responses) == 0:
                FAILED_STS += 1
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
        elif question == 'disable':
            beam_length_penalty = 0.649
            beam_min_length = 15
        else:
            beam_length_penalty = 0.65
        
        return self.predict(contexts, beam_min_length = beam_min_length,
                            beam_length_penalty = beam_length_penalty, **kwargs)

model = Blender()

LOG_PATH = args.log
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

dialog_seeds = []
for i in range(args.start,args.end):
    dialog_seeds.append(open('dialog_seeds/' + str(i) + '.txt').read().splitlines())

# # # ------------------------ SELF CHAT FOR BASELINE ----------------------------------------------
if args.baseline:
    for dialog_idx in tqdm(range(args.start,args.end)):
        dialogs = dialog_seeds[dialog_idx]
        max_turn = len(dialogs) + 40
        q_logs = []

        while True:
            reponse  = model.generate(dialogs, beam_size = 10, question = 'norm',
                                      sts_similar = False, blocking = False, delimeter = '   ')

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

# # # --------- REGENERATE QUESTIONS GENERATED BY THE BASELINE BY OTHER MODELS ------------------
if args.others:
    base_convs = []
    for i in range(args.start,args.end):
        base_convs.append('self_chat_conversations/baseline/bot_' + str(i) + '.txt')
    base_convs = [open(p).read().splitlines() for p in base_convs]

    for i in tqdm(range(args.start,args.end)):
        prev_dialog = dialog_seeds[i].copy()
        new_dialog  = base_convs[i][len(prev_dialog):].copy()
        full_dialog = base_convs[i].copy()

        for j in range(0, len(new_dialog)):
            if '?' in new_dialog[j]:

                reponse  = model.generate(prev_dialog + new_dialog[:j], beam_size = 10, question = 'force',
                                          blocking = False, sts_similar = False, delimeter = '\n')[0][0]

                cur_idx = len(prev_dialog) + j
                print("Old: ", full_dialog[cur_idx])
                full_dialog[cur_idx] = reponse
                print("New: ", full_dialog[cur_idx])
                print()

        print('end session...\n')
        print('\n............\n')
        file_name = 'bot_' + str(i)

        with open(os.path.join(LOG_PATH,file_name + '.txt'),'w') as f:
            f.write('\n'.join(full_dialog))