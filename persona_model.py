import os,sys
from parlai.core.agents import create_agent
import yaml
import uuid
from nltk import sent_tokenize
import gc
import argparse
import random
import time
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/people/20202939/scratch/trans_cache/'

MODEL_FILE = './ParlAI-1.6.0/data/models/blenderbot2/blenderbot2_3B/model'

persona_opt = {'init_opt': None, 'allow_missing_init_opts': False, 'task': 'interactive', 'download_path': None, 
               'loglevel': 'info', 'datatype': 'train', 'image_mode': 'raw', 'hide_labels': False, 
               'multitask_weights': [1], 'batchsize': 1, 'dynamic_batching': None, 'verbose': False, 
               'is_debug': False, 'datapath': './ParlAI-1.6.0/data', 'model': None, 
               'model_file': 'zoo:msc/dialog_summarizer/model', 'init_model': None, 
               'dict_class': 'parlai.core.dict:DictionaryAgent', 'display_examples': False, 
               'display_prettify': False, 'display_add_fields': '', 'interactive_task': True, 'outfile': '', 
               'save_format': 'conversations', 'local_human_candidates_file': None, 'single_turn': False, 
               'log_keep_fields': 'all', 'image_size': 256, 'image_cropsize': 224, 'embedding_size': 300, 
               'n_layers': 2, 'ffn_size': 300, 'dropout': 0.0, 'attention_dropout': 0.0, 'relu_dropout': 0.0, 
               'n_heads': 2, 'learn_positional_embeddings': False, 'embeddings_scale': True, 'n_positions': None, 
               'n_segments': 0, 'variant': 'aiayn', 'activation': 'relu', 'output_scaling': 1.0, 
               'share_word_embeddings': True, 'n_encoder_layers': -1, 'n_decoder_layers': -1, 
               'model_parallel': False, 'checkpoint_activations': False, 'beam_size': 1, 'beam_min_length': 3, 
               'beam_context_block_ngram': -1, 'beam_block_ngram': -1, 'beam_block_full_context': False,
               'beam_length_penalty': 0.65, 'skip_generation': False, 'inference': 'greedy', 'topk': 10, 
               'topp': 0.9, 'beam_delay': 30, 'beam_block_list_filename': None, 'temperature': 1.0, 
               'compute_tokenized_bleu': False, 'interactive_mode': True, 'embedding_type': 'random', 
               'embedding_projection': 'random', 'fp16': False, 'fp16_impl': 'safe', 'force_fp16_tokens': False, 
               'optimizer': 'sgd', 'learningrate': 1, 'gradient_clip': 0.1, 'adam_eps': 1e-08, 
               'adafactor_eps': (1e-30, 0.001), 'momentum': 0, 'nesterov': True, 'nus': (0.7,), 
               'betas': (0.9, 0.999), 'weight_decay': None, 'rank_candidates': False, 'truncate': -1, 
               'text_truncate': None, 'label_truncate': None, 'history_reversed': False, 'history_size': -1,
               'person_tokens': False, 'split_lines': False, 'use_reply': 'label', 'add_p1_after_newln': False, 
               'delimiter': '\n', 'history_add_global_end_token': None, 'special_tok_lst': None, 'gpu': -1,
               'no_cuda': False, 'dict_file': None, 'dict_initpath': None, 'dict_language': 'english',
               'dict_max_ngram_size': -1, 'dict_minfreq': 0, 'dict_maxtokens': -1,
               'dict_nulltoken': '__null__', 'dict_starttoken': '__start__', 'dict_endtoken': '__end__', 
               'dict_unktoken': '__unk__', 'dict_tokenizer': 're', 'dict_lower': False, 'bpe_debug': False, 
               'dict_textfields': 'text,labels', 'bpe_vocab': None, 'bpe_merge': None, 'bpe_add_prefix_space': None,
               'bpe_dropout': None, 'lr_scheduler': 'reduceonplateau', 'lr_scheduler_patience': 3, 'lr_scheduler_decay': 0.5,
               'invsqrt_lr_decay_gamma': -1, 'warmup_updates': -1, 'warmup_rate': 0.0001, 'update_freq': 1,
               'parlai_home': './ParlAI-1.6.0', 
               'override': {'model_file': 'zoo:msc/dialog_summarizer/model',
                            'datapath': './ParlAI-1.6.0/data',
                            'memory_decoder_model_file': '',
                            'memory_key': 'personas',
                           },
               'starttime': 'Jun16_21-16'}
               
import os,sys
from parlai.core.agents import create_agent
import yaml
import uuid
from nltk import sent_tokenize
import gc
import argparse
import random
import time

class PersonaModel:
    def __init__(self):
        
        opt = persona_opt
        self.opt = persona_opt
        self.bot_agent = create_agent(opt, requireModelExists=True)     
        
    def generate_memories(self,contexts, beam_min_length = 3, beam_block_ngram = 3, beam_context_block_ngram = -1, 
                beam_delay = 10, temperature = 0.5, inference = 'beam', beam_max_length = 30,
                beam_length_penalty = 0.65, topk = 5, topp = 0.9, beam_size = 5, block_list = [], delimeter = '\n'):

        self.bot_agent.reset()
        
        try:
            self.bot_agent.model.long_term_memory.active_memory_slots = []
            self.bot_agent.model.long_term_memory.n_docs = 2
        except:
            pass
        
        self.bot_agent.opt['beam_min_length'] = self.bot_agent.beam_min_length = beam_min_length
        self.bot_agent.opt['beam_max_length'] = self.bot_agent.beam_max_length = beam_max_length
        
        self.bot_agent.opt['inference'] = self.bot_agent.inference = inference
        if inference == 'beam':
            temperature = 1.0
        self.bot_agent.opt['temperature'] = self.bot_agent.temperature = temperature
        self.bot_agent.opt['topk'] = self.bot_agent.topk = topk
        self.bot_agent.opt['topp'] = self.bot_agent.topp = topp
        self.bot_agent.opt['beam_size'] = self.bot_agent.beam_size = beam_size
        self.bot_agent.opt['beam_block_ngram'] = self.bot_agent.beam_block_ngram = beam_block_ngram
        self.bot_agent.opt['beam_context_block_ngram'] = self.bot_agent.beam_context_block_ngram = beam_context_block_ngram
        self.bot_agent.opt['beam_delay'] = self.bot_agent.beam_delay = beam_delay
        self.bot_agent.opt['beam_block_full_context'] = self.bot_agent.beam_block_full_context = False
        self.bot_agent.opt['beam_length_penalty'] = self.bot_agent.beam_length_penalty = beam_length_penalty

        contexts = contexts.split('\n')
        contexts = delimeter.join(contexts)
        self.bot_agent.observe({'id': 'localHuman',
                                'episode_done': False,
                                'label_candidates': None,
                                'text': contexts})
        
        responses =  self.bot_agent.batch_act([self.bot_agent.observation])[0]
        return responses['text']

import sys
sys.path.append('./ParlAI-1.6.0/projects/blenderbot2/agents/')
from sub_modules import BB2SubmoduleMixin
from enum import Enum, auto
import os
import string
import time
import torch
import torch.nn
from typing import List, Tuple, Dict, Optional, Any
from parlai.agents.rag.retrievers import clean_vec
from parlai.core.agents import create_agent_from_model_file, create_agent_from_shared
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.torch_agent import TorchAgent
from parlai.tasks.msc.agents import NOPERSONA
import parlai.utils.logging as logging

class MemoryDecoder(BB2SubmoduleMixin):
    """
    Memory decoder.

    Given a line of context input, generate a memory to write.
    """

    def __init__(self, opt = persona_opt):
        self.opt = opt
        self.agents = []
        self.agent_dict = None
        self.generations = []
        self.input_type = 'Memory'
        self.delimiter = opt.get('memory_decoder_delimiter', '\n')
        self.one_line_memories = opt.get('memory_decoder_one_line_memories', False)
        model_file = modelzoo_path(opt['datapath'], 'zoo:blenderbot2/memory_decoder/model')
        if model_file and os.path.exists(model_file):
            overrides = {
                'skip_generation': False,
                'inference': 'beam',
                'beam_size': 3,
                'beam_min_length': 3,
                'beam_max_length': 30,
                'beam_block_ngram': 4,
            }
            if self.opt.get('memory_decoder_truncate', -1) > 0:
                overrides['text_truncate'] = self.opt['memory_decoder_truncate']
                overrides['truncate'] = self.opt['memory_decoder_truncate']
            base_agent = create_agent_from_model_file(
                model_file, opt_overrides=overrides
            )
            assert isinstance(base_agent, TorchAgent)
            self.agents = [base_agent]
            assert isinstance(self.agents[0], TorchAgent)
            copies = max(100, (opt['batchsize'] * opt.get('rag_turn_n_turns', 1)))
            self.agents += [
                create_agent_from_shared(self.agents[0].share()) for _ in range(copies)
            ]
            self.agent_dict = self.agents[0].build_dictionary()

    def generate_memories(self,context_lines):
        """
        Generate memories from input.

        Each input is split into the lines of conversational context.
        These are considered independently.

        We then assign a prefix ("your/partner's persona:") dependent on
        whether the bot or it's partner said the line.

        :param input:
            input to the memory decoder
        :param num_inputs:
            number of lines per batch item
        return : ['persona 1', 'no persona', 'persona 3']
        desired = [['persona 1','persona 2'], [], ['persona 3']]
        """
        memories = self._batch_generate(context_lines)
        memories = [sent_tokenize(memory) if memory != '__NO__PERSONA__BEAM__MIN__LEN__20__' else [] for memory in memories]
        return memories