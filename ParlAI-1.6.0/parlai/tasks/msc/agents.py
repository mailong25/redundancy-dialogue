#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
from parlai.core.opt import Opt
from parlai.utils.strings import normalize_reply
from parlai.core.teachers import MultiTaskTeacher
import inspect
import random
random.seed(10)
from .build import build
import os
import json
from typing import Optional
from parlai.core.params import ParlaiParser
import copy
import random
import math
from parlai.utils.logging import logger
from parlai.core.message import Message
import parlai.scripts.display_data as dsd
from parlai.tasks.convai2.agents import NormalizedTeacherTrait, SelfOriginalTeacher
from parlai.tasks.blended_skill_talk.agents import (
    ContextGenerator as BaseContextGenerator,
)
from parlai.tasks.msc.constants import (
    INITIAL_DATA_TO_COMPLETE,
    MODEL_OPT,
    UI_OPT,
    COMMON_CONFIG,
)
import pickle

NOPERSONA = '__NO__PERSONA__BEAM__MIN__LEN__20__'
DUMMY_TEXT = '__SILENCE__'
EXCLUDED_QUESTIONS = set(open('excluded_questions.txt').read().splitlines())

from nltk import sent_tokenize, word_tokenize
def is_valid_question(sent):
    words = word_tokenize(sent)
    words = [w.lower() for w in words if '?' not in w]
    sent  = ' '.join(words)

    if len(words) <= 2:
        return False

    if any([c for c in words if c in ['it','there','she','her','he','him','his',
                                      'they','them','yours','yourself','that','thier']]):
        return False

    block_phrases = ['how are you','how about you','what about you','what are you up to',
                     'how are you doing', 'how was','how is','how are','how has',
                     'any recommendations','any recommendation','any suggestions',
                     'any suggestion', 'up to', 'any ideas', 'any advice', 'any tips']

    for phrase in block_phrases:
        if phrase in sent:
            return False

    if sent in EXCLUDED_QUESTIONS:
        return False

    if ('how' in sent or 'what' in sent) and ('going' in sent or 'doing' in sent or 'day' in sent):
        return False

    return True

def extract_questions(sent, valid_check = True):
    q_words = ['what', 'which', 'when', 'where', 'who', 'whom', 'whose', 'why', 'whether', 'how',
           'can', 'could', 'do', 'did', 'does', 'had', 'has', 'have', 'may', 'might', 'shall',
           'should', 'will', 'would',  'are', 'is', 'was', 'were', 'won', 'hows', 'whats', 'any']

    new_question = [q for q in sent_tokenize(sent) if "?" in q]
    for i in range(0,len(new_question)):
        words = new_question[i].split()
        words.insert(0,'.')

        for j in range(1,len(words)):
            if ((words[j].lower() in q_words or words[j] == 'You') and 
                (',' in words[j-1] or '.' in words[j-1] or 
                 words[j-1].lower() in ['hey','so','hello','hi'])):
                break

        if j == len(words) - 1:
            new_question[i] = None
            continue

        for z in range(len(words) - 1, -1, -1):
            if '?' in words[z]:
                break

        words = words[j:z+1]
        new_question[i] = ' '.join(words)

    new_question = [q for q in new_question if q != None]
    if valid_check:
        new_question = [q for q in new_question if is_valid_question(q)]
    return new_question

def chunk_memory(part_memories):
    if part_memories == '':
        return part_memories

    part_memories = part_memories.split('\n')
    prefix_idx = part_memories[0].index(':') + 1
    prefix = part_memories[0][:prefix_idx]
    part_memories = [p[prefix_idx+1:] for p in part_memories]
    
    chunks = []
    accum_memory = [prefix]
    for memory in part_memories:
        if (len(accum_memory) == 4 or len(' '.join(accum_memory).split()) > 30 or
            (len(accum_memory) == 3 and len(memory.split()) > 30)
           ):
            chunks.append(' '.join(accum_memory))
            accum_memory = [prefix, memory]
        else:
            accum_memory.append(memory)
    if len(accum_memory) != 1:
        chunks.append(' '.join(accum_memory))

    return '\n'.join(chunks)

def get_sessionbase_dir_path(opt, dpath, task_name):
    assert task_name in ['msc_personasummary', 'msc_dialogue']
    dpath = os.path.join(dpath, 'msc', task_name, f'session_{opt.get("session_id", 0)}')
    return dpath


def get_predicted_summary_path(dpath, is_session_level=True):
    if is_session_level:
        return os.path.join(
            dpath, 'msc', 'msc_dialogue', 'sessionlevel_summaries_subsample5.json'
        )
    else:
        return os.path.join(dpath, 'msc', 'msc_dialogue', 'summaries_subsample5.json')


class SessionBasePersonaSummaryTeacher(DialogTeacher):
    """
    Teacher that summarizes the persona lines.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('MSC Persona Summary Teacher options')
        agent.add_argument('--session-id', type=int, default=1, help="session id")
        agent.add_argument(
            '--summary-num-turns',
            type=int,
            default=-1,
            help="number of turns to infer persona",
        )
        agent.add_argument(
            '--nopersona-subsampling-weight',
            type=float,
            default=1,
            help="subampling ratio ",
        )
        return parser

    def __init__(self, opt, shared=None):
        self.summary_num_turns = opt['summary_num_turns']
        assert (
            self.summary_num_turns < 0 or self.summary_num_turns % 2 == 0
        ), "Please choose an even number for turns"
        self.session_id = opt['session_id']
        assert opt['session_id'] <= 4, f"No data beyong session {opt['session_id']}!"
        assert (
            opt['session_id'] <= 3 or 'train' not in opt['datatype']
        ), f"No train data beyong session {opt['session_id']}!"
        self.nopersona_subsampling_weight = opt['nopersona_subsampling_weight']
        if 'test' in opt['datatype']:
            logger.warning(f'WARNING: Do not subsampling for {opt["datatype"]}')
            self.nopersona_subsampling_weight = 1
        assert (
            self.nopersona_subsampling_weight >= 0
            and self.nopersona_subsampling_weight <= 1
        ), "invalid subsampling weight"

        dpath = build(opt)
        opt['datafile'] = get_sessionbase_dir_path(opt, dpath, 'msc_personasummary')
        self.id = f'msc_personasummary_{self.session_id}'
        super().__init__(opt, shared)

    def setup_data(self, data_path):
        print('loading: ' + data_path)
        if self.datatype.startswith('train'):
            path_to_open = os.path.join(data_path, 'train.txt')
        elif self.datatype.startswith('valid'):
            path_to_open = os.path.join(data_path, 'valid.txt')
        else:
            path_to_open = os.path.join(data_path, 'test.txt')

        with PathManager.open(path_to_open) as f:
            raw_data = [json.loads(line.strip()) for line in f]

        data = []
        negative_data = []
        for dialog_dict in raw_data:
            current_episode = dialog_dict['dialog']
            init_personachat = dialog_dict['init_personachat']
            for end_idx in range(len(current_episode)):
                if self.summary_num_turns > 0:
                    start_index = max(0, end_idx - self.summary_num_turns + 1)
                else:
                    start_index = 0
                end_line_persona = (
                    current_episode[end_idx]['persona_text']
                    if 'persona_text' in current_episode[end_idx]
                    else NOPERSONA
                )
                dialog_texts = [
                    current_episode[i]['text'] for i in range(start_index, end_idx + 1)
                ]

                action = {
                    'id': self.id,
                    'text': '\n'.join(dialog_texts),
                    'labels': [end_line_persona],
                    'initial_data_id': dialog_dict['initial_data_id'],
                    'init_personas': init_personachat['init_personas'],
                    'utt_idx': end_idx,
                    'speaker_idx': end_idx % 2 + 1,
                    'session_id': self.session_id,
                }
                if end_line_persona == NOPERSONA:
                    negative_data.append(action)
                else:
                    data.append(action)

        size_to_sample = math.ceil(
            self.nopersona_subsampling_weight * len(negative_data)
        )
        data.extend(random.sample(negative_data, size_to_sample))
        random.shuffle(data)

        for episode in data:
            yield Message(episode), True


class SessionBaseMscTeacher(DialogTeacher):
    """
    Teacher that generate text in the multi-session chat.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Multi-Session Chat Task options')
        agent.add_argument(
            '--session-id',
            type=int,
            default=2,
            help="session id, session_id = 1 refers to convai2 teacher and it's not supported here",
        )
        agent.add_argument(
            '--previous-persona-type',
            type=str,
            default="raw_history",
            choices=[
                'none',
                'goldsum_self',
                'goldsum_both',
                'goldsum_their',
                'predsum_self',
                'predsum_both',
                'predsum_their',
                'predsum_utt_self',
                'predsum_utt_both',
                'predsum_utt_their',
                'init_self',
                'init_both',
                'init_their',
                'raw_history',
            ],
            help="type of previous context to include as context. "
            "the 'goldsum_' prefix refers to gold persona summaries from crowdworkers; "
            "the 'predsum_' prefix refers to predicted persona summaries from a summarization model; "
            "the 'init_' prefix refers to the original persona lines used to ground the PersonaChat conversations. ",
        )
        agent.add_argument(
            '--your-persona-first',
            type=bool,
            default=False,
            help="whether to prepend your persona first or not",
        )
        agent.add_argument(
            '--session-openning',
            type=bool,
            default=False,
            help="whether to only include session opening or not",
        )
        agent.add_argument(
            '--label-speaker-id',
            type=str,
            default="both",
            choices=['self', 'both', 'their'],
            help="the speaker id of the 'labels' field,",
        )
        agent.add_argument(
            '--include-time-gap',
            type=bool,
            default=False,
            help="whether to include time passed since last conversation in the context",
        )
        agent.add_argument(
            '--history-time-gaps-token',
            type=str,
            default=None,
            help="time tokens in the previous raw dialogue history, e.g. 'time:' ",
        )
        agent.add_argument(
            '--history-person-tokens',
            type=str,
            default=None,
            help="person tokens in the previous raw dialogue history, e.g. 'p1:,p2:' ",
        )
        agent.add_argument(
            '--previous-session-delimiter',
            type=str,
            default=None,
            help="delimiter between previous sessions in the context, such as '__NEXT_SESSION__' ",
        )
        return parser

    def __init__(self, opt, shared=None):
        assert opt['session_id'] <= 5, f"No data beyong session {opt['session_id']}!"
        assert (
            opt['session_id'] <= 4 or 'train' not in opt['datatype']
        ), f"No train data beyong session {opt['session_id']}!"
        assert (
            not opt['previous_persona_type'].startswith('predsum')
            or opt['session_id'] <= 4
            or (
                opt['session_id'] == 5
                and ('valid' in opt['datatype'] or 'test' in opt['datatype'])
            )
        ), f"No predicted summary for session {opt['session_id']}"
        self.previous_persona_type = opt['previous_persona_type']
        self.session_openning = opt.get('session_openning', False)
        if self.session_openning:
            opt['label_speaker_id'] = 'their'
        # NOTE: session_id = 1: personachat
        self.session_id = opt['session_id']
        self.label_speaker_id = opt["label_speaker_id"]
        self.your_persona_first = opt['your_persona_first']
        self.include_last_time_gap = opt['include_time_gap']
        self.history_time_gaps_token = opt['history_time_gaps_token']
        if self.history_time_gaps_token:
            self.include_last_time_gap = False
        self.history_person_tokens = opt['history_person_tokens']
        self.use_predicted_summary = self.previous_persona_type.startswith('predsum')
        self.previous_session_delimiter = opt.get('previous_session_delimiter', None)
        if self.history_person_tokens is not None:
            self.history_person_tokens = self.history_person_tokens.split(",")
        self.msc_dpath = build(opt)
        opt['datafile'] = get_sessionbase_dir_path(opt, self.msc_dpath, 'msc_dialogue')

        self.id = f'msc_dialogue_{self.session_id}'
        super().__init__(opt, shared)

    def normalize_replies(self, x):
        xs = [xt.strip() for xt in x.split('\n')]
        xs2 = []
        for x in xs:
            if 'your persona:' in x:
                # Normalize the sentence appearing after 'your persona:'
                x = x[len('your persona: ') :]
                x = normalize_reply(x)
                x = 'your persona: ' + x
            elif "partner's persona: " in x:
                x = x[len("partner's persona: ") :]
                x = normalize_reply(x)
                x = "partner's persona: " + x
            elif x != DUMMY_TEXT:
                x = normalize_reply(x)
            xs2.append(x)
        return "\n".join(xs2)

    def setup_data2(self, datafile):
        print('loading: ' + datafile)
        if self.datatype.startswith('train'):
            path_to_open = os.path.join(datafile, 'train.txt')
        elif self.datatype.startswith('valid'):
            path_to_open = os.path.join(datafile, 'valid.txt')
        else:
            path_to_open = os.path.join(datafile, 'test.txt')

        with PathManager.open(path_to_open) as f:
            raw_data = [json.loads(line.strip()) for line in f]

        data = []
        label_speaker_id_range = {}
        predicted_summary_dict = {}
        if self.use_predicted_summary:
            is_session_level = not ('utt_' in self.previous_persona_type)
            predsum_path = get_predicted_summary_path(self.msc_dpath, is_session_level)
            logger.warning(f"use the predicted summary from {predsum_path}")
            with PathManager.open(predsum_path) as jsonfile:
                predicted_summary_dict = json.load(jsonfile)

        def _get_time_gap(time_num, time_unit, time_token=""):
            time_gap = str(time_num) + ' ' + time_unit
            return f'{time_token} {time_gap}' if len(time_token) > 0 else time_gap

        def _compile_persona_dialog_input(
            dialog, personas, previous_dialogs, label_speaker_id
        ):
            new_dialog = copy.deepcopy(dialog)
            new_previous_dialogs = copy.deepcopy(previous_dialogs)
            your_persona = ""
            partner_persona = ""
            if label_speaker_id == 'self':
                your_persona = '\n'.join([f'your persona: {x}' for x in personas[1]])
                partner_persona = '\n'.join(
                    [f"partner's persona: {x}" for x in personas[0]]
                )
            elif label_speaker_id == 'their':
                your_persona = '\n'.join([f'your persona: {x}' for x in personas[0]])
                partner_persona = '\n'.join(
                    [f"partner's persona: {x}" for x in personas[1]]
                )
                for prev_dialog in new_previous_dialogs:
                    prev_dialog['dialog'].insert(0, {"text": DUMMY_TEXT})
                    if len(prev_dialog['dialog']) % 2 == 1 and (
                        self.history_person_tokens is None
                    ):
                        prev_dialog['dialog'].append({"text": DUMMY_TEXT})
                new_dialog.insert(0, {"text": DUMMY_TEXT})

            return your_persona, partner_persona, new_dialog, new_previous_dialogs

        for dialog_dict in raw_data:
            initial_data_id = dialog_dict['metadata']['initial_data_id']
            if self.label_speaker_id == 'both':
                label_speaker_id_range = ['their', 'self']
            else:
                label_speaker_id_range = [self.label_speaker_id]

            for label_speaker_id in label_speaker_id_range:
                if self.use_predicted_summary:
                    personas_to_complie = predicted_summary_dict[
                        str(self.session_id - 1)
                    ][initial_data_id]
                elif self.previous_persona_type.startswith('init'):
                    personas_to_complie = dialog_dict['init_personas']
                else:
                    personas_to_complie = dialog_dict['personas']

                (
                    your_persona,
                    partner_persona,
                    new_dialog,
                    new_previous_dialogs,
                ) = _compile_persona_dialog_input(
                    dialog_dict['dialog'],
                    personas_to_complie,
                    dialog_dict['previous_dialogs'],
                    label_speaker_id,
                )
                previous_sessions_msgs = []
                if self.previous_persona_type == 'raw_history':
                    for d_id in range(len(new_previous_dialogs)):
                        previous_dialog_msg = [
                            x['text'] for x in new_previous_dialogs[d_id]['dialog']
                        ]
                        if self.history_person_tokens:
                            previous_dialog_msg = [
                                self.history_person_tokens[i % 2] + ' ' + text
                                for i, text in enumerate(previous_dialog_msg)
                                if text != DUMMY_TEXT
                            ]
                        if self.history_time_gaps_token:
                            time_gap_i = _get_time_gap(
                                new_previous_dialogs[d_id]['time_num'],
                                new_previous_dialogs[d_id]['time_unit'],
                                time_token=self.history_time_gaps_token,
                            )
                            previous_sessions_msgs.append(
                                '\n'.join(previous_dialog_msg + [time_gap_i])
                            )
                        else:
                            previous_sessions_msgs.append(
                                '\n'.join(previous_dialog_msg)
                            )

                if self.previous_session_delimiter is not None:
                    previous_sessions_msgs = [
                        val
                        for pair in zip(
                            previous_sessions_msgs,
                            [self.previous_session_delimiter]
                            * len(previous_sessions_msgs),
                        )
                        for val in pair
                    ]
                previous_sessions_msgs = '\n'.join(previous_sessions_msgs)

                episode = []
                for i in range(0, len(new_dialog) - 1, 2):
                    text = new_dialog[i]['text']
                    partner_persona_one_line = partner_persona.replace('\n', '').split(
                        "partner's persona: "
                    )
                    your_persona_one_line = your_persona.replace('\n', '').split(
                        "your persona: "
                    )
                    action = {
                        'id': self.id,
                        'text': self.normalize_replies(text),
                        'labels': [self.normalize_replies(new_dialog[i + 1]['text'])],
                        'session_id': self.session_id,
                        'initial_data_id': initial_data_id,
                        'personas': f'{partner_persona}\n{your_persona}',
                        'personas_one_line': f"partner's persona: {' '.join(partner_persona_one_line)}\nyour persona: {' '.join(your_persona_one_line)}",
                    }
                    if i == 0:
                        action.update(
                            {
                                'time_num': dialog_dict['previous_dialogs'][-1][
                                    'time_num'
                                ],
                                'time_unit': dialog_dict['previous_dialogs'][-1][
                                    'time_unit'
                                ],
                            }
                        )

                    episode.append(action)
                    if self.session_openning:
                        break

                persona_context_str = ""
                if 'self' in self.previous_persona_type:
                    persona_context_str = your_persona
                elif 'their' in self.previous_persona_type:
                    persona_context_str = partner_persona
                elif 'both' in self.previous_persona_type:
                    if self.your_persona_first:
                        persona_context_str = (
                            (your_persona + '\n') if len(your_persona) > 0 else ""
                        ) + partner_persona
                    else:
                        persona_context_str = (
                            (partner_persona + '\n') if len(partner_persona) > 0 else ""
                        ) + your_persona
                elif self.previous_persona_type == 'raw_history':
                    persona_context_str = previous_sessions_msgs

                if self.include_last_time_gap:
                    time_gap = _get_time_gap(
                        dialog_dict['previous_dialogs'][-1]['time_num'],
                        dialog_dict['previous_dialogs'][-1]['time_unit'],
                    )
                    persona_context_str = (
                        (persona_context_str + '\n')
                        if len(persona_context_str) > 0
                        else ""
                    ) + f'[{time_gap}]'

                if persona_context_str and len(persona_context_str) > 0:
                    episode[0]['text'] = persona_context_str + '\n' + episode[0]['text']

                data.append(episode)

        for episode in data:
            start_idx = 0
            for i, turn in enumerate(episode):
                yield Message(turn), i == start_idx
    
    def setup_data(self, datafile):
        print('loading: ' + datafile)
        if self.datatype.startswith('train'):
            path_to_open = os.path.join(datafile, 'train.txt')
        elif self.datatype.startswith('valid'):
            path_to_open = os.path.join(datafile, 'valid.txt')
        else:
            path_to_open = os.path.join(datafile, 'test.txt')

        with PathManager.open(path_to_open) as f:
            raw_data = [json.loads(line.strip()) for line in f]

        data = []
        label_speaker_id_range = {}
        predicted_summary_dict = {}
        if self.use_predicted_summary:
            is_session_level = not ('utt_' in self.previous_persona_type)
            predsum_path = get_predicted_summary_path(self.msc_dpath, is_session_level)
            logger.warning(f"use the predicted summary from {predsum_path}")
            with PathManager.open(predsum_path) as jsonfile:
                predicted_summary_dict = json.load(jsonfile)

        def _get_time_gap(time_num, time_unit, time_token=""):
            time_gap = str(time_num) + ' ' + time_unit
            return f'{time_token} {time_gap}' if len(time_token) > 0 else time_gap

        def _compile_persona_dialog_input(
            dialog, personas, previous_dialogs, label_speaker_id
        ):
            new_dialog = copy.deepcopy(dialog)
            new_previous_dialogs = copy.deepcopy(previous_dialogs)
            your_persona = ""
            partner_persona = ""
            if label_speaker_id == 'self':
                your_persona = '\n'.join([f'your persona: {x}' for x in personas[1]])
                partner_persona = '\n'.join(
                    [f"partner's persona: {x}" for x in personas[0]]
                )
            elif label_speaker_id == 'their':
                your_persona = '\n'.join([f'your persona: {x}' for x in personas[0]])
                partner_persona = '\n'.join(
                    [f"partner's persona: {x}" for x in personas[1]]
                )
                for prev_dialog in new_previous_dialogs:
                    prev_dialog['dialog'].insert(0, {"text": DUMMY_TEXT})
                    if len(prev_dialog['dialog']) % 2 == 1 and (
                        self.history_person_tokens is None
                    ):
                        prev_dialog['dialog'].append({"text": DUMMY_TEXT})
                new_dialog.insert(0, {"text": DUMMY_TEXT})

            return your_persona, partner_persona, new_dialog, new_previous_dialogs
                
        for dialog_dict in raw_data:
            initial_data_id = dialog_dict['metadata']['initial_data_id']
            if self.label_speaker_id == 'both':
                label_speaker_id_range = ['their', 'self']
            else:
                label_speaker_id_range = [self.label_speaker_id]

            for label_speaker_id in label_speaker_id_range:
                if self.use_predicted_summary:
                    personas_to_complie = predicted_summary_dict[
                        str(self.session_id - 1)
                    ][initial_data_id]
                elif self.previous_persona_type.startswith('init'):
                    personas_to_complie = dialog_dict['init_personas']
                else:
                    personas_to_complie = dialog_dict['personas']

                (
                    your_persona,
                    partner_persona,
                    new_dialog,
                    new_previous_dialogs,
                ) = _compile_persona_dialog_input(
                    dialog_dict['dialog'],
                    personas_to_complie,
                    dialog_dict['previous_dialogs'],
                    label_speaker_id,
                )
                previous_sessions_msgs = []
                if self.previous_persona_type == 'raw_history':
                    for d_id in range(len(new_previous_dialogs)):
                        previous_dialog_msg = [
                            x['text'] for x in new_previous_dialogs[d_id]['dialog']
                        ]
                        if self.history_person_tokens:
                            previous_dialog_msg = [
                                self.history_person_tokens[i % 2] + ' ' + text
                                for i, text in enumerate(previous_dialog_msg)
                                if text != DUMMY_TEXT
                            ]
                        if self.history_time_gaps_token:
                            time_gap_i = _get_time_gap(
                                new_previous_dialogs[d_id]['time_num'],
                                new_previous_dialogs[d_id]['time_unit'],
                                time_token=self.history_time_gaps_token,
                            )
                            previous_sessions_msgs.append(
                                '\n'.join(previous_dialog_msg + [time_gap_i])
                            )
                        else:
                            previous_sessions_msgs.append(
                                '\n'.join(previous_dialog_msg)
                            )

                if self.previous_session_delimiter is not None:
                    previous_sessions_msgs = [
                        val
                        for pair in zip(
                            previous_sessions_msgs,
                            [self.previous_session_delimiter]
                            * len(previous_sessions_msgs),
                        )
                        for val in pair
                    ]
                previous_sessions_msgs = '\n'.join(previous_sessions_msgs)

                ## Changes
                partner_persona = chunk_memory(partner_persona)
                your_persona = chunk_memory(your_persona)
                ###
                
                episode = []
                for i in range(0, len(new_dialog) - 1, 2):
                    text = '\n'.join([t['text'] for t in new_dialog[:i+1]])
                    partner_persona_one_line = partner_persona.replace('\n', '').split(
                        "partner's persona: "
                    )
                    your_persona_one_line = your_persona.replace('\n', '').split(
                        "your persona: "
                    )
                    action = {
                        'id': self.id,
                        'text': self.normalize_replies(text),
                        'labels': [self.normalize_replies(new_dialog[i + 1]['text'])],
                        'session_id': self.session_id,
                        'initial_data_id': initial_data_id,
                        'personas': f'{partner_persona}\n{your_persona}',
                        'personas_one_line': f"partner's persona: {' '.join(partner_persona_one_line)}\nyour persona: {' '.join(your_persona_one_line)}",
                    }
                    if i == 0:
                        action.update(
                            {
                                'time_num': dialog_dict['previous_dialogs'][-1][
                                    'time_num'
                                ],
                                'time_unit': dialog_dict['previous_dialogs'][-1][
                                    'time_unit'
                                ],
                            }
                        )

                    episode.append(action)
                    if self.session_openning:
                        break

                persona_context_str = ""
                if 'self' in self.previous_persona_type:
                    persona_context_str = your_persona
                elif 'their' in self.previous_persona_type:
                    persona_context_str = partner_persona
                elif 'both' in self.previous_persona_type:
                    if self.your_persona_first:
                        persona_context_str = (
                            (your_persona + '\n') if len(your_persona) > 0 else ""
                        ) + partner_persona
                    else:
                        persona_context_str = (
                            (partner_persona + '\n') if len(partner_persona) > 0 else ""
                        ) + your_persona
                elif self.previous_persona_type == 'raw_history':
                    persona_context_str = previous_sessions_msgs

                if self.include_last_time_gap:
                    time_gap = _get_time_gap(
                        dialog_dict['previous_dialogs'][-1]['time_num'],
                        dialog_dict['previous_dialogs'][-1]['time_unit'],
                    )
                    persona_context_str = (
                        (persona_context_str + '\n')
                        if len(persona_context_str) > 0
                        else ""
                    ) + f'[{time_gap}]'

                if persona_context_str and len(persona_context_str) > 0:
                    for z in range(0,len(episode)):
                        episode[z]['text'] = persona_context_str + '\n' + episode[z]['text']
                data.append(episode)
        
        all_samples = [j for i in data for j in i]
        
        if 'session_2' in path_to_open:
            random.seed(15)
        else:
            random.seed(44)
        random.shuffle(all_samples)
        
        print(len(all_samples))
        
        for sample in all_samples:
            yield sample, True

def random_remove(personas):
    num_remove = 0
    if len(personas) <= 1:
        return personas
    if len(personas) <= 4:
        num_remove = 1
    else:
        num_remove = random.randint(1,3)
    new_personas = personas.copy()
    indices = random.sample(range(len(new_personas)), len(new_personas) - num_remove)
    return [new_personas[i] for i in sorted(indices)]

def random_insert(personas, negative_persona):
    insert_id = random.choice(range(0,len(personas) + 1))    
    new_personas = personas.copy()
    new_personas.insert(insert_id, negative_persona)
    return new_personas

def flatten_persona(partner_persona):
    partner_persona = [sent_tokenize(p) for p in partner_persona]
    partner_persona = [z for p in partner_persona for z in p]
    partner_persona = ["partner's persona: " + p if "partner's persona: " not in p else p for p in partner_persona]
    return partner_persona

class NegativeMscTeacher(DialogTeacher):
    """
    Teacher that generate text in the multi-session chat.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Multi-Session Chat Task options')
        agent.add_argument(
            '--session-id',
            type=int,
            default=6,
            help="session id, session_id = 1 refers to convai2 teacher and it's not supported here",
        )
        return parser

    def __init__(self, opt, shared=None):
        self.session_id = opt['session_id']
        self.msc_path = build(opt)
        opt['datafile'] = get_sessionbase_dir_path(opt, self.msc_path, 'msc_dialogue')
        self.id = f'msc_dialogue_{self.session_id}'
        super().__init__(opt, shared)

    def normalize_replies(self, x):
        xs = [xt.strip() for xt in x.split('\n')]
        xs2 = []
        for x in xs:
            if 'your persona:' in x:
                # Normalize the sentence appearing after 'your persona:'
                x = x[len('your persona: ') :]
                x = normalize_reply(x)
                x = 'your persona: ' + x
            elif "partner's persona: " in x:
                x = x[len("partner's persona: ") :]
                x = normalize_reply(x)
                x = "partner's persona: " + x
            elif x != DUMMY_TEXT:
                x = normalize_reply(x)
            xs2.append(x)
        return "\n".join(xs2)

    def setup_data(self, datafile):
        print('loading: ' + datafile)
        if self.datatype.startswith('train'):
            path_to_open = os.path.join(datafile, 'train.txt')
        elif self.datatype.startswith('valid'):
            path_to_open = os.path.join(datafile, 'valid.txt')
        else:
            path_to_open = os.path.join(datafile, 'test.txt')

        with PathManager.open(path_to_open) as f:
            raw_data = [json.loads(line.strip()) for line in f]
        
        data = []
        
        for dialog_dict in raw_data:
            partner_persona = '\n'.join(["partner's persona: " + d for d in dialog_dict['part_persona']])
            your_persona    = '\n'.join(["your persona: " + d for d in dialog_dict['self_persona']])
            partner_persona_one_line = partner_persona.replace('\n', '').split("partner's persona: ")
            your_persona_one_line = your_persona.replace('\n', '').split("your persona: ")
            
            partner_persona = partner_persona.split('\n')
            random.shuffle(partner_persona)
            partner_persona = chunk_memory('\n'.join(partner_persona))
            your_persona = chunk_memory(your_persona)
            dialog_text =  '\n'.join(dialog_dict['text'].split('\n'))
            
            action = {
                'id': self.id,
                'text': self.normalize_replies(dialog_text),
                'labels': [self.normalize_replies(dialog_dict['labels'])],
                'session_id': self.session_id,
                'initial_data_id': dialog_dict['initial_data_id'],
                'personas': f'{partner_persona}\n{your_persona}',
                'personas_one_line': f"partner's persona: {' '.join(partner_persona_one_line)}\nyour persona: {' '.join(your_persona_one_line)}",
                }
            
            if 'negative_question' in dialog_dict:
                action.update({'negative_questions': [dialog_dict['negative_question']]})
                partner_persona  = ["partner's persona: " + d for d in dialog_dict['part_persona']]
                negative_persona = "partner's persona: " + dialog_dict['part_persona_negative'][0]
                
                if 'para_negative_question' in dialog_dict:
                    para_question = random.choice(dialog_dict['para_negative_question'])
                    test = action['labels'][0].index(dialog_dict['negative_question']['question'])
                    para_label = action['labels'][0].replace(dialog_dict['negative_question']['question'], para_question)
                    action.update({'para_label': para_label})
                    action.update({'para_negative_question': para_question})
                    negative_para = "partner's persona: " + random.choice(dialog_dict['para_negative_persona'])
                    
                    alter_partner_personas = dialog_dict['hard_negatives'][:4] + dialog_dict['hard_positives'][:4]
                    alter_partner_personas = ["partner's persona: " + p for p in alter_partner_personas]
                    alter_partner_personas = [negative_persona] + alter_partner_personas
                    alter_partner_personas = [random_insert(partner_persona,p) for p in alter_partner_personas]
                    
                    for i in range(0,9):
                        partner_persona = alter_partner_personas[i]
                        partner_persona = flatten_persona(partner_persona)
                        random.shuffle(partner_persona)
                        partner_persona = chunk_memory('\n'.join(partner_persona))
                        if i == 0:
                            action.update({'negative_personas': f'{partner_persona}\n{your_persona}'})
                        else:
                            action.update({'negative_personas' + str(i): f'{partner_persona}\n{your_persona}'})
                else:
                    partner_persona = random_insert(partner_persona,negative_persona)
                    partner_persona = flatten_persona(partner_persona)
                    random.shuffle(partner_persona)
                    partner_persona = chunk_memory('\n'.join(partner_persona))
                    action.update({'negative_personas': f'{partner_persona}\n{your_persona}'})
            
            data.append(action)
        
        random.seed(99)
        if 'train' in datafile:
            random.shuffle(data)
        
        print(len(data))
        
        for i, turn in enumerate(data):
            yield Message(turn), i == i

class PersonaSummaryTeacher(MultiTaskTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        parser = parser.add_argument_group('MSC Summary Teacher Args')
        parser.add_argument(
            '--include-last-session',
            type=bool,
            default=False,
            help="whether to include session 4 for valid and test splits",
        )
        SessionBasePersonaSummaryTeacher.add_cmdline_args(parser, partial_opt)
        return parser

    def __init__(self, opt, shared=None):
        msc_tasks = [
            'msc:SessionBasePersonaSummary:session_id=1',
            'msc:SessionBasePersonaSummary:session_id=2',
            'msc:SessionBasePersonaSummary:session_id=3',
        ]
        if opt.get('include_last_session', False) and 'train' not in opt['datatype']:
            msc_tasks += ['msc:SessionBasePersonaSummary:session_id=4']
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join(msc_tasks)
        super().__init__(opt, shared)


class Session1NormalizedTrait(NormalizedTeacherTrait):
    """
    Trait for flatten persona into one line.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Session Level NormalizedTeacher arguments')
        agent.add_argument(
            '--is-convai2-session-level',
            type=bool,
            default=False,
            help="whether to flatten the persona lines into a single persona line per speaker",
        )
        return agent

    def __init__(self, opt, shared=None):
        self.is_convai2_session_level = opt.get('is_convai2_session_level', False)
        super().__init__(opt, shared)

    def normalize_replies(self, x):
        xs = x.split('\n')
        your_personas = []
        partner_personas = []
        non_personas = []
        for x in xs:
            if x.startswith('your persona: '):
                # Normalize the sentence appearing after 'your persona:'
                x = x[len('your persona: ') :]
                x = normalize_reply(x)
                your_personas.append(x)
            elif x.startswith("partner's persona: "):
                x = x[len("partner's persona: ") :]
                x = normalize_reply(x)
                partner_personas.append(x)
            else:
                x = normalize_reply(x)
                non_personas.append(x)
        xs2 = []
        if not self.is_convai2_session_level:
            your_personas = ['your persona: ' + yx for yx in your_personas]
            partner_personas = ["partner's persona: " + px for px in partner_personas]
        else:
            if your_personas:
                your_personas = ['your persona: ' + " ".join(your_personas)]
            if partner_personas:
                partner_personas = ["partner's persona: " + " ".join(partner_personas)]
        if self.your_persona_first:
            xs2.extend(your_personas)
            xs2.extend(partner_personas)
        else:
            xs2.extend(partner_personas)
            xs2.extend(your_personas)
        xs2.extend(non_personas)
        return '\n'.join(xs2)

class Session1SelfTeacher(Session1NormalizedTrait, SelfOriginalTeacher):
    """
    Convai2 as Session 1.
    """
    pass

class MscTeacher(MultiTaskTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        parser = parser.add_argument_group('Multi Session Chat (MSC) Teacher Args')
        parser.add_argument(
            '--include-session1',
            type=bool,
            default=True,
            help="whether to include session 1 (convai2:normalized)",
        )
        parser.add_argument(
            '--include-last-session',
            type=bool,
            default=False,
            help="whether to include session 5",
        )
        SessionBaseMscTeacher.add_cmdline_args(parser, partial_opt)
        NegativeMscTeacher.add_cmdline_args(parser, partial_opt)
        Session1SelfTeacher.add_cmdline_args(parser, partial_opt)
        return parser

    def __init__(self, opt, shared=None):
        msc_tasks = [
            'msc:Session1Self:is_convai2_session_level=False',
            'msc:SessionBaseMsc:session_id=2',
            'msc:SessionBaseMsc:session_id=3',
            'msc:SessionBaseMsc:session_id=4',
            'msc:NegativeMsc:session_id=6',
        ]
        if opt.get('include_session1', False) and not opt['session_openning']:
            if opt['previous_persona_type'] in [
                'predsum_self',
                'predsum_both',
                'predsum_their',
            ]:
                msc_tasks = [
                    'msc:Session1Self:is_convai2_session_level=True'
                ] + msc_tasks
            else:
                msc_tasks = [
                    'msc:Session1Self:is_convai2_session_level=False'
                ] + msc_tasks
        if opt.get('include_last_session', False) and 'train' not in opt['datatype']:
            msc_tasks += ['msc:SessionBaseMsc:session_id=5']
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join(msc_tasks)
        super().__init__(opt, shared)

class DefaultTeacher(MscTeacher):
    pass

class ContextGenerator(BaseContextGenerator):
    """
    Generates contexts shown to bots for generating prompt when collecting human-human
    followup chat in the personal knowledge human evaluation.

    This generator was used to generate the context information shown to bots at the
    beginning of a conversation, when crowdsourcing the conversations that for per-turn
    human evaluation.
    """

    def __init__(self, override_opt, datatype='valid', seed: Optional[int] = None):
        """
        Initalize the context generator.

        override_opt: only a 'datapath' key is required, to specify the ParlAI data folder
        """

        def setup_opt(opt):
            parser = dsd.setup_args()
            parser.set_params(**opt)
            return parser.parse_args([])

        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()

        with open(override_opt['completed_run_stats']) as f:
            override_opt.update(json.load(f))

        bot_model_name = override_opt['bot_model_name']
        bot_msc_opt = copy.deepcopy(COMMON_CONFIG)
        bot_msc_opt.update(MODEL_OPT[bot_model_name])
        ui_msc_opt = copy.deepcopy(COMMON_CONFIG)
        ui_msc_opt.update(UI_OPT[bot_model_name])

        self.ui_msc_teacher = SessionBaseMscTeacher(setup_opt(ui_msc_opt))
        self.bot_msc_teacher = SessionBaseMscTeacher(setup_opt(bot_msc_opt))
        self.bot_sorted_initial_data_indices_to_episode = {}
        self.ui_sorted_initial_data_indices_to_episode = {}
        self.initial_data_indices_to_complete = override_opt.get(
            'initial_data_indices_to_complete', INITIAL_DATA_TO_COMPLETE
        )
        self._set_teacher_data_map()
        self.context_done_statistics = copy.deepcopy(
            override_opt.get('context_done_statistics', {})
        )

    def get_context(self, model_name: str = None) -> dict:
        """
        Get context information to be shown at the beginning of one conversation.

        Values in return dict:
        - context_dataset: the dataset ('msc') used to generate the context information.
        - your_persona_strings: persona strings for the "self" side
        - their_persona_strings: persona strings for the "partner" side
        - context_for_bot_prompt: text of dialogue context shown to the bot to generate the session opennings
        - observation_for_bot: observation containing dialogue context shown to the bot to generate the session opennings
        - time_num: number of hours/days that have transpired since last chat session
        - time_unit: unit(hours/days) of the time that have transpired since last chat session
        """

        # Determine which dataset we will show context for
        if model_name not in self.context_done_statistics:
            self.context_done_statistics[model_name] = []
        initial_data_indices_list = [
            x
            for x in self.initial_data_indices_to_complete
            if x not in self.context_done_statistics[model_name]
        ]
        if len(initial_data_indices_list) == 0:
            return None
        # Select episode
        initial_data_index = self.rng.sample(initial_data_indices_list, 1)[0]
        # Mark context seletected
        self.context_done_statistics[model_name].append(initial_data_index)
        # Extract personas
        return self._extract_personas(initial_data_index)

    def _set_teacher_data_map(self):
        self.ui_sorted_initial_data_indices_to_episode = {
            episode[0]['initial_data_id']: episode
            for episode in self.ui_msc_teacher.data.data
        }
        self.bot_sorted_initial_data_indices_to_episode = {
            episode[0]['initial_data_id']: episode
            for episode in self.bot_msc_teacher.data.data
        }

    def _extract_personas(self, initial_data_index: str) -> dict:
        """
        For the given ConvAI2 conversation, return strings of both speakers' personas.
        """
        ui_first_entry = self.ui_sorted_initial_data_indices_to_episode[
            initial_data_index
        ][0]
        bot_first_entry = self.bot_sorted_initial_data_indices_to_episode[
            initial_data_index
        ][0]

        ui_context = ui_first_entry['text'].split('\n')
        your_persona_strings = []
        their_persona_strings = []
        for str_ in ui_context[:-1]:  # The last string is the first utterance
            if str_.startswith('your persona: '):  # Here, "you" are Person 2
                your_persona_strings.append(str_[len('your persona: ') :])
            elif str_.startswith("partner's persona: "):
                their_persona_strings.append(str_[len("partner's persona: ") :])
        return {
            'context_dataset': bot_first_entry['id'],
            'your_persona_strings': your_persona_strings,
            'their_persona_strings': their_persona_strings,
            'context_for_bot_prompt': bot_first_entry['text'],
            'observation_for_bot': bot_first_entry,
            'time_num': bot_first_entry['time_num'],
            'time_unit': bot_first_entry['time_unit'],
        }
