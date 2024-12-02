import warnings
warnings.filterwarnings("ignore")
import os
import torch
import numpy
from utils import extract_questions
import pickle
import sys
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer, util
sys.path.append('./Redundant_Classifier/')
from redudant_classifier import Redundant_Classifier
from tqdm import tqdm
import random
from collections import Counter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--log", default=None, required=False, type=str)
parser.add_argument("--start", default=0, required=False, type=int)
parser.add_argument("--end", default=50, required=False, type=int)
parser.add_argument('--baseline', action='store_true')
args = parser.parse_args()

os.environ['TRANSFORMERS_CACHE'] = '/home/people/20202939/scratch/trans_cache/'
model = CrossEncoder('cross-encoder/quora-roberta-large')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.model = model.model.to(device)

redudant_model = Redundant_Classifier('roberta-large', './Redundant_Classifier/ckpt')
persona_map = pickle.load(open('self_chat_conv/baseline/personas.pkl','rb'))
sim_model = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')

base_convs = []
dialog_seeds = []
redundant_questions = []
all_questions = []
all_cos_scores = []

for i in range(args.start,args.end):
    base_convs.append('self_chat_conv/baseline/bot_' + str(i) + '.txt')
    dialog_seeds.append(open('dialog_seeds/' + str(i) + '.txt').read().splitlines())
base_convs = [open(p).read().splitlines() for p in base_convs]

num_response, num_questions, num_repetitive = 0, 0, 0
convs = base_convs

LOG_PATH = args.log

if not args.baseline:
    others = []
    for i in range(0,args.end):
        others.append(open(os.path.join(LOG_PATH, 'bot_' + str(i) + '.txt')).read().splitlines())

for i in tqdm(range(0,args.end)):
    prev_dialog = dialog_seeds[i]
    new_dialog  = convs[i][len(prev_dialog):]
    num_response += len(new_dialog)
    
    #others
    if not args.baseline:
        new_dialog_other = others[i][len(prev_dialog):]
    
    for j in range(0,len(new_dialog)):
        if '?' in new_dialog[j]:
            
            if args.baseline:
                cur_response = new_dialog[j]
            else:
                cur_response = new_dialog_other[j]
            
            if len(extract_questions(cur_response, valid_check = False)) == 0:
                continue
            
            cur_question  = extract_questions(cur_response, valid_check = False)
            recent_dialog = convs[i][len(prev_dialog) + (j-1)] + '\n' + cur_response
            
            if len(cur_question) == 0:
                continue
            all_questions += cur_question
            
            # Calculate coherence score
            for question in cur_question:
                prefix = convs[i][len(prev_dialog) + (j-1)] + ' ' + cur_response[:cur_response.index(question) - 1]
                prefix_emd = sim_model.encode([prefix], convert_to_tensor=True)
                q_emd  = sim_model.encode([question], convert_to_tensor=True)
                all_cos_scores.append(util.cos_sim(prefix_emd, q_emd).tolist()[0][0])
            
            num_questions += 1
            check = False
            
            # Implicit questions
            recent_dialog = recent_dialog[:recent_dialog.rfind('?') + 1]
            recent_dialog = recent_dialog.replace(cur_question[0], '</s></s>' + cur_question[0])
            cur_idx = len(prev_dialog) + j
            prev_personas = list(reversed(persona_map[i][:cur_idx]))
            prev_personas = [k for z in prev_personas[0::2] for k in z]
            
            if check is False and len(prev_personas) > 0 and len(cur_question) > 0:
                preds = redudant_model.predict([recent_dialog] * len(prev_personas), prev_personas)
                for pred in preds:
                    if pred['Redundant'] > 0.85:
                        num_repetitive += 1
                        redundant_questions += cur_question
                        break

all_questions = [q.replace('?','').lower() for q in all_questions]
def generateNgram(paper, ngram = 2, deli = '_', rmSet = {}):
    words = paper.split()
    if len(words) == 1:
        return ''    
    ngrams = []
    for i in range(0,len(words) - ngram + 1):
        block = words[i:i + ngram]
        if not any(w in rmSet for w in block):
            ngrams.append(deli.join(block))
            
    return ngrams

def get_repetition(sentences, ngram):
    ngrams = []
    for sentence in sentences:
        ngrams += generateNgram(sentence, ngram = ngram)
    
    return 1.0 - ( len(set(ngrams)) / len(ngrams) )

def get_diversity(sentences):
    score = 1.0
    for ngram in range(2,5):
        score = score * (1.0 - get_repetition(sentences, ngram))
    return score

print(num_response, num_questions, num_repetitive, num_repetitive / num_questions)
print("Coherence:", sum(all_cos_scores) / len(all_cos_scores))
print("Diversity:", get_diversity(all_questions))
print("Num questions:", len(redundant_questions))
# print(Counter(redundant_questions).most_common(20))
