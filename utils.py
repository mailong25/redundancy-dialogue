EXCLUDED_QUESTIONS  = open('/home/people/20202939/scratch/chatbot/bot_research/questions/exclude_1.txt').read().lower().splitlines()
EXCLUDED_QUESTIONS += open('/home/people/20202939/scratch/chatbot/bot_research/questions/exclude_2.txt').read().splitlines()
# INCLUDED_QUESTIONS = set(open('questions/include_all.txt').read().splitlines())

import os
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
    
def chunk_memory(part_memories, prefix):            
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

    return chunks

def generateNgram(words, ngram = 4, deli = ' '):    
    ngrams = []
    for i in range(0,len(words) - ngram + 1):
        block = words[i:i + ngram]
        ngrams.append(deli.join(block))
    return ngrams

def is_valid_question(sent):
    from nltk import sent_tokenize, word_tokenize
    words = word_tokenize(sent)
    words = [w.lower() for w in words if '?' not in w]
    sent  = ' '.join(words).replace('?','')
    
    if len(words) <= 2:
        return False
    
    if any([c for c in words if c in ['it','there','she','her','he','him','his',
                                      'they','them','yours','yourself','thier']]):
        return False

    block_phrases = ['how are you','how about you','what about you','what are you up to',
                     'how are you doing', 'how was','how is','how are','how has',
                     'any recommendations','any recommendation','any suggestions',
                     'any suggestion', 'up to', 'any ideas', 'any advice', 'any tips',
                     'last', 'lately', 'recently', 'else', 'other']

    for phrase in block_phrases:
        if phrase in sent:
            return False

    if ('how' in sent or 'what' in sent) and ('going' in sent or 'doing' in sent or 'day' in sent):
        return False

    if sent in EXCLUDED_QUESTIONS:
        return False
    
    return True

def extract_questions(sent, valid_check = True):
    from nltk import sent_tokenize, word_tokenize

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

    new_question = [q for q in new_question if q != None and q != '']
    if valid_check:
        new_question = [q for q in new_question if is_valid_question(q)]
    
    return new_question