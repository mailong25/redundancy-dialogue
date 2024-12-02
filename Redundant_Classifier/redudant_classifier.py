# Load pretrained model and tokenizer
# In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.

import torch
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import torch
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class Redundant_Classifier:
    def __init__(self, model_name_or_path, ckpt_path):
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=self.config,
            ignore_mismatched_sizes=False,
        )

        label_to_id = {'no': 0, 'yes': 1}
        self.model.config.label2id = label_to_id
        self.model.config.id2label = {id: label for label, id in self.config.label2id.items()}
        
        self.model.load_state_dict(AutoModelForSequenceClassification.from_pretrained(ckpt_path).state_dict(), strict=False)
        self.model = self.model.to('cuda') if torch.cuda.is_available() else self.model
        self.model.eval()

    def predict(self, questions, personas, batch_size = 64):
        id_to_label = {0:'non-Redundant', 1: 'Redundant'}
        results = []
        
        batchs = list(chunks(list(zip(questions,personas)), batch_size))
        
        for batch in batchs:
            inputs = self.tokenizer(batch, padding=True, max_length=512, truncation=True, return_tensors = 'pt')
            for key in inputs:
                inputs[key] = inputs[key].to('cuda') if torch.cuda.is_available() else inputs[key]
            
            with torch.no_grad():
                outputs = self.model(**inputs).logits
                probs = torch.softmax(outputs, dim = 1)
                for i in range(0,len(probs)):
                    results.append({'non-Redundant': probs[i][0].item(), 'Redundant': probs[i][1].item()})
        
        return results