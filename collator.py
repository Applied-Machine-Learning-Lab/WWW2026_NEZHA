import torch
import copy
import argparse
from dataclasses import dataclass

import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration


class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.use_mtp = args.use_mtp
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        if self.use_mtp:
            self.special_token_ids = [len(tokenizer)-i for i in range(len(args.docid_num),0,-1)]
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        full_texts = [d["input_ids"] + d["labels"] + self.tokenizer.eos_token for d in batch]

        inputs = self.tokenizer(
            text = full_texts,
            text_target = input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        labels = copy.deepcopy(inputs["input_ids"])
        if self.only_train_response:
            # ignore padding
            labels[labels == self.tokenizer.pad_token_id] = -100
            # ignore input text
            labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100
        if self.use_mtp:
            response_mask = labels != -100
            num_response_tokens = torch.sum(response_mask)
            if num_response_tokens > 0:
                num_repeats = num_response_tokens // len(self.special_token_ids)
                mtp_sequence = torch.tensor(self.special_token_ids, device=inputs["input_ids"].device).repeat(num_repeats)
                inputs["input_ids"][response_mask] = mtp_sequence

        inputs["labels"] = labels
        return inputs



class TestCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.use_mtp = args.use_mtp
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"
        if self.use_mtp:
            self.special_tokens = [f"<sp_{i}>" for i in range(len(args.docid_num))]
            #[len(tokenizer)-i for i in range(len(args.docid_num),0,-1)]

    def __call__(self, batch):
        if self.use_mtp:
            input_texts = [d["input_ids"]+''.join(self.special_tokens)+ self.tokenizer.eos_token for d in batch]
        else:
            input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]
        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return (inputs, targets)

