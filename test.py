import os
from tqdm import tqdm
import time
import torch
from torch.utils.data import DataLoader
from transformers.utils import logging
import json
import argparse
from utils.args import parse_test_args,parse_global_args,parse_dataset_args,parse_model_args
from utils.loader import load_models_tokenizer, load_test_dataset
from transformers import AutoTokenizer
from transformers.models.clip import modeling_clip

from collator import TestCollator

logging.set_verbosity(logging.INFO)
logger = logging.get_logger(__name__)

import math

def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    B = len(targets)
    if len(predictions)>4:
        predictions = [_.split("Response:")[-1] for _ in predictions]
        predictions = [_.strip().replace(" ","") for _ in predictions]
    # print(predictions,scores,targets)
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]
        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        # if b==1: 
        #     print(batch_seqs,batch_scores,target_item)
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results

def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError

    return res


def ndcg_k(topk_results, k):

    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit


class InferenceManager(object):
    def __init__(self,args):
        self.args = args
        self.metrics = args.metrics.split(",")
        

    def prepare_model(self):
        model, tokenizer = load_models_tokenizer(
            self.args, is_eval=True
        )
        print("Model prepared: \n{}".format(model))
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.offset = self.model.config.offset

    def model_infer(self, batch):
        inputs = batch[0].to(self.model.device)
        targets = batch[1]
        if self.args.use_mtp:
            topk=512
            start = time.time()
            output_ids,scores = self.model.mtp_generate(input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],topk=topk)
            end=time.time()
            output_ids=output_ids.reshape(-1,output_ids.shape[-1])
            scores = scores.flatten()
            output = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )

            topk_res = get_topk_results(output,scores,targets,topk,
                                        all_items= None)

        else:
            start = time.time()
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=3,
                # max_length=10,
                # prefix_allowed_tokens_fn=self.prefix_allowed_tokens,
                num_beams=self.args.num_beams,
                num_return_sequences=self.args.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )
            end=time.time()
            output_ids = output["sequences"]
            scores = output["sequences_scores"]

            output = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )

            topk_res = get_topk_results(output,scores,targets,self.args.num_beams,
                                        all_items= None)

        batch_metrics_res = get_metrics_results(topk_res, self.metrics)
        batch_metrics_res['latency'] = end-start
        return batch_metrics_res

    def predict(self):
        test_data = load_test_dataset(self.args)
        # prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(self.tokenizer)
        collator = TestCollator(self.args, self.tokenizer)
        test_loader = DataLoader(test_data, batch_size=self.args.test_batch_size, collate_fn=collator,
                            num_workers=2, pin_memory=True)
        with torch.no_grad():
            metrics_results = {}
            total = 0
            for step, batch in tqdm(enumerate(test_loader)):
                total+=len(batch[1])
                batch_metrics_res = self.model_infer(batch)
                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                if (step + 1) % 50 == 0:
                    temp = {}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / total
                    print(temp)
            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total
        os.makedirs(self.args.results_file, exist_ok=True)
        with open(self.args.results_file+'/res.json', "a") as f:
            json.dump(metrics_results, f, indent=4)
            json.dump(vars(self.args),f,indent=4)


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test')
    parser = parse_global_args(parser)
    parser = parse_test_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_model_args(parser)
    args = parser.parse_args()
    infer_engine = InferenceManager(args)

    infer_engine.prepare_model()
    infer_engine.predict()