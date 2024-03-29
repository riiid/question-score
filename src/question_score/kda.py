from typing import List, Optional
from question_score.model import KDAModel
from transformers import AutoModelForMultipleChoice, AutoTokenizer, AdamW
import torch
import transformers
import math
from tqdm import tqdm

transformers.logging.set_verbosity_error()
import warnings
import torch.multiprocessing as mp
import numpy as np

KDA_SMALL = [
    "google/t5-small-ssm-nq",
    "Riiid/kda-albert-xlarge-v2-race",
    "Riiid/kda-mpnet-base-race",
    "Riiid/kda-scibert-uncased-race",
]

KDA_LARGE = [
    "google/t5-small-ssm-nq",
    "google/t5-large-ssm-nq",
    "Riiid/kda-albert-xlarge-v2-race",
    "Riiid/kda-mpnet-base-race",
    "Riiid/kda-scibert-uncased-race",
    "Riiid/kda-bert-base-uncased-race",
    "Riiid/kda-biobert-race",
    "Riiid/kda-roberta-base-race",
    "Riiid/kda-roberta-large-race",
    "Riiid/kda-xlnet-large-cased-race",
]

KDA_FULL = [
    "google/t5-small-ssm-nq",
    "google/t5-large-ssm-nq",
    "google/t5-xxl-ssm-nq",
    "Riiid/kda-biobert-large-race",
    "Riiid/kda-biobert-race",
    "Riiid/kda-mpnet-base-race",
    "Riiid/kda-scibert-uncased-race",
    "Riiid/kda-roberta-base-race",
    "Riiid/kda-roberta-large-race",
    "Riiid/kda-albert-xlarge-v2-race",
    "Riiid/kda-albert-xxlarge-v2-race",
    "Riiid/kda-distilbert-base-uncased-race",
    "Riiid/kda-distilroberta-base-race",
    "Riiid/kda-xlnet-base-cased-race",
    "Riiid/kda-xlnet-large-cased-race",
    "Riiid/kda-bert-large-uncased-race",
    "Riiid/kda-bert-base-uncased-race",
    "Riiid/kda-matscibert-race",
]

BERT_MAX_TOKEN_LEN = 512
TOK_MARGIN = 30


def get_bert_postfix(question: str, option: str):
    if question.find("_") != -1:
        return question.replace("_", " " + option + " ")
    return question + " " + option


class KDA:
    def __init__(self, mode="small", device_list=None):
        # Run KDA small as a default.
        # possible option for mode : small, large, full
        if mode == "small":
            self.model_list = KDA_SMALL
        elif mode == "large":
            self.model_list = KDA_LARGE
        elif mode == "full":
            self.model_list = KDA_FULL
        else:
            raise ValueError("mode should be one of small, large, full")

        if device_list is None:
            self.device_list = ["cpu" for _ in self.model_list]
            # self.device_list = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
        else:
            self.device_list = device_list

        if len(self.device_list) != len(self.model_list):
            raise ValueError("Number of device is not same with number of models.")

        self.models = {}
        print("loading models..")
        for model_name, device in tqdm(
            zip(self.model_list, self.device_list), total=len(self.model_list)
        ):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = KDAModel(model_name).to(device)
            self.models[model_name] = (model, tokenizer, device)

    def softmax(self, k):
        return np.exp(k) / sum(np.exp(k))

    def infer_model(
        self,
        model_name: str,
        kda_model,
        tokenizer,
        device: str,
        p: str,
        q: str,
        ops,
        res=None,
    ):
        # returns {'wf': [], 'wof': []}
        if "t5" in model_name:
            return self.infer_t5_model(
                model_name, kda_model, tokenizer, device, p, q, ops, res
            )
        return self.infer_bert_model(
            model_name, kda_model, tokenizer, device, p, q, ops, res
        )

    def infer_t5_model(self, model_name, kda_model, tokenizer, device, p, q, ops, res):
        result_dict = {}
        for prompt, key in [(p, "wf"), ("", "wof")]:
            src = [prompt + q for _ in ops]
            encoded_src = tokenizer(
                src,
                max_length=kda_model.t5_max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded_tgt = tokenizer(
                ops,
                max_length=kda_model.t5_max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            src_tokens = encoded_src["input_ids"].to(device)
            src_mask = encoded_src["attention_mask"].to(device)

            tgt_tokens = encoded_tgt["input_ids"].to(device)
            tgt_mask = encoded_tgt["attention_mask"]
            tgt_len = tgt_mask.sum(dim=1).to(device)

            output = kda_model.model(
                input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens
            )
            logits = output.logits.view(-1, kda_model.model.config.vocab_size)
            loss = kda_model.t5_loss_fct(kda_model.t5_lsm(logits), tgt_tokens.view(-1))
            loss = loss.view(tgt_tokens.shape[0], -1)
            loss = loss.sum(dim=1) / tgt_len
            result_dict[key] = [-x.item() for x in loss]
        if res is not None:
            res[model_name] = result_dict
        return result_dict

    def infer_bert_model(
        self, model_name, kda_model, tokenizer, device, p, q, ops, res
    ):
        result_dict = {}
        for prompt, key in [(p, "wf"), ("", "wof")]:
            input_id_list = []
            attention_mask_list = []
            token_type_ids_list = []

            for option in ops:
                inputs = tokenizer.encode_plus(
                    prompt + get_bert_postfix(q, option),
                    add_special_tokens=True,
                    max_length=BERT_MAX_TOKEN_LEN,
                )
                if model_name in kda_model.non_token_type_model_list:
                    token_type_ids = torch.zeros_like(
                        torch.tensor(inputs["input_ids"])
                    ).tolist()
                    input_ids = inputs["input_ids"]
                else:
                    input_ids, token_type_ids = (
                        inputs["input_ids"],
                        inputs["token_type_ids"],
                    )
                attention_mask = [1] * len(input_ids)
                pad_token_id = tokenizer.pad_token_id
                padding_length = 512 - len(input_ids)
                input_ids = input_ids + ([pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_id] * padding_length)
                input_id_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                token_type_ids_list.append(token_type_ids)

            # labels = torch.tensor(answer_idx).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

            # wo_fact_encoding = tokenizer([wo_fact, wo_fact, wo_fact, wo_fact], [row['option1'], row['option2'], row['option3'], row['option4']], return_tensors="pt", padding=True)
            # w_fact_encoding = tokenizer([w_fact, w_fact, w_fact, w_fact],
            #                              [row['option1'], row['option2'], row['option3'], row['option4']],
            #                              return_tensors="pt", padding=True)
            with torch.no_grad():
                batch = {
                    "input_ids": torch.tensor(input_id_list).unsqueeze(0).to(device),
                    "attention_mask": torch.tensor(attention_mask_list)
                    .unsqueeze(0)
                    .to(device),
                    "token_type_ids": torch.tensor(token_type_ids_list)
                    .unsqueeze(0)
                    .to(device),
                }
                output = kda_model(batch)  # batch size is 1
                result_dict[key] = output.logits.detach().tolist()[0]
        if res is not None:
            res[model_name] = result_dict
        return result_dict

    def get_correct_prob(self, outputs, answer_idx):
        return self.softmax(outputs)[answer_idx]

    def get_correctness(self, outputs, answer_idx):
        return self.softmax(outputs)[answer_idx] == max(self.softmax(outputs))

    def get_model_outputs(
        self, p: str, q: str, ops: List[str], model_list, do_multiprocess
    ):
        if do_multiprocess:
            res = mp.Manager().dict()
            mp.set_start_method("spawn", force=True)
            processes = []
            for model_name in model_list:
                process = mp.Process(
                    target=self.infer_model,
                    args=(model_name, *self.models[model_name], p, q, ops, res),
                )
                process.start()
                processes.append(process)
            for process in processes:
                process.join()
            return res

        res = {}
        for model_name in model_list:
            res[model_name] = self.infer_model(
                model_name, *self.models[model_name], p=p, q=q, ops=ops
            )
        return res

    def calc_kda(
        self,
        p: str,
        q: str,
        ops: List[str],
        answer_idx: int,
        model_list,
        do_multiprocess,
        is_disc=False,
    ):
        res = self.get_model_outputs(p, q, ops, model_list, do_multiprocess)
        if is_disc:
            prior_corr_list = [
                (1 if self.get_correctness(x["wof"], answer_idx) else 0)
                for x in res.values()
            ]
            with_fact_corr_list = [
                (1 if self.get_correctness(x["wf"], answer_idx) else 0)
                for x in res.values()
            ]
            if sum(prior_corr_list) == len(prior_corr_list):
                warnings.warn(
                    "All model picked correct answer, so can not calculate KDA_disc."
                    "The KDA value will be replaced by 0, please use KDA_cont."
                )
                return 0
            return sum(
                [
                    1 if not wof and wf else 0
                    for wof, wf in zip(prior_corr_list, with_fact_corr_list)
                ]
            ) / sum([1 if not wof else 0 for wof in prior_corr_list])

        upper = [
            (1 - self.get_correct_prob(x["wof"], answer_idx))
            * self.get_correct_prob(x["wf"], answer_idx)
            for x in res.values()
        ]
        lower = [1 - self.get_correct_prob(x["wof"], answer_idx) for x in res.values()]
        return {"score": sum(upper) / sum(lower), "model_outputs": res}

    def split_passages(
        self, p: str, q: str, ops: List[str], model_list
    ) -> List[str]:  # split passages if prompt exceeds maximum tokens.
        target_split_cnt = 1
        target_tokenizer = None
        max_postfix_token_len = 0
        for model in model_list:
            if "t5" in model:
                continue
            tok = self.models[model][1]
            max_postfix_len = -1
            for option in ops:
                postfix = get_bert_postfix(q, option)
                token_len = len(
                    tok.encode_plus(postfix, add_special_tokens=True)["input_ids"]
                )
                max_postfix_len = max(max_postfix_len, token_len)
            passage_token_len = len(
                tok.encode_plus(p, add_special_tokens=False)["input_ids"]
            )
            if passage_token_len + max_postfix_len > BERT_MAX_TOKEN_LEN:
                required_split_cnt = math.ceil(
                    passage_token_len
                    / (BERT_MAX_TOKEN_LEN - TOK_MARGIN - max_postfix_len)
                )
                if required_split_cnt > target_split_cnt:
                    target_split_cnt = required_split_cnt
                    target_tokenizer = tok
                    max_postfix_token_len = max(max_postfix_len, max_postfix_token_len)

        if target_split_cnt == 1:
            return [p]

        passage_list: List[str] = []
        tokenized = target_tokenizer.encode_plus(p, add_special_tokens=False)[
            "input_ids"
        ]
        # passage size : BERT_MAX_TOKEN_LEN - max_postfix_token_len - TOK_MARGIN
        for i in range(target_split_cnt):
            passage_len = BERT_MAX_TOKEN_LEN - max_postfix_token_len - TOK_MARGIN
            start_position = i * math.ceil(
                (len(tokenized) - passage_len) / (target_split_cnt - 1)
            )
            end_position = min(start_position + passage_len, len(tokenized))
            passage_list.append(
                target_tokenizer.decode(tokenized[start_position:end_position])
            )
        return passage_list

    def get_kda(
        self, p, q, ops, answer_idx, model_list, do_multiprocess, is_disc=False
    ):
        passage_list: List[str] = self.split_passages(p, q, ops, model_list)
        if len(passage_list) == 1:
            return self.calc_kda(
                p, q, ops, answer_idx, model_list, do_multiprocess, is_disc
            )
        warnings.warn(
            f"The passage is too long, so it is divided into {len(passage_list)} passages and returns the largest kda score. In this case, it is not a benchmark verified performance, so please enter a shorter fingerprint. It will take huge time to calculate."
        )
        kda_list = [
            self.calc_kda(
                passage_split, q, ops, answer_idx, model_list, do_multiprocess, is_disc
            )
            for passage_split in tqdm(passage_list)
        ]
        return max(kda_list, key=lambda x: x["score"])

    def kda_cont(
        self,
        p: str,
        q: str,
        ops: List[str],
        answer_idx: int,
        do_multiprocess: Optional[bool] = False,
    ):
        if len(self.model_list) < len(KDA_FULL):
            warnings.warn("KDA_cont needs all models, so reload all models")
            self.__init__(mode="full")
        return self.get_kda(p, q, ops, answer_idx, KDA_FULL, do_multiprocess, False)

    def kda_disc(
        self,
        p: str,
        q: str,
        ops: List[str],
        answer_idx: int,
        do_multiprocess: Optional[bool] = False,
    ):
        if len(self.model_list) < len(KDA_FULL):
            warnings.warn("KDA_disc needs all models, so reload all models")
            self.__init__(mode="full")
        return self.get_kda(p, q, ops, answer_idx, KDA_FULL, do_multiprocess, True)

    def kda_large(
        self,
        p: str,
        q: str,
        ops: List[str],
        answer_idx: int,
        do_multiprocess: Optional[bool] = False,
    ):
        if len(self.model_list) < len(KDA_LARGE):
            warnings.warn("KDA_large needs more models, so reload 6 more models")
            self.__init__(mode="large")
        return self.get_kda(p, q, ops, answer_idx, KDA_LARGE, do_multiprocess, False)

    def kda_small(
        self,
        p: str,
        q: str,
        ops: List[str],
        answer_idx: int,
        do_multiprocess: Optional[bool] = False,
    ):
        return self.get_kda(p, q, ops, answer_idx, KDA_SMALL, do_multiprocess, False)


if __name__ == "__main__":
    kda = KDA("small")
    print("model load complete")
    print(
        kda.kda_large(
            "Puppy is a baby of dog.",
            "What is the term for baby dog?",
            ["cat", "puppy", "kitty", "cow"],
            answer_idx=1,
        )
    )
