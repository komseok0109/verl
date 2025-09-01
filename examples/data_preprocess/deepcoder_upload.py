# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Preprocess the DeepCoder-Train dataset to parquet format
"""

import argparse
import os
import json

import datasets
from datasets import concatenate_datasets, DatasetDict
from huggingface_hub import create_repo, upload_file

# from verl.utils.hdfs_io import copy, makedirs


from huggingface_hub import upload_file
import tempfile, os, json

def push_jsonl_streaming(dataset, repo_id, split, shard_size=50000):
    file_index = 0
    buffer = []
    line_count = 0

    for ex in dataset:
        buffer.append(json.dumps(ex, ensure_ascii=False))
        line_count += 1

        if line_count >= shard_size:
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmpf:
                tmpf.write("\n".join(buffer))
                tmp_path = tmpf.name

            filename = f"{split}_{file_index:05d}.jsonl"
            upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset"
            )
            os.remove(tmp_path)

            buffer = []
            line_count = 0
            file_index += 1
    if buffer:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmpf:
            tmpf.write("\n".join(buffer))
            tmp_path = tmpf.name

        filename = f"{split}_{file_index:05d}.jsonl"
        upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset"
        )
        os.remove(tmp_path)



def extract_test_case(tests, args, branch):
    item = {}
    if branch == "taco":
        assert len(tests["inputs"]) == len(tests["outputs"]), "Mismatch in number of inputs and outputs"
        max_tests = min(len(tests["inputs"]), args.max_gt_tests)
        item["inputs"] = tests["inputs"][:max_tests]
        item["outputs"] = tests["outputs"][:max_tests]
    else:
        inputs = [d["input"] for d in tests]
        outputs = [d["output"] for d in tests]
        assert len(inputs) == len(outputs), "Mismatch in number of inputs and outputs"
        max_tests = min(len(inputs), args.max_gt_tests)
        item["inputs"] = inputs[:max_tests]
        item["outputs"] = outputs[:max_tests]
    return item

THRESHOLD = 80 * 1024 * 1024  # 100MB

def mark_large_rows(example, idx):
    gt = example["reward_model"]["ground_truth"]
    if len(gt.encode("utf-8")) >= THRESHOLD:
        print(f"{idx} DELETEd")
        return False   
    return True     

def reindex_batch(batch, indices):
    batch["extra_info"] = [
        {**ei, "index": idx} for ei, idx in zip(batch["extra_info"], indices)
    ]
    return batch



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/deepcoder")
    parser.add_argument("--max_gt_tests", type=int, default=15)

    args = parser.parse_args()

    data_source = "agentica-org/DeepCoder-Preview-Dataset"

    prime_train_data = datasets.load_dataset(data_source, "primeintellect")["train"]
    taco_train_data = datasets.load_dataset(data_source, "taco")["train"]
    lcb_train_data = datasets.load_dataset(data_source, "lcbv5")["train"]
    cf_test_data = datasets.load_dataset(data_source, "codeforces")["test"]
    lcb_test_data = datasets.load_dataset(data_source, "lcbv5")["test"]

    primeintellect_prompt = """Solve the following coding problem using the programming language python:

    {problem}

    The input will be stdin and you should print your solution to stdout. You should use input() to input and print() to output in your code.

    Now solve the problem and return the code."""

    # add a row to each data item that represents a unique id
    def make_map_fn(split, branch, args, start_index):
        def train_process_fn(batch, indices):
            data_sources = []
            abilitys = []
            prompts = []
            reward_models = []
            extra_infos = []
            i = 0
            for q, idx in zip(batch["problem"], indices):
                tests = json.loads(batch["tests"][i])
                if branch == "primeintellect":
                    question = q
                else:
                    question = primeintellect_prompt.format(problem=q)
                test_cases = extract_test_case(tests, args, branch)
                data = {
                    "data_source": branch,
                    "prompt": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ],
                    "ability": "code",
                    "reward_model": {"style": "rule", "ground_truth": json.dumps(test_cases)},
                    "extra_info": {
                        "split": split,
                        "index": start_index + idx,
                    },
                }
                data_sources.append(data["data_source"])
                prompts.append(data["prompt"])
                abilitys.append(data["ability"])
                extra_infos.append(data["extra_info"])
                reward_models.append(data["reward_model"])
                i += 1
            
            return {
                "prompt": prompts,
                "reward_model": reward_models,
                "extra_info": extra_infos,
                "ability": abilitys,
                "data_source": data_sources
            }

        def test_process_fn(example, idx):
            question_raw = example.pop("problem")
            question = primeintellect_prompt.format(problem=question_raw)
            tests = json.loads(example.pop("tests"))
            test_inputs = []
            test_outputs = []
            for i, t in enumerate(tests):
                if i == args.max_gt_tests - 1:
                    break
                test_inputs.append(t["input"])
                test_outputs.append(t["output"])
            data = {
                "data_source": branch,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": json.dumps({"inputs": test_inputs, "outputs": test_outputs})},
                "extra_info": {
                    "split": split,
                    "index": start_index + idx,
                },
            }
            return data
        return train_process_fn if split == "train" else test_process_fn

    prime_train_dataset = prime_train_data.map(function=make_map_fn("train", "primeintellect", args, 0), with_indices=True, batched = True, batch_size = 100, remove_columns =["problem", "solutions", "tests"])
    taco_train_dataset = taco_train_data.map(function=make_map_fn("train", "taco", args, len(prime_train_data)), with_indices=True, batched = True, batch_size = 100, remove_columns =["problem", "solutions", "tests"])
    lcb_train_dataset = lcb_train_data.map(function=make_map_fn("train", "livecodebench", args, len(prime_train_data) + len(taco_train_data)), with_indices=True, batched = True, batch_size = 100, remove_columns =['problem', 'starter_code', 'tests', 'metadata'])
    cf_test_dataset = cf_test_data.map(function=make_map_fn("test", "codeforces", args, 0),with_indices = True, remove_columns = ["problem", "tests"])
    lcb_test_dataset = lcb_test_data.map(function=make_map_fn("test", "livecodebench", args, len(cf_test_data)), remove_columns = ['problem', 'starter_code', 'tests', 'metadata'], with_indices=True)

    train_dataset = concatenate_datasets([prime_train_dataset, taco_train_dataset, lcb_train_dataset])
    test_dataset = concatenate_datasets([cf_test_dataset, lcb_test_dataset])

    train_dataset = train_dataset.filter(mark_large_rows, with_indices=True)
    train_dataset = train_dataset.map(reindex_batch, with_indices=True, batched = True, batch_size = 100)

    push_jsonl_streaming(train_dataset, "ming9999/deepcoder-train", "train", shard_size=500)
    push_jsonl_streaming(test_dataset, "ming9999/deepcoder-train", "test")

    # local_dir = args.local_dir

    # train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    # test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # dataset_dict = DatasetDict({
    #     "train": train_dataset,
    #     "test": test_dataset
    # })

    # dataset_dict.push_to_hub("ming9999/deepcoder")