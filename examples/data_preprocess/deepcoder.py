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

# from verl.utils.hdfs_io import copy, makedirs


def extract_test_case(tests, args):
    item = {}
    ground_truth = json.loads(tests["ground_truth"])
    if tests["dataset_type"] == "taco":
        assert len(ground_truth["inputs"]) == len(ground_truth["outputs"]), "Mismatch in number of inputs and outputs"
        max_tests = min(len(ground_truth["inputs"]), args.max_gt_tests)
        item["inputs"] = ground_truth["inputs"][:max_tests]
        item["outputs"] = ground_truth["outputs"][:max_tests]
    else:
        inputs = [d["input"] for d in ground_truth]
        outputs = [d["output"] for d in ground_truth]
        assert len(inputs) == len(outputs), "Mismatch in number of inputs and outputs"
        max_tests = min(len(inputs), args.max_gt_tests)
        item["inputs"] = inputs[:max_tests]
        item["outputs"] = outputs[:max_tests]
    return item


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/deepcoder-train")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--max_gt_tests", type=int, default=15)

    args = parser.parse_args()

    train_data = "justus27/deepcoder-train"
    test_data = "agentica-org/DeepCoder-Preview-Dataset"

    train_dataset = datasets.load_dataset(train_data)["train"]
    test_dataset = datasets.load_dataset(test_data, "codeforces")["test"]

    primeintellect_prompt = """Solve the following coding problem using the programming language python:

    {problem}

    The input will be stdin and you should print your solution to stdout. You should use input() to input and print() to output in your code.

    Now solve the problem and return the code."""

    # add a row to each data item that represents a unique id
    def make_map_fn(split, args):
        def train_process_fn(batch, indices):
            data_sources = []
            abilitys = []
            prompts = []
            reward_models = []
            extra_infos = []
            i = 0
            for q, idx in zip(batch["prompt"], indices):
                tests = json.loads(batch["verification_info"][i])
                if tests["dataset_type"] == "primeintellect":
                    question = q
                else:
                    question = primeintellect_prompt.format(problem=q)
                test_cases = extract_test_case(tests, args)
                data = {
                    "data_source": tests["dataset_type"],
                    "prompt": [
                        # {
                        #     "role": "system",
                        #     "content": 
                        # },
                        {
                            "role": "user",
                            "content": question,
                        }
                    ],
                    "ability": "code",
                    "reward_model": {"style": "rule", "ground_truth": json.dumps(test_cases)},
                    "extra_info": {
                        "split": split,
                        "index": idx,
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
                "data_source": "codeforces",
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
                    "index": idx,
                },
            }
            return data
        return train_process_fn if split == "train" else test_process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train", args), with_indices=True, batched=True, batch_size = 100)
    test_dataset = test_dataset.map(function=make_map_fn("test", args), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))