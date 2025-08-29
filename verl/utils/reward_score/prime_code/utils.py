# Copyright 2024 PRIME team and/or its affiliates
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

# Borrowed from: https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/utils.py

import multiprocessing
import os
import sys
import traceback
import re
from typing import Optional, Tuple

from .testing_util import run_test


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            res, metadata = run_test(in_outs=sample, test=generation, debug=debug, timeout=timeout)
            result.append(res)
            metadata_list.append(metadata)
        except Exception:
            # print(e) # some tracebacks are extremely long.
            traceback.print_exc(10)
            result.append([-1 for i in range(len(sample["inputs"]))])
            metadata_list.append({})


def check_correctness(in_outs: Optional[dict], generation, timeout=10, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(in_outs, generation, debug, result, metadata_list, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
        # p.terminate()
    if not result:
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print("global timeout")
    return result[0], metadata_list

OPEN = re.compile(r"<\s*think\s*>", re.IGNORECASE)
CLOSE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)


def extract_after_validation(text: str) -> Tuple[bool, str]:
    """
    Validate whether the input string contains exactly one <think>...</think> block
    with non-empty content inside, and return the text that follows the block.

    Validation rules:
        1. There must be exactly one <think> opening tag and one </think> closing tag.
        2. The tags must be in the correct order (open before close).
        3. The content between <think> and </think> must not be empty or whitespace-only.

    Args:
        text (str): The model-generated output string to check.

    Returns:
        Tuple[bool, str]:
            - bool: True if all validation conditions are satisfied, otherwise False.
            - str: The text that comes after the </think> closing tag
                   (empty string if validation fails).
    """
    open_matches = list(OPEN.finditer(text))
    close_matches = list(CLOSE.finditer(text))

    if len(open_matches) != 1 or len(close_matches) != 1:
        return False, ""

    open_m, close_m = open_matches[0], close_matches[0]

    if close_m.start() < open_m.end():
        return False, ""

    inside_content = text[open_m.end():close_m.start()].strip()
    if not inside_content:  
        return False, ""

    after_text = text[close_m.end():].strip()
    return True, after_text
