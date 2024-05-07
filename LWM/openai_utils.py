import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union

import openai
import tqdm
import copy


def init_openai():
    openai_org = os.getenv("OPENAI_ORG")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_org is not None and openai_api_key is not None:
        openai.organization = openai_org
        openai.api_key = openai_api_key


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 512
    temperature: float = 1
    top_p: float = 1.0
    n: int = 1
    seed: int = 42
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


def openai_completion(
    prompt: str,
    decoding_args: OpenAIDecodingArguments,
    model_name="gpt-3.5-turbo",
    sleep_time=2,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=True,
    **decoding_kwargs,
):
    batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args

    while True:
        try:
            shared_kwargs = dict(
                model=model_name,
                **batch_decoding_args.__dict__,
                **decoding_kwargs,
            )
            if "instruct" in model_name:
                completion = openai.completions.create(
                    prompt=prompt,
                    **shared_kwargs,
                )
                break
            else:
                completion = openai.chat.completions.create(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    **shared_kwargs,
                )
                break
        except openai.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
            else:
                logging.warning("Hit request rate limit; retrying...")
                time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text == True:
        if "instruct" in model_name:
            return completion.choices[0].text.strip()
        return completion.choices[0].message.content.strip()

    return completion


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
