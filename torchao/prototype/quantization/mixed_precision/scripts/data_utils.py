# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import random


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """Build a WikiText-2 calibration set.

    Returns:
        trainloader: list of ``(input_ids, target_ids)`` tuples sampled from the
            training split, with all but the final target token masked to -100.
        testenc: the tokenized test split (a ``BatchEncoding``). Use
            ``testenc.input_ids`` when only the token ids are needed.
    """
    # Imported lazily so importing this helper does not require ``datasets`` to
    # be installed unless the data is actually loaded.
    from datasets import load_dataset

    traindata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_wikitext2_test_input_ids(nsamples, seed, seqlen, tokenizer):
    """Same as :func:`get_wikitext2` but returns only the test split token ids
    (``testenc.input_ids``) instead of the full ``BatchEncoding``."""
    trainloader, testenc = get_wikitext2(nsamples, seed, seqlen, tokenizer)
    return trainloader, testenc.input_ids
