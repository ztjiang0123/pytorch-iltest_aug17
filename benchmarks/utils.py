# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F
from tqdm import tqdm
from triton.testing import do_bench


def bench_fwd_bwd_microseconds(fn, *args, use_compile=False, fullgraph=True, **kwargs):
    # Run once to get output shape for labels
    with torch.no_grad():
        out_sample = fn(*args, **kwargs)
    labels = torch.ones_like(out_sample)

    def fwd_bwd(*args, **kwargs):
        out = fn(*args, **kwargs)
        loss = F.mse_loss(out, labels)
        loss.backward()

    fwd_bwd_compiled = (
        torch.compile(fwd_bwd, fullgraph=fullgraph) if use_compile else fwd_bwd
    )
    return benchmark_cuda_function_in_microseconds(
        fwd_bwd_compiled,
        *args,
        **kwargs,
    )


def bench_fwd_microseconds(fn, *args, use_compile=False, fullgraph=True, **kwargs):
    fn_compiled = torch.compile(fn, fullgraph=fullgraph) if use_compile else fn

    def inference_fn(*args, **kwargs):
        with torch.no_grad():
            return fn_compiled(*args, **kwargs)

    return benchmark_cuda_function_in_microseconds(
        inference_fn,
        *args,
        **kwargs,
    )


def profile_fwd_bwd(
    fn,
    *args,
    use_compile=False,
    fullgraph=True,
    profile_name="profile",
    **kwargs,
):
    # Run once to get output shape for labels
    with torch.no_grad():
        out_sample = fn(*args, **kwargs)
    labels = torch.ones_like(out_sample)

    fn = torch.compile(fn, fullgraph=fullgraph) if use_compile else fn
    wait, warmup, active = 1, 3, 1
    total_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=0
        ),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(total_steps):
            out = fn(*args, **kwargs)
            loss = F.mse_loss(out, labels)
            loss.backward()
            prof.step()

    # Save profiler results
    prof.export_chrome_trace(f"{profile_name}.json")
    print(f"Saved: {profile_name}.json")


def profile_fn(fn, *args, profile_name="profile", distributed=False, **kwargs):
    wait, warmup, active = 1, 1, 1
    total_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=0
        ),
        record_shapes=True,
    ) as prof:
        for _ in range(total_steps):
            _ = fn(*args, **kwargs)
            prof.step()

    if distributed:
        rank = torch.distributed.get_rank()
        prof.export_chrome_trace(f"{profile_name}_rank{rank}.json")
        print(f"Saved: {profile_name}_rank{rank}.json")
    else:
        prof.export_chrome_trace(f"{profile_name}.json")
        print(f"Saved: {profile_name}.json")


def benchmark_cuda_function_in_microseconds(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e3


def run_experiments_and_print(
    get_configs,
    run_experiment,
    print_results,
    experiment_cls,
    *,
    seed=123,
    run_experiment_kwargs=None,
):
    """Shared driver for the ``main()`` of the config/experiment benchmark scripts.

    Seeds the RNG, builds configs via ``get_configs()``, runs ``run_experiment``
    on each, wraps the ``(config, result)`` pair in ``experiment_cls`` and hands
    the collected experiments to ``print_results``. Every benchmark script that
    follows the config -> run_experiment -> print_results pattern shares this
    driver instead of copying it.
    """
    run_experiment_kwargs = run_experiment_kwargs or {}
    torch.random.manual_seed(seed)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config, **run_experiment_kwargs)
        results.append(experiment_cls(config=config, result=result))
    print_results(results)
