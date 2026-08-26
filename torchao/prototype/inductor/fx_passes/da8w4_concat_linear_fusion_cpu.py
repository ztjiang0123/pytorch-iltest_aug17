# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch


# Inductor FX passes for concat linear for DA8W4
def _is_valid_concat_linear_da8w4_fusion(computation_nodes):
    if "CPU" not in torch._C._dispatch_dump("torchao::da8w4_linear_cpu"):
        # cpp kernels not built
        return False
    # OP schema:
    # da8w4_linear_cpu(Tensor input, Tensor input_scales, Tensor input_qzeros, Tensor weight, Tensor weight_scales, Tensor weight_qzeros, Tensor compensation, Tensor? bias, ScalarType output_dtype) -> Tensor
    computation_op = torch.ops.torchao.da8w4_linear_cpu.default
    act = computation_nodes[0].args[0]
    act_scales = computation_nodes[0].args[1]
    act_zp = computation_nodes[0].args[2]
    wgt = computation_nodes[0].args[3]
    in_feature_size = act.meta.get("val").size(1)  # type: ignore[union-attr]
    if len(wgt.meta.get("val").shape) != 4:
        return False
    block_k = wgt.meta.get("val").size(2)  # type: ignore[union-attr]
    with_bias = computation_nodes[0].args[7] is not None
    output_dtype = computation_nodes[0].args[-1]

    def check_in_feature_of_wgt(wgt):
        return (
            wgt.meta.get("val").size(1) * wgt.meta.get("val").size(2) == in_feature_size
        )  # type: ignore[union-attr]

    def check_block_k_of_wgt(wgt):
        return wgt.meta.get("val").size(2) == block_k

    def check_bias(b):
        return (b is not None) if with_bias else (b is None)

    return len(computation_nodes) >= 2 and all(
        (
            node.target == computation_op
            and node.args[0] == act  # share same activation
            and node.args[1] == act_scales  # same act scale
            and node.args[2] == act_zp  # same act zero point
            and check_in_feature_of_wgt(node.args[3])  # same in-feature size
            and (node.args[3] != wgt or gemm_idx == 0)
            and node.args[3].op == "get_attr"  # wgt are all constants
            and check_block_k_of_wgt(node.args[3])  # same block_k
            and check_bias(node.args[7])  # bias is either all None or all not None
            and node.args[-1] == output_dtype  # same output dtype
        )
        for gemm_idx, node in enumerate(computation_nodes)
    )


def _is_fusable_da8w4_node(node: torch.fx.Node) -> bool:
    """A da8w4 linear node is fusable only if it is a live CPU tensor op."""
    return (
        not node._erased
        and isinstance(node.meta.get("val"), torch.Tensor)
        and node.meta["val"].device.type == "cpu"
    )


def _register_concat_buffer(gm, name: str, tensor: torch.Tensor) -> None:
    gm.register_buffer(name, tensor)
    setattr(gm, name, tensor)


def _build_concat_buffers(gm, users, with_bias: bool):
    """Concat the per-linear constant weights/scales/qzeros/compensation (and
    optionally bias) along N/block_n, register them as buffers on ``gm``, and
    return the buffer names plus the resulting out-feature split sizes.
    """
    computation_node_0 = users[0]

    def gather(arg_idx):
        return [getattr(gm, user.args[arg_idx].target) for user in users]

    def concat_name(arg_idx):
        return computation_node_0.args[arg_idx].target + "_concat"

    # Shape of packed weight: [N/block_n, K/block_k, block_k, block_n/2]
    # Shape of weight scales/qzeros: [N/block_n, G, block_n]
    # Shape of compensation: [N/block_n, K/block_k, block_n]
    # Concat them along N/block_n
    packed_wgts = gather(3)
    out_feature_size_list = [(w.size(0) * w.size(-1) * 2) for w in packed_wgts]

    names = {
        "weight": concat_name(3),
        "scales": concat_name(4),
        "qzeros": concat_name(5),
        "compensation": concat_name(6),
        "bias": concat_name(7) if with_bias else None,
    }
    _register_concat_buffer(gm, names["weight"], torch.cat(packed_wgts, dim=0))
    _register_concat_buffer(gm, names["scales"], torch.cat(gather(4), dim=0))
    _register_concat_buffer(gm, names["qzeros"], torch.cat(gather(5), dim=0))
    _register_concat_buffer(gm, names["compensation"], torch.cat(gather(6), dim=0))
    if with_bias:
        _register_concat_buffer(gm, names["bias"], torch.cat(gather(7), dim=0))

    return names, out_feature_size_list


def _create_node_after(graph, anchor, *create_args, **create_kwargs):
    """Create a node inserted immediately after ``anchor`` and return it."""
    with graph.inserting_after(anchor):
        return graph.create_node(*create_args, **create_kwargs)


def _create_concat_get_attr_nodes(graph, names, anchor):
    """Create the chained get_attr nodes for the concatenated buffers.

    Returns (weight_node, {scales/qzeros/compensation nodes}, bias_node, last_node).
    ``bias_node`` is None when there is no bias.
    """
    weight_node = _create_node_after(graph, anchor, "get_attr", names["weight"], (), {})
    prev = weight_node
    get_attr_nodes = {}
    for key in ("scales", "qzeros", "compensation"):
        prev = _create_node_after(graph, prev, "get_attr", names[key], (), {})
        get_attr_nodes[key] = prev

    bias_node = None
    if names["bias"] is not None:
        bias_node = _create_node_after(graph, prev, "get_attr", names["bias"], (), {})
        prev = bias_node

    return weight_node, get_attr_nodes, bias_node, prev


def _replace_users_with_split(graph, split_node, users):
    """Replace each original linear ``user`` with getitem(split, i) + clone."""
    anchor = split_node
    for gemm_idx, user in enumerate(users):
        get_item = _create_node_after(
            graph, anchor, "call_function", operator.getitem, (split_node, gemm_idx)
        )
        clone_node = _create_node_after(
            graph,
            get_item,
            "call_function",
            torch.ops.aten.clone.default,
            (get_item,),
            {"memory_format": torch.contiguous_format},
        )
        user.replace_all_uses_with(clone_node)
        graph.erase_node(user)
        anchor = clone_node


def _fuse_da8w4_users(graph, node, users):
    """Replace the group of da8w4 linear ``users`` sharing ``node``'s activation
    with a single concatenated linear followed by a split.
    """
    gm = graph.owning_module
    computation_op = torch.ops.torchao.da8w4_linear_cpu.default
    act, act_scales, act_qzeros = node.args[0], node.args[1], node.args[2]
    output_dtype = node.args[-1]
    with_bias = users[0].args[7] is not None

    with graph.inserting_before(node):
        names, out_feature_size_list = _build_concat_buffers(gm, users, with_bias)
        concat_w_node = graph.create_node("get_attr", names["weight"], (), {})

    # Create the get_attr nodes for the concatenated buffers in order, each
    # inserted after the previous one, then the concat linear + split.
    _, get_attr_nodes, concat_bias_node, last_attr = _create_concat_get_attr_nodes(
        graph, names, concat_w_node
    )
    new_linear_node = _create_node_after(
        graph,
        last_attr,
        "call_function",
        computation_op,
        (
            act,
            act_scales,
            act_qzeros,
            concat_w_node,
            get_attr_nodes["scales"],
            get_attr_nodes["qzeros"],
            get_attr_nodes["compensation"],
            concat_bias_node,
            output_dtype,
        ),
    )
    split_node = _create_node_after(
        graph,
        new_linear_node,
        "call_function",
        torch.ops.aten.split_with_sizes.default,
        (
            new_linear_node,
            out_feature_size_list,
            -1,  # split along the out feature dimension
        ),
    )
    _replace_users_with_split(graph, split_node, users)


def _concat_linear_dq8w4_cpu(graph: torch.fx.Graph):
    """
    Concat Linear optimization pass for DA8W4 on CPU
    This pass fuses the original pattern:
    def ...
        return (da8w4_linear_cpu(x, ..., w1, ...), da8w4_linear_cpu(x, ..., w2, ...), ...)
    into a single operation:
    def ...
        concat_res = da8w4_linear_cpu(x, ..., concat_w, ...)
        return split(concat_res, split_size_list)
    """
    if "CPU" not in torch._C._dispatch_dump("torchao::da8w4_linear_cpu"):
        # cpp kernels not built
        return
    from torch._inductor import config as inductor_config

    if not inductor_config.cpp.enable_concat_linear:
        # only concat linear if the flag is set
        return
    computation_op = torch.ops.torchao.da8w4_linear_cpu.default
    # OP schema:
    # da8w4_linear_cpu(Tensor input, Tensor input_scales, Tensor input_qzeros, Tensor weight, Tensor weight_scales, Tensor weight_qzeros, Tensor compensation, Tensor? bias, ScalarType output_dtype) -> Tensor
    for node in graph.find_nodes(op="call_function", target=computation_op):
        if not _is_fusable_da8w4_node(node):
            continue
        act = node.args[0]
        users = list(act.users)
        if not _is_valid_concat_linear_da8w4_fusion(users):
            continue
        _fuse_da8w4_users(graph, node, users)


# Define and register a custom pass for concat linear
# We always register the pass when calling this function
# but it only takes effect when config.cpp.enable_concat_linear is set to True
def register_da8w4_concat_linear_cpu_pass():
    from torch._inductor import config as inductor_config

    inductor_config.post_grad_custom_post_pass = _concat_linear_dq8w4_cpu
