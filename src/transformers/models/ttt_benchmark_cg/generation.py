# Copyright (c) 2023, Albert Gu, Tri Dao.
import gc
import pdb
import time
from collections import namedtuple
from dataclasses import dataclass, field
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.profiler import ProfilerActivity, profile, record_function
from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


class TttCache:
    def __init__(self, max_seqlen, max_batch_size, model):
        self.model = model
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        self.dtype = model.config.dtype
        self.inner_net = model.config.inner_net
        self.inner_chunk_size = model.config.inner_net_chunk_size
        self.params_dict = defaultdict(dict)
        if 'mlp_1' in self.inner_net:
            self.param_names = ["W1",]
        else:
            self.param_names = ["W1", "W2"]

    def allocate_inference_cache(self):
        for layer_idx in range(self.model.config.num_hidden_layers):
            for name in self.param_names:
                weight = getattr(self.model.model.layers[layer_idx].self_attn, name)
                tiled_weight = torch.tile(weight, (self.max_batch_size,) + (1,) * (weight.dim() - 1))  # [B*nh,f,f]
                self.params_dict[f"{name}_states"][layer_idx] = tiled_weight
                self.params_dict[f"{name}_grad"][layer_idx] = torch.zeros_like(tiled_weight)

    def update_last_in_chunk(self, py_tree, layer_idx):
        for name in self.param_names:
            self.params_dict[f"{name}_states"][layer_idx].copy_(py_tree[f"{name}_states"])
            self.params_dict[f"{name}_grad"][layer_idx].zero_()

    def update_non_last_in_chunk(self, py_tree, layer_idx):
        for name in self.param_names:
            self.params_dict[f"{name}_grad"][layer_idx].copy_(py_tree[f"{name}_grad"])

    def to_dic(self, layer_idx):
        return {name: self.params_dict[name][layer_idx] for name in self.params_dict}

    def reset(self, max_seqlen, max_batch_size, model, i=0):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        self.model = model
        for layer_idx in range(self.model.config.num_hidden_layers):
            for name in self.param_names:
                weight = getattr(self.model.model.layers[layer_idx].self_attn, name)
                tiled_weight = torch.tile(weight, (max_batch_size,) + (1,) * (weight.dim() - 1))  # [B*nh,f,f]
                self.params_dict[f"{name}_states"][layer_idx].copy_(tiled_weight  + i * 0.1)
                self.params_dict[f"{name}_grad"][layer_idx].zero_()


# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L231
def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for none top-k values to -inf. Done in-place."""
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits.masked_fill_(indices_to_remove, float("-Inf"))


# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L170
def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf. Done in-place."""
    if top_p <= 0.0 or top_p >= 1.0:
        return
    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits.masked_fill_(indices_to_remove, float("-inf"))


def sample(logits, top_k=1, top_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, "top-p should be in (0, 1]."
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            if temperature != 1.0:
                logits_top /= temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1),
            ]
        else:
            # Clone so that when we modify for top_p we don't change the original logits
            logits_top = logits / temperature if temperature != 1.0 else logits.clone()
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(
                dim=-1
            )


@torch.inference_mode()
def decode(
    input_ids,
    model,
    max_length,
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    eos_token_id=None,
    teacher_outputs=None,
    vocab_size=None,
    tensor_parallel=1,
    cg=False,
    enable_timing=False,
    i = 0,
):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
        teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
            logits, the next token is taken from the teacher_outputs. Useful for testing.
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    batch_size, seqlen_og = input_ids.shape
    teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
    if cg:
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None

        ## Capture is_last_in_chunk = False
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
            tensor_parallel=tensor_parallel,
            is_prefill=False,
            is_last_in_chunk=False,
        )

        inference_params = model._decoding_cache.inference_params

        inference_params.reset(max_length, batch_size, model, i)  # TODO: must reset keep the shape of inference_params?

        ## Capture is_last_in_chunk = True
        # model._decoding_cache = update_graph_cache(
        #     model,
        #     model._decoding_cache,
        #     batch_size,
        #     seqlen_og,
        #     max_length,
        #     tensor_parallel=tensor_parallel,
        #     is_prefill=False,
        #     is_last_in_chunk=True,
        # )
        # inference_params = model._decoding_cache.inference_params
        # inference_params.reset(max_length, batch_size, model)  # TODO: must reset keep the shape of inference_params?

    else:
        inference_params = TttCache(max_seqlen=max_length, max_batch_size=batch_size, model=model)


    def get_logits(input_ids, inference_params):
        decoding = inference_params.seqlen_offset > 0  # after prompt

        if not cg or not decoding:
            # before prompt
            is_last_in_chunk = (input_ids.shape[1] % inference_params.inner_chunk_size == 0)  # TODO: Prompt must be a multiple of CS
            logits = model(
                input_ids,
                is_prefill=True,
                is_last_in_chunk=is_last_in_chunk,
                cache_params=inference_params,
            ).logits.squeeze(dim=1)
        else:
            # after prompt: continue generating
            is_prefill = False
            # is_last_in_chunk = ((inference_params.seqlen_offset + 1) % inference_params.inner_chunk_size == 0)
            is_last_in_chunk = False
            logits = model._decoding_cache.run(
                input_ids, is_prefill, is_last_in_chunk
            ).squeeze(dim=1)  # [BS,decode_len,vocab_size]

        return logits[..., :vocab_size] if vocab_size is not None else logits


    def sample_tokens(logits, inference_params):
        if teacher_outputs is None or teacher_output_len <= inference_params.seqlen_offset:
            token = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        else:
            token = teacher_outputs[:, inference_params.seqlen_offset]
        # return rearrange(token, "b -> b 1")
        return token.unsqueeze(1)

    def should_stop(current_token, inference_params):
        if inference_params.seqlen_offset == 0:
            return False
        if eos_token_id is not None and (current_token == eos_token_id).all():
            return True
        if inference_params.seqlen_offset >= max_length - 1:
            return True
        return False

    start = torch.cuda.Event(enable_timing=enable_timing)
    end = torch.cuda.Event(enable_timing=enable_timing)

    if enable_timing:
        if tensor_parallel > 1:
            torch.distributed.barrier()
        start.record()

    scores, sequences = [], [input_ids]
    if input_ids.shape[1] == 1:
        inference_params.seqlen_offset = 1  # @xinhao: prompt=1 use decode mode directly as a hack

    while not should_stop(sequences[-1], inference_params):

        scores.append(
            get_logits(sequences[-1], inference_params)
        )

        inference_params.seqlen_offset += sequences[-1].shape[1]

        sequences.append(
            sample_tokens(scores[-1], inference_params)
        )

    if enable_timing:
        end.record()
        if tensor_parallel > 1:
            torch.distributed.barrier()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(start.elapsed_time(end)):.0f}ms")

    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput

    return output_cls(sequences=torch.cat(sequences, dim=1), scores=tuple(scores))


class GenerationMixin:
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError

    def generate(
        self,
        input_ids,
        max_length,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs,  # include cg
    ):
        output = decode(
            input_ids, self, max_length, top_k=top_k, top_p=top_p, temperature=temperature, **kwargs
        )
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences


@dataclass
class DecodingCGCache:
    max_batch_size: int = 0
    max_seqlen: int = 0
    device = None
    dtype = None
    callables: dict = field(default_factory=dict)
    mempool = None
    inference_params: Optional[InferenceParams | TttCache] = None
    run: Optional[Callable] = None


@torch.inference_mode()
def update_graph_cache(
    model,
    cache,
    batch_size,
    seqlen_og,
    max_seqlen,
    decoding_seqlens=(1,),
    tensor_parallel=1,
    dtype=None,
    n_warmups=2,
    is_prefill=False,
    is_last_in_chunk=False,
):
    if cache is None:
        cache = DecodingCGCache()

    param_example = next(iter(model.parameters()))
    device = param_example.device

    if dtype is None:
        dtype = param_example.dtype

    if (
        (device, dtype) != (cache.device, cache.dtype)
        or batch_size > cache.max_batch_size
        or max_seqlen > cache.max_seqlen
    ):  # Invalidate the cache (@xinhao: always activated the first time enter updte_graph_cache)
        cache.callables = {}
        cache.mempool = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = batch_size, max_seqlen
        cache.inference_params = TttCache(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            model=model,
        )
        cache.inference_params.allocate_inference_cache()
        cache.mempool = torch.cuda.graphs.graph_pool_handle()

    ###
    for decoding_seqlen in decoding_seqlens:

        if (batch_size, decoding_seqlen, is_prefill, is_last_in_chunk) not in cache.callables:

            # key: (batch_size, decoding_seqlen)=(bs, 1), val: a function returned by capture_graph
            cache.callables[batch_size, decoding_seqlen, is_prefill, is_last_in_chunk] = capture_graph(
                model,
                cache.inference_params,
                batch_size,
                max_seqlen,
                decoding_seqlen=decoding_seqlen,
                mempool=cache.mempool,
                n_warmups=n_warmups,
                is_prefill=is_prefill,
                is_last_in_chunk=is_last_in_chunk,
            )

    def dispatch(input_ids, is_prefill, is_last_in_chunk):
        batch_size, decoding_seqlen = input_ids.shape[:2]
        return cache.callables[batch_size, decoding_seqlen, is_prefill, is_last_in_chunk](input_ids)

    cache.run = dispatch
    cache.inference_params.seqlen_offset = 0  # Reset so it's not confusing
    return cache


def capture_graph(
        model,
        inference_params,
        batch_size, max_seqlen, decoding_seqlen=1,
        mempool=None,
        n_warmups=2,
        is_prefill=False,
        is_last_in_chunk=False
):
    device = next(iter(model.parameters())).device
    input_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    position_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)

    seqlen_offset_og = inference_params.seqlen_offset
    inference_params.seqlen_offset = max_seqlen - decoding_seqlen
    # inference_params.lengths_per_sample[:] = inference_params.seqlen_offset

    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            logits = model(
                input_ids,
                cache_params=inference_params,
                is_prefill=is_prefill,
                is_last_in_chunk=is_last_in_chunk,
            ).logits
        s.synchronize()
        # This might be needed for correctness if we run with NCCL_GRAPH_MIXING_SUPPORT=0,
        # which requires that graph launch and non-captured launch to not overlap (I think,
        # that's how I interpret the documentation). I'm not sure if this is required.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    torch.cuda.current_stream().wait_stream(s)

    # Captures the graph
    # To allow capture, automatically sets a side stream as the current stream in the context
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        logits = model(
            input_ids,
            cache_params=inference_params,
            is_prefill=is_prefill,
            is_last_in_chunk=is_last_in_chunk,
        ).logits

    def run(new_input_ids):
        input_ids.copy_(new_input_ids)
        graph.replay()
        return logits.clone()

    inference_params.seqlen_offset = seqlen_offset_og

    return run
