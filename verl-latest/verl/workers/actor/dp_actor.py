
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os

import torch
import torch.distributed as dist

import itertools
import logging
import os
from typing import Tuple
import debugpy
import os
import torch
import torch._dynamo

from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.workers.config import ActorConfig
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs_for_topk_probs_and_indices, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        # --- START: ADD THIS BLOCK ---
        # This is the critical fix.
        # It tells this process that any new tensor should be created on the
        # assigned GPU by default, preventing the optimizer state from
        # being accidentally created on the CPU.
        if torch.cuda.is_available():
            # In a single-GPU run, rank will be 0. In multi-GPU, it will be 0, 1, 2...
            try:
                rank = dist.get_rank()
                torch.cuda.set_device(rank)
                print(f"INFO: Worker on rank {rank} successfully set default device to cuda:{rank}")
            except Exception as e:
                # If distributed is not initialized, we are on a single GPU, so use device 0
                torch.cuda.set_device(0)
                print(f"INFO: Worker (non-distributed context) set default device to cuda:0. Error: {e}")
        # --- END: ADD THIS BLOCK ---
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")


        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        
        # Gumbel parameters for soft thinking consistency
        self.enable_gumbel = self.config.get("enable_gumbel", False) and self.config.get("enable_soft_thinking", False)
        self.enable_max_topk = self.config.get("enable_max_topk", False) and self.config.get("enable_soft_thinking", False)
        self.gumbel_tau = self.config.get("gumbel_tau", 1.0)
        
        self.device_name = get_device_name()
        

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a micro batch with optional soft thinking support.
        
        Returns when soft_thinking_topk_indices is None:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        Returns when soft_thinking_topk_indices is not None:
            entropy: # (bs, response_len) - still per-token entropy
            log_probs: # (bs, response_len, topk) - weighted log probs for topk tokens
        """

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)


        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            
            # Detect soft thinking mode
            soft_thinking_topk_indices = micro_batch.get("soft_thinking_topk_indices", None)
            soft_thinking_topk_probs = micro_batch.get("soft_thinking_topk_probs", None)
            is_soft_thinking = soft_thinking_topk_indices is not None and soft_thinking_topk_probs is not None
            
            if soft_thinking_topk_probs is not None:
                soft_thinking_topk_probs = soft_thinking_topk_probs.to(torch.bfloat16)
            if soft_thinking_topk_indices is not None:
                soft_thinking_topk_indices = soft_thinking_topk_indices
                
                
            if is_soft_thinking:
                # we use prompts to construct full_topk_indices and full_topk_probs
                topk_size = soft_thinking_topk_indices.shape[-1]
                prompt_length = seqlen - response_length
                prompt_topk_indices = torch.full(
                    (batch_size, prompt_length, topk_size), 
                    0, 
                    dtype=soft_thinking_topk_indices.dtype, 
                    device=soft_thinking_topk_indices.device
                )

                prompt_topk_indices[:,:,0] = input_ids[:,:prompt_length]  # (batch_size, prompt_length)
                prompt_topk_probs = torch.zeros(
                    (batch_size, prompt_length, topk_size), 
                    dtype=soft_thinking_topk_probs.dtype, 
                    device=soft_thinking_topk_probs.device
                )
                prompt_topk_probs[:,:,0] = 1.0
                prompt_mask = attention_mask[:,:prompt_length]
                prompt_topk_probs = prompt_topk_probs * prompt_mask.unsqueeze(-1).float()
                full_topk_indices = torch.cat([prompt_topk_indices, soft_thinking_topk_indices], dim=1)  # (batch_size, seqlen, topk)
                full_topk_probs = torch.cat([prompt_topk_probs, soft_thinking_topk_probs], dim=1)  # (batch_size, seqlen, topk)
                
                if not full_topk_indices.is_contiguous():
                    full_topk_indices = full_topk_indices.contiguous()
                if not full_topk_probs.is_contiguous():
                    full_topk_probs = full_topk_probs.contiguous()
                    
                del soft_thinking_topk_indices, soft_thinking_topk_probs
                del prompt_topk_indices, prompt_topk_probs
                
            
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                if is_soft_thinking:
                    topk_indices_rmpad, indices, cu_seqlens, *_ = unpad_input(
                        full_topk_indices, attention_mask
                    )  # topk_indices_rmpad (total_nnz, topk)
                    topk_probs_rmpad, *_ = unpad_input(full_topk_probs, attention_mask)  # topk_probs_rmpad (total_nnz, topk)

                else:
                    input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    pass                    
                # for compute the log_prob - prepare labels for different modes
                if is_soft_thinking:
                    # For soft thinking, we'll use topk_indices as labels after rolling
                    assert topk_indices_rmpad.dim() == 2,"topk_indices_rmpad.dim() = {topk_indices_rmpad.dim()}"
                    topk_indices_rmpad_rolled = torch.roll(topk_indices_rmpad, shifts=-1, dims=0)  # ( total_nnz, topk)
                    # topk_probs_rmpad_rolled = torch.roll(topk_probs_rmpad, shifts=-1, dims=0)  # ( total_nnz, topk)
                    #embeddings_rmpad_rolled = torch.roll(embeddings_rmpad, shifts=-1, dims=1)  # (total_nnz, hidden_size)
                else:
                    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp: 
                    assert False, "use_ulysses_sp to be double checked"
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        pass
                    else:
                        if is_soft_thinking:
                            topk_probs_rmpad,topk_indices_rmpad,position_ids_rmpad, pad_size= ulysses_pad_and_slice_inputs_for_topk_probs_and_indices(
                                topk_probs_rmpad,
                                topk_indices_rmpad,
                                position_ids_rmpad=position_ids_rmpad,
                                sp_size=self.ulysses_sequence_parallel_size,
                            )

                        else:
                            input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                                input_ids_rmpad,
                                position_ids_rmpad=position_ids_rmpad,
                                sp_size=self.ulysses_sequence_parallel_size,
                            )

                    
                    if is_soft_thinking:

                        topk_probs_rmpad_rolled, topk_indices_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs_for_topk_probs_and_indices(
                            topk_probs_rmpad_rolled,
                            topk_indices_rmpad_rolled,
                            position_ids_rmpad=None,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )

                    else:
                        input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad_rolled,
                            position_ids_rmpad=None,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )

                if is_soft_thinking:
                    pass
                else:
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
                
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True


                if is_soft_thinking:
                    output = self.actor_module(
                        input_ids=None,
                        topk_probs=topk_probs_rmpad,
                        topk_indices=topk_indices_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )
                    
                else:
                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    assert False, "use_fused_kernels to be double checked"
                    if is_soft_thinking:
                        # For fused kernels + soft thinking, get logits and compute topk log probs
                        logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                        # temperature = 1.0                        logits_rmpad.div_(temperature)
                        log_probs_list = []
                        for k in range(topk_indices_rmpad.shape[1]):
                            # CRITICAL: We .clone() the logits tensor on each iteration. This prevents the in-place
                            # operations within logprobs_from_logits from corrupting the computation graph for the
                            # next iteration, which would otherwise cause a RuntimeError during backward().
                            log_prob_k = logprobs_from_logits(
                                logits=logits_rmpad.unsqueeze(0),
                                labels=topk_indices_rmpad_rolled[:, k].unsqueeze(0),
                                inplace_backward=inplace_backward,
                            ).squeeze(0)
                            log_probs_list.append(log_prob_k)
                        log_probs = torch.stack(log_probs_list, dim=1)
                    else:
                        log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    
                    # if calculate_entropy:
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    
                    logits_rmpad.div_(temperature)
                    
                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False


                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:

                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                    if is_soft_thinking:
                        soft_mask = ~((topk_probs_rmpad[:, 0] == 1.0) & (topk_probs_rmpad[:, 1:].sum(dim=-1) == 0.0))
                        
                        del full_topk_indices, full_topk_probs,topk_indices_rmpad,topk_probs_rmpad,position_ids_rmpad
                        if self.enable_gumbel:
                            logprobs = torch.log_softmax(logits_rmpad, dim=-1)
                            del logits_rmpad
                            log_probs = logprobs.gather(dim=-1, index=topk_indices_rmpad_rolled)
                            del logprobs,topk_indices_rmpad_rolled    
                        else:
                            logprobs = torch.log_softmax(logits_rmpad, dim=-1)
                            log_probs = logprobs.gather(dim=-1, index=topk_indices_rmpad_rolled)
                            del logprobs
                    else:
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                if self.use_ulysses_sp:
                    raise NotImplementedError("use_ulysses_sp to be double checked")
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )

                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                        
                # pad back to (bsz, seqlen) or (bsz, seqlen, topk)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1) if not is_soft_thinking else log_probs,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                    
                    if is_soft_thinking:
                        # Recreate soft_mask for response part only
                        # The soft_mask needs to be reshaped to match entropy shape (bsz, response_length)
                        soft_mask_full = pad_input(
                            hidden_states=soft_mask.float().unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        
                        soft_mask_response = soft_mask_full.squeeze(-1)[:, -response_length - 1 : -1].bool()  # (bsz, response_length)
                        entropy[soft_mask_response] = entropy[soft_mask_response] * topk_size
                        
                if is_soft_thinking:
                    log_probs = full_log_probs[:, -response_length - 1 : -1, :]  # (bsz, response_length, topk)
                else:
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:    # not using rmpad and no ulysses sp
                assert False, "not using rmpad and no ulysses sp to be double checked"
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                if is_soft_thinking:
                    
                    output = self.actor_module(
                        topk_indices=topk_indices_rmpad,
                        topk_probs=topk_probs_rmpad,
                        attention_mask=attention_mask,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )
                else:
                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    if is_soft_thinking:
                        # For fused kernels with soft thinking, get logits for topk computation
                        logits = output.logits
                        logits.div_(temperature)
                        logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                        log_probs_list = []
                        for i in range(topk_indices_rmpad.shape[-1]):
                            log_probs_list.append(logprobs_from_logits(
                                logits=logits,
                                labels=soft_thinking_topk_indices[:,i].squeeze(-1).unsqueeze(0), 
                                inplace_backward=inplace_backward,
                            ).squeeze(0))
                        log_probs = torch.stack(log_probs_list, dim=1)
                    else:
                        log_probs = output.log_probs[:, -response_length - 1 : -1]  # (bsz, response_length)
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)
                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    
                    if is_soft_thinking:
                        log_probs_list = []
                        for i in range(topk_indices_rmpad.shape[-1]):
                            log_probs_list.append(logprobs_from_logits(
                                logits=logits,
                                labels=soft_thinking_topk_indices[:,i].squeeze(-1).unsqueeze(0),
                                inplace_backward=inplace_backward,
                            ).squeeze(0))
                        log_probs = torch.stack(log_probs_list, dim=1)
                    else:
                        # Standard single-token log prob computation
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])  # (bsz, response_length)
                    
                    if calculate_entropy:
                        # entropy = self.compute_entropy_from_logits(logits)  # (bsz, response_length)
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    
    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
            
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm
    
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.


                ``soft_thinking_topk_indices``: tensor of shape [batch_size, response_length, topk]. torch.int64. when soft thinking is enabled, this key is required.

                ``soft_thinking_topk_probs``: tensor of shape [batch_size, response_length, topk]. torch.float32. when soft thinking is enabled, this key is required.

        Returns:
            torch.Tensor: the log_prob tensor
        """

        
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        # select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"] if "soft_thinking_topk_indices" not in data.batch.keys() else ["responses", "input_ids", "attention_mask", "position_ids", "soft_thinking_topk_indices", "soft_thinking_topk_probs"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)


        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )

            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)

        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys


    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = ["responses", "response_mask", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"] if "soft_thinking_topk_indices" not in data.batch.keys() else ["responses", "response_mask", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "soft_thinking_topk_indices", "soft_thinking_topk_probs"]
        # if multi_turn:
        #     select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")
            
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )

                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    soft_thinking_topk_indices = model_inputs.get("soft_thinking_topk_indices", None)
                    soft_thinking_topk_probs = model_inputs.get("soft_thinking_topk_probs", None)
                    
                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    
                    
                    # without soft thinking, all return: (bsz, response_length) 
                    # with soft thinking, log_prob return: (bsz, response_length, topk), entropy return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(micro_batch=model_inputs, temperature=temperature, calculate_entropy=calculate_entropy)
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    if loss_mode == "vanilla":
                        assert not self.config.enable_soft_thinking, "enable_soft_thinking is not supported for vanilla loss mode"
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_log_probs=rollout_log_probs,
                        )
                        answer_token_ratio = None
                    elif loss_mode == "multiplex_thinking":
                        assert self.config.enable_soft_thinking, "enable_soft_thinking is required for multiplex_thinking loss mode"
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, answer_token_ratio, response_mask_ratio = policy_loss_fn(old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                            topk_probs=soft_thinking_topk_probs,
                            enable_low_logprob_mask=self.config.optim.enable_low_logprob_mask,
                            recompute_topk_probs=self.config.optim.recompute_topk_probs,)
                        answer_token_ratio = None
                    else:
                        raise ValueError(f"Loss mode to be double checked under soft thinking mode: {self.config.policy_loss.loss_mode}")
                            
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor

                    loss.backward()
                    
                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                            
                        }
                    )
                    if answer_token_ratio is not None:
                        # Handle both tensor and float types
                        ratio_value = answer_token_ratio.item() if isinstance(answer_token_ratio, torch.Tensor) else answer_token_ratio
                        micro_batch_metrics.update({"actor/answer_token_ratio": ratio_value})
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics

