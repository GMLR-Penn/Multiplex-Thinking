import logging
from typing import List

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import crash_on_warnings, get_bool_env_var, is_cuda

if is_cuda():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )


logger = logging.getLogger(__name__)

SYNC_TOKEN_IDS_ACROSS_TP = get_bool_env_var("SYNC_TOKEN_IDS_ACROSS_TP")


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_nan_detection = global_server_args_dict["enable_nan_detection"]
        self.tp_sync_group = get_tp_group().device_group

        if global_server_args_dict["enable_dp_attention"]:
            self.tp_sync_group = get_attention_tp_group().device_group

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        # ==========
        # begin of soft thinking
        # ==========
        enable_soft_thinking: bool = False,
        # ==========
        # end of soft thinking
        # ==========
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_info: Metadata for sampling
            return_logprob: If set, store the output logprob information to
                logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
            batch_next_token_ids: next token IDs. If set, skip sampling and only
                compute output logprobs It is used for speculative decoding which
                performs sampling in draft workers.
        """
        logits = logits_output.next_token_logits

        # Apply the custom logit processors if registered in the sampling info.
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(logits, sampling_info)

        if self.use_nan_detection and torch.any(torch.isnan(logits)):
            logger.warning("Detected errors during sampling! NaN in the logits.")
            logits = torch.where(
                torch.isnan(logits), torch.full_like(logits, -1e5), logits
            )
            if crash_on_warnings():
                raise ValueError("Detected errors during sampling! NaN in the logits.")

        if sampling_info.is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            batch_next_token_ids = torch.argmax(logits, -1)
            if return_logprob:
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            # ==========
            # begin of soft thinking
            # ==========
            if enable_soft_thinking:
                raise ValueError("Soft thinking is not supported in greedy mode")
            # ==========
            # end of soft thinking
            # ==========
        else:
            logits.div_(sampling_info.temperatures)
            probs = torch.softmax(logits, dim=-1)

            if True:  # Keep this redundant check to simplify some internal code sync
                if global_server_args_dict["sampling_backend"] == "flashinfer":
                    # ==========
                    # begin of soft thinking
                    # ==========
                    if enable_soft_thinking:
                        if sampling_info.enable_max_topk:
                            soft_mask = sampling_info.soft_thinking_modes # Shape (B,)
                            top_ps = torch.where(soft_mask, sampling_info.top_ps, sampling_info.after_thinking_top_p)
                            top_ks = torch.where(soft_mask, sampling_info.top_ks, sampling_info.after_thinking_top_k)
                            min_ps = torch.where(soft_mask, sampling_info.min_ps, sampling_info.after_thinking_min_p)
                            dirichlet_alpha = sampling_info.dirichlet_alpha
                            enable_gumbel = sampling_info.enable_gumbel
                            gumbel_tau = sampling_info.gumbel_tau
                            enable_replacement = sampling_info.enable_replacement
                            enable_gumbel_after_thinking = sampling_info.enable_gumbel_after_thinking
                            early_stopping_entropy_threshold = sampling_info.early_stopping_entropy_threshold
                            
                            probs = top_k_renorm_prob(probs, top_ks)
                            probs = top_p_renorm_prob(probs, top_ps)
                            
                            if sampling_info.need_min_p_sampling or sampling_info.need_after_thinking_min_p_sampling: # slow
                                max_prob = probs.max(dim=-1, keepdim=True).values
                                min_p_thresholds = max_prob * min_ps.view(-1, 1)
                                min_p_mask = probs < min_p_thresholds
                                probs.masked_fill_(min_p_mask, 0.0)
                                probs = probs / probs.sum(dim=-1, keepdim=True)
                                
                            # Dirichlet
                            if not sampling_info.is_all_no_noise: # slow
                                conc = probs[soft_mask] * dirichlet_alpha
                                conc = torch.clamp(conc, min=torch.finfo(conc.dtype).min)
                                gamma_dist = torch.distributions.Gamma(conc, torch.ones_like(conc))
                                gamma_samples = gamma_dist.sample()
                                probs_new = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)
                                probs[soft_mask] = probs_new

                            # Apply Gumbel-Softmax trick selectively
                            topk_probs, topk_indices = torch.topk(probs, k=sampling_info.max_topk, dim=-1) # slow
                            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True))
                            
                            if enable_gumbel: 
                                topk_logits = torch.log(topk_probs)
                                gumbels = (
                                    -torch.empty_like(topk_logits)
                                    .exponential_()
                                    .log()
                                )  # ~Gumbel(0,1)
                                gumbels = (topk_logits + gumbels) / sampling_info.gumbel_softmax_temperatures  # ~Gumbel(logits,tau)
                                topk_probs = gumbels.softmax(-1)
                                sorted_weights, sorted_idx = torch.sort(topk_probs, dim=-1, descending=True)
                                topk_probs = sorted_weights
                                topk_indices = torch.gather(topk_indices, dim=1, index=sorted_idx)
                            
                            
                            non_soft_mask = ~soft_mask
                            if any(non_soft_mask):
                                sampled_token_ids = torch.multinomial(probs, num_samples=1)

                                # For rows where soft_thinking_modes is False
                                topk_probs[non_soft_mask] = 0.0
                                topk_indices[non_soft_mask] = 0

                                # Assign the first element of each row to sampled_token_ids and set it to 1.0 in topk_probs
                                topk_probs[non_soft_mask, 0] = 1.0
                                topk_indices[non_soft_mask, 0] = sampled_token_ids[non_soft_mask].view(-1)
                            logits_output.topk_probs = topk_probs
                            logits_output.topk_indices = topk_indices
                            batch_next_token_ids = topk_indices[:, 0].to(torch.int32)
                            
                        else:
                            soft_mask = sampling_info.soft_thinking_modes # Shape (B,)
                            top_ps = torch.where(soft_mask, sampling_info.top_ps, sampling_info.after_thinking_top_p)
                            top_ks = torch.where(soft_mask, sampling_info.top_ks, sampling_info.after_thinking_top_k)
                            min_ps = torch.where(soft_mask, sampling_info.min_ps, sampling_info.after_thinking_min_p)
                            dirichlet_alpha = sampling_info.dirichlet_alpha
                            enable_gumbel = sampling_info.enable_gumbel
                            enable_max_topk = sampling_info.enable_max_topk
                            gumbel_tau = sampling_info.gumbel_tau
                            enable_replacement = sampling_info.enable_replacement
                            enable_gumbel_after_thinking = sampling_info.enable_gumbel_after_thinking
                            enable_unweighting = sampling_info.enable_unweighting
                            early_stopping_entropy_threshold = sampling_info.early_stopping_entropy_threshold

                            # top k top p renorm
                            probs = top_k_renorm_prob(probs, top_ks)
                            probs = top_p_renorm_prob(probs, top_ps)
                            
                            if (sampling_info.enable_entropy_mask or early_stopping_entropy_threshold>0):
                                # compute entropy
                                entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=-1)
                                logits_output.entropy = entropy
                                if sampling_info.enable_entropy_mask:
                                    entropy_mask = entropy > sampling_info.entropy_mask_threshold
                                    soft_mask = soft_mask & entropy_mask

                            # minp renorm
                            if sampling_info.need_min_p_sampling or sampling_info.need_after_thinking_min_p_sampling: # slow
                                max_prob = probs.max(dim=-1, keepdim=True).values
                                min_p_thresholds = max_prob * min_ps.view(-1, 1)
                                min_p_mask = probs < min_p_thresholds
                                probs.masked_fill_(min_p_mask, 0.0)
                                probs = probs / probs.sum(dim=-1, keepdim=True)

                            # Dirichlet
                            if not sampling_info.is_all_no_noise: # slow
                                conc = probs[soft_mask] * dirichlet_alpha
                                conc = torch.clamp(conc, min=torch.finfo(conc.dtype).min)
                                gamma_dist = torch.distributions.Gamma(conc, torch.ones_like(conc))
                                gamma_samples = gamma_dist.sample()
                                probs_new = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)
                                probs[soft_mask] = probs_new
                                
                                
                            if enable_max_topk: # slow
                                topk_probs, topk_indices = torch.topk(probs, k=sampling_info.max_topk, dim=-1) # slow
                                topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True))


                            # Apply Gumbel-Softmax trick selectively
                            if enable_gumbel: 
                                # Apply Gumbel-Softmax trick on probs for samples where enable_gumbel is True
                                # Sample Gumbel noise: g_i ~ Gumbel(0, 1)
                                # Using the property that if u ~ Uniform(0,1), then -log(-log(u)) ~ Gumbel(0,1)
                                if enable_gumbel_after_thinking:
                                    uniform_noise = torch.rand_like(logits)
                                else:
                                    uniform_noise = torch.rand_like(logits[soft_mask])
                                uniform_noise.clamp_(min=1e-20, max=1.0 - 1e-7)

                                uniform_noise.log_()
                                uniform_noise.mul_(-1.0)
                                uniform_noise.log_()    
                                uniform_noise.mul_(-1.0)
                                gumbel_noise = uniform_noise
                                del uniform_noise

                                # Apply Gumbel-Softmax: y_i = exp((g_i + log(π_i))/τ) / Σ exp((g_k + log(π_k))/τ)
                                if enable_gumbel_after_thinking:
                                    log_probs = torch.log(torch.clamp(probs, min=1e-20))
                                    gumbel_logits = (gumbel_noise + log_probs) / gumbel_tau
                                    # gumbel_logits = (gumbel_noise + logits) / gumbel_tau
                                else:
                                    log_probs = torch.log(torch.clamp(probs[soft_mask], min=1e-20))
                                    # gumbel_logits = (gumbel_noise + logits[soft_mask]) / gumbel_tau
                                    gumbel_logits = (gumbel_noise + log_probs) / gumbel_tau
                                
                                # Apply softmax to get the Gumbel-Softmax distribution
                                gumbel_probs = torch.softmax(gumbel_logits, dim=-1)
                                
                                # Only apply Gumbel-Softmax to samples where enable_gumbel is True
                                # probs = torch.where(enable_gumbel.view(-1, 1), gumbel_probs, probs)
                                if enable_gumbel_after_thinking:
                                    del probs
                                    probs = gumbel_probs
                                else:
                                    probs[soft_mask] = gumbel_probs
                                
                            assert sampling_info.used_topk == sampling_info.max_topk
                            
                            topk_indices = torch.multinomial(probs, num_samples=sampling_info.max_topk, replacement=enable_replacement)  # (B, k)
                            
                            if enable_unweighting:
                                topk_probs = torch.ones_like(topk_probs,dtype=topk_probs.dtype,device=topk_probs.device)
                            else:
                                topk_probs = torch.gather(probs, dim=1, index=topk_indices)  # (B, k)
                            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)                   

                            # orginal sampling (after thinking)
                            non_soft_mask = ~soft_mask #shape
                            if torch.any(non_soft_mask):
                                # Note: dim parameter was added in PyTorch 1.11.0, use version-compatible approach
                                try:
                                    sampled_token_ids = torch.multinomial(probs, num_samples=1, dim=-1)
                                except TypeError:
                                    # Fallback for older PyTorch versions
                                    sampled_token_ids = torch.multinomial(probs, num_samples=1)

                                # For rows where soft_thinking_modes is False
                                topk_probs[non_soft_mask] = 0.0
                                topk_indices[non_soft_mask] = 0

                                # Assign the first element of each row to sampled_token_ids and set it to 1.0 in topk_probs
                                topk_probs[non_soft_mask, 0] = 1.0
                                topk_indices[non_soft_mask, 0] = sampled_token_ids[non_soft_mask].view(-1)
                            
                            logits_output.topk_probs = topk_probs
                            logits_output.topk_indices = topk_indices
                            batch_next_token_ids = topk_indices[:, 0].to(torch.int32)
                        # ==========
                        # end of soft thinking
                        # ==========
                    elif sampling_info.need_min_p_sampling:
                        probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                        probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                        batch_next_token_ids = min_p_sampling_from_probs(
                            probs, sampling_info.min_ps
                        )
                    else:
                        batch_next_token_ids = top_k_top_p_sampling_from_probs(
                            probs.contiguous(),
                            sampling_info.top_ks,
                            sampling_info.top_ps,
                            filter_apply_order="joint",
                            check_nan=self.use_nan_detection,
                        )
                elif global_server_args_dict["sampling_backend"] == "pytorch":
                    # A slower fallback implementation with torch native operations.
                    batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                        probs,
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        sampling_info.min_ps,
                        sampling_info.need_min_p_sampling,
                    )
                else:
                    raise ValueError(
                        f"Invalid sampling backend: {global_server_args_dict['sampling_backend']}"
                    )

            if return_logprob:
                # clamp to avoid -inf
                logprobs = torch.log(probs).clamp(min=torch.finfo(probs.dtype).min)

        # Attach logprobs to logits_output (in-place modification)
        if return_logprob:
            if any(x > 0 for x in top_logprobs_nums):
                (
                    logits_output.next_token_top_logprobs_val,
                    logits_output.next_token_top_logprobs_idx,
                ) = get_top_logprobs(logprobs, top_logprobs_nums)

            if any(x is not None for x in token_ids_logprobs):
                (
                    logits_output.next_token_token_ids_logprobs_val,
                    logits_output.next_token_token_ids_logprobs_idx,
                ) = get_token_ids_logprobs(logprobs, token_ids_logprobs)

            logits_output.next_token_logprobs = logprobs[
                torch.arange(len(batch_next_token_ids), device=sampling_info.device),
                batch_next_token_ids,
            ]

        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.

            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )

        return batch_next_token_ids


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def sampling_from_probs_torch(probs: torch.Tensor):
    """A sampling implementation with native pytorch operations, without
    top-k, top-p, or min-p filtering."""
    sampled_index = torch.multinomial(probs, num_samples=1)
    batch_next_token_ids = sampled_index.view(-1).to(torch.int32)
    return batch_next_token_ids


def top_p_normalize_probs_torch(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
):
    # See also top_k_top_p_min_p_sampling_from_probs_torch
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


def get_top_logprobs(logprobs: torch.Tensor, top_logprobs_nums: List[int]):
    max_k = max(top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    output_top_logprobs_val = []
    output_top_logprobs_idx = []
    for i, k in enumerate(top_logprobs_nums):
        output_top_logprobs_val.append(values[i][:k])
        output_top_logprobs_idx.append(indices[i][:k])
    return output_top_logprobs_val, output_top_logprobs_idx


def get_token_ids_logprobs(logprobs: torch.Tensor, token_ids_logprobs: List[List[int]]):
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []
    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None:
            output_token_ids_logprobs_val.append(logprobs[i, token_ids].tolist())
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append([])
            output_token_ids_logprobs_idx.append([])

    return output_token_ids_logprobs_val, output_token_ids_logprobs_idx


def apply_custom_logit_processor(
    logits: torch.Tensor,
    sampling_batch_info: SamplingBatchInfo,
    num_tokens_in_batch: int = 1,
):
    """Apply custom logit processors to the logits.
    This function will modify the logits in-place.
    num_tokens_in_batch is needed to support spec decoding, where each batch can contain multiple
    tokens. By default, we assume each batch contains only 1 token.
    """

    assert logits.shape[0] == len(sampling_batch_info) * num_tokens_in_batch, (
        f"The batch size of logits ({logits.shape[0]}) does not match the batch size of "
        f"sampling_batch_info ({len(sampling_batch_info)}) x num_tokens_in_batch "
        f"({num_tokens_in_batch})"
    )

    for _, (
        processor,
        batch_mask,
    ) in sampling_batch_info.custom_logit_processor.items():
        # Get the batch indices that need to be processed
        batch_indices = batch_mask.nonzero(as_tuple=True)[0]

        assert batch_mask.shape[0] == len(sampling_batch_info), (
            f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
            f"sampling_batch_info ({len(sampling_batch_info)})"
        )
        batch_mask = torch.repeat_interleave(batch_mask, num_tokens_in_batch)

        # Apply the processor to the logits
        logits[batch_mask] = processor(
            logits[batch_mask],
            [sampling_batch_info.custom_params[i] for i in batch_indices],
        )

        logger.debug(
            f"Custom logit processor {processor.__class__.__name__} is applied."
        )
