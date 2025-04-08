from types import MethodType
import torch
import torch.nn.functional as F
from trl import SFTTrainer
from ..configs import SFTConfig
from .sparse_attention_utils import local_attn_forward, CPLSHForCausalLM
from .repeat_attention import RepeatedForCausalLM

class SparseSFTTrainer(SFTTrainer):
    '''
    def _create_model_from_path(self, model_path: str, args: SFTConfig):
        model = super()._create_model_from_path(model_path, args)

        def patch_forward(module: torch.nn.Module) -> None:
            for name, child in module.named_children():
                if "self_attn" in name:
                    if args.sparse_attn == "local":
                        child.forward = MethodType(local_attn_forward, child)
                        if child.layer_idx not in args.full_attn_layers:
                            child.local = args.local
                    else:
                        raise NotImplementedError(f"Sparse attention type {args.sparse_attn} not implemented")
                        # child.sink = args.sink
                        # child.local = args.local
                        # child.topk = args.sparse_attn_topk
                else:
                    patch_forward(child)

        patch_forward(model)
        return model
        '''
    
    def _create_model_from_path(self, model_path: str, args: SFTConfig):
        """Creates a model from a path or model identifier."""
        model_init_kwargs = args.model_init_kwargs or {}
        # Handle torch dtype
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `SFTConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        if args.gradient_checkpointing:
            model_init_kwargs["use_cache"] = False

        # Create model
        # model = CPLSHForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        if args.sparse_attn == "local":
            model_init_kwargs["sliding_window"] = args.local
        model = RepeatedForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        return model
    
    @staticmethod
    def align_inputs(inputs):
        BLOCK_SIZE = 128
        seq_len = inputs["input_ids"].shape[1]
        if seq_len % BLOCK_SIZE == 0:
            return inputs
        # Align the input sequence length to the nearest multiple of 128
        aligned_seq_len = (seq_len // BLOCK_SIZE + 1) * BLOCK_SIZE
        bsz = inputs["input_ids"].shape[0]
        input_ids = F.pad(inputs["input_ids"], (0, aligned_seq_len - seq_len), value=0)
        attention_mask = F.pad(inputs["attention_mask"], (0, aligned_seq_len - seq_len), value=0)
        labels = F.pad(inputs["labels"], (0, aligned_seq_len - seq_len), value=-100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "eval" if self.control.should_evaluate else "train"
        inputs = self.align_inputs(inputs)
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                num_tokens_in_batch = (
                    self.accelerator.gather_for_metrics(torch.tensor(inputs["position_ids"].size(1))).sum().item()
                )
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if "labels" in inputs and not self.args.use_liger_kernel:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss
