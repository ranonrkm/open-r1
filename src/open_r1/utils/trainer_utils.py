from types import MethodType
import torch
from trl import SFTTrainer
from ..configs import SFTConfig
from .sparse_attention_utils import local_attn_forward

class SparseSFTTrainer(SFTTrainer):
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
