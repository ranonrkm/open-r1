from types import MethodType
import torch
from trl import SFTTrainer
from ..configs import SFTConfig
from .sparse_attention_utils import sparse_attn_forward

class SparseSFTTrainer(SFTTrainer):
    def _create_model_from_path(self, model_path: str, args: SFTConfig):
        model = super()._create_model_from_path(model_path, args)

        def patch_forward(module: torch.nn.Module) -> None:
            for name, child in module.named_children():
                if "self_attn" in name:
                    child.forward = MethodType(sparse_attn_forward, child)
                    child.sink = args.sink
                    child.local = args.local
                    child.topk = args.sparse_attn_topk
                else:
                    patch_forward(child)

        patch_forward(model)
        return model