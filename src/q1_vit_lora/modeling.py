import timm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model



def create_vit_small_for_cifar100() -> nn.Module:
    model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=100)
    return model



def freeze_all_but_head(model) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True



class LoRAViTWrapper(nn.Module):
    """Wrapper to make PEFT-wrapped timm ViT work correctly."""
    
    def __init__(self, peft_model):
        super().__init__()
        self.peft_model = peft_model
    
    def forward(self, x):
        # Call the base model directly to avoid PEFT's forward signature issues
        return self.peft_model.base_model.model(x)
    
    def named_parameters(self, *args, **kwargs):
        return self.peft_model.named_parameters(*args, **kwargs)
    
    def parameters(self, *args, **kwargs):
        return self.peft_model.parameters(*args, **kwargs)
    
    def state_dict(self, *args, **kwargs):
        return self.peft_model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        return self.peft_model.load_state_dict(*args, **kwargs)
    
    def train(self, mode=True):
        self.peft_model.train(mode)
        return self
    
    def eval(self):
        self.peft_model.eval()
        return self
    
    def to(self, *args, **kwargs):
        self.peft_model.to(*args, **kwargs)
        return self



def apply_lora_to_vit(model, rank: int, alpha: int, dropout: float):
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["qkv"],
        modules_to_save=["head"],
    )
    peft_model = get_peft_model(model, config)
    
    # Enable gradients for head
    for name, param in peft_model.named_parameters():
        if "head" in name or "lora_" in name:
            param.requires_grad = True
    
    return LoRAViTWrapper(peft_model)
