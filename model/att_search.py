
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from typing import List, Tuple
from types import MethodType

class AttributePromptSearcher(nn.Module):

    def __init__(self, model,
                 attr_words: List[str],
                 max_k: int = 2):
        super().__init__()
        assert len(attr_words) > 0, ""

        self.attr_words = list(dict.fromkeys(attr_words))  
        self.max_k = int(max_k)
        self.classnames = [c.replace("_", " ") for c in model.classnames]  
        combos: List[Tuple[str, ...]] = []
        for k in range(1, self.max_k + 1):
            combos.extend(list(combinations(self.attr_words, k)))
        self.attr_combinations = combos  
        self.attr_inputs = []  
        tok = model.tokenizer
        for comb in self.attr_combinations:
            texts = [("{} {}.".format(" ".join(comb), name)) for name in self.classnames]
            tokenized = tok(texts, padding=True, return_tensors="pt")
            self.attr_inputs.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"]
            })

        self.att_weight = nn.Parameter(torch.zeros(len(self.attr_combinations), dtype=torch.float32))

    @torch.no_grad()
    def _ids_to_embeds(self, model ,ids: torch.Tensor) -> torch.Tensor:

        emb_tab = model.text_base.get_input_embeddings()
        return emb_tab(ids.to(emb_tab.weight.device))

    def _text_feat_for_comb(self, model ,comb_idx: int) -> torch.Tensor:
        
        item = self.attr_inputs[comb_idx]
        ids = item["input_ids"]              # [C, L]
        attn = item["attention_mask"]        # [C, L]

        word_emb = self._ids_to_embeds(model, ids)  # [C,L,E]

        prefix = word_emb[:, :1, :]
        suffix = word_emb[:, 1:, :]
        ctx = model.prompt_learner.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(len(self.classnames), -1, -1)

        prompts = torch.cat([prefix, ctx.to(prefix.dtype).to(prefix.device), suffix.to(prefix.device)], dim=1)

        dev = prompts.device  
        ctx_ones = torch.ones(attn.shape[0], ctx.shape[1], device=dev, dtype=attn.dtype)
        attn_full = torch.cat([attn[:, :1].to(dev), ctx_ones, attn[:, 1:].to(dev)], dim=1)

        feat = model._encode_text_with_flair(prompts, attn_full)
        feat = model.text2latent(feat)
        feat = F.normalize(feat, dim=-1)
        return feat  # [C,D]

    def fused_text_feat(self,model) -> torch.Tensor:
    
        feats = [self._text_feat_for_comb(model, i) for i in range(len(self.attr_combinations))]  # K*[C,D]
        stacked = torch.stack(feats, dim=0)    # [K,C,D]
        w = F.softmax(self.att_weight, dim=0)  # [K]
        fused = (w.view(-1,1,1) * stacked).sum(dim=0)  # [C,D]
        return F.normalize(fused, dim=-1)

def _make_patched_student_logits(searcher: AttributePromptSearcher):
    
    def _patched(self, fundus_img: torch.Tensor):
        img_latent = self.flair_model.vision_model(fundus_img).float()
        img_latent = F.normalize(img_latent, dim=-1)
        text_feat = searcher.fused_text_feat(self)   
        logit_scale = self.logit_scale.exp().to(img_latent.device)
        logits = logit_scale * (img_latent @ text_feat.t())                 # [B,C]
        return logits, img_latent, text_feat
    return _patched

def install_attribute_prompt_search(model,
                                    attr_words: List[str],
                                    max_k: int = 2) -> AttributePromptSearcher:
    
    searcher = AttributePromptSearcher(model, attr_words=attr_words, max_k=max_k)
    model.attribute_searcher = searcher

    model._student_logits_with_prompts = MethodType(_make_patched_student_logits(searcher), model)
    return searcher
