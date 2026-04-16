import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flair import FLAIRModel
from typing import Optional


def kmeans_select_indices(emb: torch.Tensor, k: int = 3, iters: int = 10):
    
    N, D = emb.shape
    k = min(k, N)
    if k <= 0:
        return []

    centers = emb[:k].clone()  # [k, D]

    for _ in range(iters):
        # [N, k]
        dist2 = ((emb.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
        labels = dist2.argmin(dim=1)  # [N]
        new_centers = []
        for j in range(k):
            mask = (labels == j)
            if mask.sum() == 0:
                new_centers.append(centers[j:j+1])
            else:
                new_centers.append(emb[mask].mean(dim=0, keepdim=True))
        new_centers = torch.cat(new_centers, dim=0)
        if torch.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    dist2 = ((emb.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)  # [N, k]
    selected = []
    for j in range(k):
        idx = int(dist2[:, j].argmin())
        if idx not in selected:
            selected.append(idx)

    if len(selected) < k:
        remain = [i for i in range(N) if i not in selected]
        selected.extend(remain[: k - len(selected)])

    return selected[:k]

class PromptLearnerFL(nn.Module):
    
    def __init__(self, cfg, classnames, flair_text_model, flair_tokenizer, class_token_position="end"):
        super().__init__()

        self.classnames = [name.replace("_", " ") for name in classnames]
        self.n_cls = len(self.classnames)
        trainer_cfg = getattr(cfg, "TRAINER", None)
        biomed_cfg = getattr(trainer_cfg, "BIOMEDCOOP", None) if trainer_cfg is not None else None
        base_n_ctx = getattr(biomed_cfg, "N_CTX", 4) if biomed_cfg is not None else 4
        self.CSC = getattr(biomed_cfg, "CSC", False) if biomed_cfg is not None else False
        self.class_token_position = class_token_position
        use_atp  = getattr(cfg, "use_atprompt", False)
        att_num  = getattr(cfg, "att_num", 0)
        n_att1   = getattr(cfg, "n_att1", 0)
        n_att2   = getattr(cfg, "n_att2", 0)
        att1_txt = getattr(cfg, "att1_text", "")
        att2_txt = getattr(cfg, "att2_text", "")
        self.use_atp   = bool(use_atp)
        self.atp_num   = int(att_num)
        self.n_att1    = int(n_att1) if (self.use_atp and self.atp_num >= 1) else 0
        self.n_att2    = int(n_att2) if (self.use_atp and self.atp_num >= 2) else 0
        self.att1_text = att1_txt
        self.att2_text = att2_txt
        self.base_n_ctx = int(base_n_ctx)
        self.prompt_prefix = " ".join(["X"] * self.base_n_ctx)

        
        self.n_ctx = self.base_n_ctx + self.n_att1 + self.n_att2

        print(f"[PromptLearnerFL] base_n_ctx={self.base_n_ctx}, "
              f"n_att1={self.n_att1}, n_att2={self.n_att2}, "
              f"total n_ctx={self.n_ctx}, use_atp={self.use_atp}, att_num={self.atp_num}")

        self.text_model = flair_text_model
        self.tokenizer = flair_tokenizer

        base = flair_text_model.model  
        emb = base.get_input_embeddings()
        if emb is None:
            raise TypeError("text_model.model can not get_input_embeddings()")
        emb_dim = emb.weight.shape[1]

        ctx_device = emb.weight.device
        if self.CSC:
            ctx = torch.empty(self.n_cls, self.n_ctx, emb_dim, device=ctx_device)
        else:
            ctx = torch.empty(self.n_ctx, emb_dim, device=ctx_device)
        nn.init.normal_(ctx, std=0.02)
        self.ctx = nn.Parameter(ctx)

        if self.use_atp:
            prefix1 = " ".join(["X"] * self.n_att1) if self.n_att1 > 0 else ""
            prefix2 = " ".join(["X"] * self.n_att2) if self.n_att2 > 0 else ""

            prompts = []
            for name in self.classnames:
                parts = []
                if self.atp_num >= 1 and self.att1_text != "":
                    if prefix1:
                        parts.append(prefix1)
                    parts.append(self.att1_text)
                if self.atp_num >= 2 and self.att2_text != "":
                    if prefix2:
                        parts.append(prefix2)
                    parts.append(self.att2_text)
                if self.prompt_prefix:
                    parts.append(self.prompt_prefix)
                parts.append(name)
                prompts.append(" ".join(parts) + ".")
        else:
            prompts = [f"{self.prompt_prefix} {name}." for name in self.classnames]

        print("[PromptLearnerFL] example prompt[0]:", prompts[0])
        tokenized = self.tokenizer(prompts, padding=True, return_tensors="pt")
        self.register_buffer("ids", tokenized["input_ids"], persistent=False)
        self.register_buffer("attn_mask", tokenized["attention_mask"], persistent=False)
        with torch.no_grad():
            dev = emb.weight.device
            emb_full = emb(self.ids.to(dev))  # [C, L, E]
        self.register_buffer("token_prefix", emb_full[:, :1, :], persistent=False)  # [C,1,E]
        self.register_buffer("token_suffix", emb_full[:, 1:, :], persistent=False)  # [C,L-1,E]

        self.name_lens = []
        for i in range(self.n_cls):

            valid = int(self.attn_mask[i].sum().item()) - 1
            self.name_lens.append(max(valid, 1))

    def construct_prompts(self, ctx, prefix, suffix):
        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [C,n_ctx,E]

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
            return prompts
        elif self.class_token_position in ["middle", "front"]:
            half = self.n_ctx // 2
            out = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]
                class_i  = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                if self.class_token_position == "middle":
                    ctx1, ctx2 = ctx[i:i+1, :half, :], ctx[i:i+1, half:, :]
                    prompt = torch.cat([prefix_i, ctx1, class_i, ctx2, suffix_i], dim=1)
                else:  # front
                    prompt = torch.cat([prefix_i, class_i, ctx[i:i+1, :, :], suffix_i], dim=1)
                out.append(prompt)
            return torch.cat(out, dim=0)
        else:
            raise ValueError("class_token_position must be end/middle/front")

    def forward(self):
        # embeddings
        prompts = self.construct_prompts(self.ctx, self.token_prefix, self.token_suffix)
        prompts = prompts.to(self.ctx.dtype).to(self.ctx.device)

        C = self.ids.shape[0]
        device = prompts.device
        dtype = self.attn_mask.dtype

        base_mask = self.attn_mask.to(device)

        if self.class_token_position == "end":
            # [prefix(1) | ctx(n_ctx) | suffix(L-1)]
            prefix_m = base_mask[:, :1]                                 # [C,1]
            ctx_m = torch.ones(C, self.n_ctx, dtype=dtype, device=device)
            suffix_m = base_mask[:, 1:]                                 # [C,L-1]
            attn = torch.cat([prefix_m, ctx_m, suffix_m], dim=1)        # [C, 1+n_ctx+(L-1)]

        elif self.class_token_position in ["middle", "front"]:
            parts = []
            half = self.n_ctx // 2
            for i in range(C):
                name_len = self.name_lens[i]                          
                prefix_i = base_mask[i:i+1, :1]                         # [1,1]
                class_i  = base_mask[i:i+1, 1:1+name_len]               # [1,name_len]
                pad_tail = base_mask[i:i+1, 1+name_len:]                # [1, L-1-name_len]

                if self.class_token_position == "middle":
                    ctx1 = torch.ones(1, half,    dtype=dtype, device=device)
                    ctx2 = torch.ones(1, self.n_ctx - half, dtype=dtype, device=device)
                    attn_i = torch.cat([prefix_i, ctx1, class_i, ctx2, pad_tail], dim=1)
                else:  # front
                    ctx_all = torch.ones(1, self.n_ctx, dtype=dtype, device=device)
                    attn_i = torch.cat([prefix_i, class_i, ctx_all, pad_tail], dim=1)
                parts.append(attn_i)
            attn = torch.cat(parts, dim=0)                              # [C, 1+n_ctx+(L-1)]
        else:
            raise ValueError("class_token_position must be end/middle/front")
        return prompts, attn.to(dtype=self.attn_mask.dtype, device=prompts.device)



class FLAIRMultiLayer(nn.Module):
    def __init__(self, args, device, concept_feat_path, modality='fundus', class_weights=None):
        super().__init__()
        self.modality = modality        
        concept_arr = np.load(concept_feat_path, allow_pickle=True)

        if concept_arr.ndim == 2 and concept_arr.shape[1] >= 2:
            descriptions = concept_arr[:, 0].tolist()
            categories   = concept_arr[:, 1].tolist()
        else:
            descriptions = concept_arr.tolist()
            categories   = None

        self.raw_concepts = descriptions
        self.concept_categories = categories
        self.device = device
        self.flair_model = FLAIRModel(device=device, from_checkpoint=True)
        self.latent_dim = self.flair_model.vision_model.proj_dim

        teacher_dim = getattr(args, "teacher_dim", 1024)
        self.teacher_proj = nn.Linear(teacher_dim, self.latent_dim)
        self.concept_classifier = nn.Linear(len(self.raw_concepts), args.n_classes)

        with torch.no_grad():
            text_input_ids, text_attention_mask = self.flair_model.preprocess_text(self.raw_concepts)
            self.register_buffer(
                "embed_concepts",
                self.flair_model.text_model(text_input_ids, text_attention_mask).float()
            )
        self.text_model = self.flair_model.text_model
        self.text_base = self.text_model.model                     
        self.text_proj = self.text_model.projection_head_text       
        self.tokenizer = self.text_model.tokenizer                  
        tok = getattr(self.flair_model, "tokenizer", None)
        if tok is None:
            def _tok(texts, padding=True, return_tensors="pt"):
                ids, attn = self.flair_model.preprocess_text(texts)
                return {"input_ids": ids, "attention_mask": attn}
            self.tokenizer = _tok
        else:
            self.tokenizer = tok

        self.classnames = getattr(args, "classnames", [f"class_{i}" for i in range(args.n_classes)])

        if (getattr(self, "concept_categories", None) is not None) and (getattr(self, "embed_concepts", None) is not None):
            all_cls_embeds = []
            cats = list(self.concept_categories)

            for cname in self.classnames:
                idxs = [i for i, cat in enumerate(cats) if cat == cname]
                if len(idxs) == 0:
                    print(f"[FLAIRMultiLayer] WARNING: no concepts for class {cname} in concepts_raw.npy")
                    feats = self.embed_concepts.mean(dim=0, keepdim=True).repeat(3, 1)
                else:
                    feats = self.embed_concepts[idxs]

                feats = F.normalize(feats.float(), dim=-1)
                N = feats.size(0)
                k = min(3, N)

                if N <= k:
                    selected = feats[:k]
                    if selected.size(0) < 3:
                        pad = selected[-1:].repeat(3 - selected.size(0), 1)
                        selected = torch.cat([selected, pad], dim=0)
                else:
                    sel_idxs = kmeans_select_indices(feats, k=k)
                    selected = feats[sel_idxs]
                    if selected.size(0) < 3:
                        pad = selected[-1:].repeat(3 - selected.size(0), 1)
                        selected = torch.cat([selected, pad], dim=0)

                all_cls_embeds.append(selected.unsqueeze(0))

            if len(all_cls_embeds) == len(self.classnames):
                teacher_text_embeds = torch.cat(all_cls_embeds, dim=0)
                self.register_buffer("teacher_text_embeds", teacher_text_embeds)
                print("[FLAIRMultiLayer] teacher_text_embeds shape:", teacher_text_embeds.shape)
            else:
                print("[FLAIRMultiLayer] WARNING: teacher_text_embeds not built for all classes; KD logits will be disabled")
        else:
            print("[FLAIRMultiLayer] WARNING: concept_categories is None; teacher_text_embeds disabled (KD will fallback to logits-teacher only)")

        self.prompt_learner = PromptLearnerFL(
            cfg=args,
            classnames=self.classnames,
            flair_text_model=self.text_model,
            flair_tokenizer=self.tokenizer,
            class_token_position=getattr(getattr(getattr(args, "TRAINER", None), "BIOMEDCOOP", None), "CLASS_TOKEN_POSITION", "end")
        )

        self.text_dim = getattr(self.text_proj, "out_features", self.latent_dim)
        self.text2latent = nn.Identity()
        if not hasattr(self, "logit_scale"):
            self.logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07), dtype=torch.float32))
        self.project_4 = nn.Identity()
        if class_weights is not None:
            class_weights = class_weights.to(device)
            self.register_buffer("base_class_weights", class_weights)
            self.ce_weight_factor = 1.0
            print("[FLAIRMultiLayer] base_class_weights for CE:", class_weights)
        else:
            self.base_class_weights = None
            self.ce_weight_factor = 0.0

    def set_ce_weight_factor(self, factor: float):
        factor = float(factor)
        factor = max(0.0, min(1.0, factor))
        self.ce_weight_factor = factor
    @torch.no_grad()
    def _teacher_logits_from_encoder(self, teacher, img: torch.Tensor):
        if getattr(self, "teacher_text_embeds", None) is None:
            raise RuntimeError("teacher_text_embeds is None; please provide concept_categories to build it")

        if hasattr(teacher, "forward_features"):
            t_feat = teacher.forward_features(img)
        else:
            t_feat = teacher(img)

        if isinstance(t_feat, (tuple, list)):
            t_feat = t_feat[0]
        if t_feat.ndim == 3:
            t_feat = t_feat.squeeze(1)

        t_feat = t_feat.float()
        t_feat = F.normalize(t_feat, dim=-1)

        if t_feat.shape[1] == self.teacher_proj.in_features:
            t_proj = self.teacher_proj(t_feat)
        elif t_feat.shape[1] == self.latent_dim:
            t_proj = t_feat
        else:
            raise ValueError(
                f"Teacher feature dim {t_feat.shape[1]} mismatches teacher_dim={self.teacher_proj.in_features} "
                f"and latent_dim={self.latent_dim}. Set args.teacher_dim accordingly."
            )

        t_proj = F.normalize(t_proj, dim=-1)

        text_proto = self.teacher_text_embeds.to(t_proj.device)   # [C,K,D]
        C, K, D = text_proto.shape
        sim = torch.matmul(t_proj, text_proto.view(C*K, D).t())   # [B, C*K]
        sim = sim.view(t_proj.size(0), C, K).mean(dim=-1)         # [B, C]

        logit_scale = self.logit_scale.exp().to(t_proj.device)
        return logit_scale * sim



    def forward(self, image):            
        logits, _, _ = self._student_logits_with_prompts(image)
        return logits

    
    def _encode_text_with_flair(self, prompts_embeds, attn_mask=None):
        
       
        outputs = self.text_base(inputs_embeds=prompts_embeds, attention_mask=attn_mask)
        hs = outputs["hidden_states"]  

        
        last_hidden_states = torch.stack([hs[1], hs[2], hs[-1]])          # [3, B, L, H]
        feat = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)     # [B, H]

        
        feat = self.text_proj(feat)                                      
        return feat

    def _student_logits_with_prompts(self, fundus_img: torch.Tensor):
        
        img_latent = self.flair_model.vision_model(fundus_img).float()      # [B, latent_dim]
        img_latent = F.normalize(img_latent, dim=-1)
        prompts, attn_mask = self.prompt_learner()

        text_feat = self._encode_text_with_flair(prompts, attn_mask)        # [C, hidden or emb_dim]
        text_feat = self.text2latent(text_feat)                              # [C, latent_dim]
        text_feat = F.normalize(text_feat, dim=-1)

        logit_scale = self.logit_scale.exp().to(img_latent.device)
        logits = logit_scale * (img_latent @ text_feat.t())                  # [B, C]
        return logits, img_latent, text_feat

    def _fixed_text_prototypes(self):
        """
        用“零样本”的类名文本（不带可学习 ctx）走同一套文本编码器+投影头，
        得到每类的固定文本原型 [C, latent_dim]，与训练状态无关、与分类器解耦。
        """
        prefix = self.prompt_learner.token_prefix    # [C, 1, E]
        suffix = self.prompt_learner.token_suffix    # [C, L-1, E]
        prompts = torch.cat([prefix, suffix], dim=1) # [C, L, E]

        attn_mask = self.prompt_learner.attn_mask.to(prompts.device)   # [C, L]

        with torch.no_grad():
            feat = self._encode_text_with_flair(prompts, attn_mask)    # [C, H] -> proj -> [C, D]
            feat = self.text2latent(feat)                              
            feat = F.normalize(feat, dim=-1)                            # [C, latent_dim]
        return feat
    
    def _concept_center_proto(self):

        with torch.no_grad():
            C = len(self.classnames)
            if getattr(self, "concept_categories", None) is None:
                center = self.embed_concepts.mean(dim=0, keepdim=True)     # [1,D]
                center = F.normalize(center, dim=-1)
                return center.expand(C, -1)

            centers = []
            cats = list(self.concept_categories)
            for cname in self.classnames:
                idxs = [i for i, cat in enumerate(cats) if cat == cname]
                if len(idxs) == 0:
                    center_i = self.embed_concepts.mean(dim=0, keepdim=True)
                else:
                    center_i = self.embed_concepts[idxs].mean(dim=0, keepdim=True)
                centers.append(F.normalize(center_i, dim=-1))
            return torch.cat(centers, dim=0)  # [C,D]

    def forward_train(
        self,
        fundus_img: torch.Tensor,
        label: torch.Tensor,
        tea_img: Optional[torch.Tensor] = None,
        teacher: Optional["FLAIRMultiLayer"] = None,
        tau: float = 0.5,
        sccm_lambda: float = 1.0,
        kdsp_lambda: float = 1.0,
        temperature: float = 1.0,
    ):
        dev = fundus_img.device
        logits, image_features, text_features = self._student_logits_with_prompts(fundus_img)
        base_w = getattr(self, "base_class_weights", None)

        if base_w is not None:
            alpha = float(getattr(self, "ce_weight_factor", 1.0))  
            if alpha >= 1.0:
                cur_w = base_w
            elif alpha <= 0.0:
                cur_w = None
            else:
                cur_w = 1.0 + alpha * (base_w - 1.0)

            if cur_w is None:
                loss_ce = F.cross_entropy(logits, label)
            else:
                loss_ce = F.cross_entropy(
                    logits, label,
                    weight=cur_w.to(logits.device, dtype=torch.float32)
                )
        else:
            loss_ce = F.cross_entropy(logits, label)

        concept_center = self._concept_center_proto().to(text_features.device).float()
        cos = (text_features * concept_center).sum(dim=1)  # [C]
        loss_sccm = (1.0 - cos).mean() * float(sccm_lambda)

        if (teacher is not None) and (tea_img is not None) and hasattr(teacher, "_student_logits_with_prompts"):
            with torch.no_grad():
                target_logits, _, _ = teacher._student_logits_with_prompts(tea_img.to(dev))

            T = float(temperature)
            loss_kd = F.kl_div(
                F.log_softmax(logits / T, dim=1),
                F.log_softmax(target_logits / T, dim=1),
                reduction="batchmean",
                log_target=True
            )
            loss_kdsp = loss_kd * (T * T) * float(kdsp_lambda)

        elif (teacher is not None) and (getattr(self, "teacher_text_embeds", None) is not None):
            with torch.no_grad():
                t_img = tea_img.to(dev) if (tea_img is not None) else fundus_img
                target_logits = self._teacher_logits_from_encoder(teacher, t_img)

            T = float(temperature)
            loss_kd = F.kl_div(
                F.log_softmax(logits / T, dim=1),
                F.log_softmax(target_logits / T, dim=1),
                reduction="batchmean",
                log_target=True
            )
            loss_kdsp = loss_kd * (T * T) * float(kdsp_lambda)

        else:
            loss_kdsp = torch.tensor(0.0, device=dev)

        return logits, loss_ce, loss_sccm, loss_kdsp
        
    def get_concepts_feat(self):  
        return self.embed_concepts
    
    

    
   

    