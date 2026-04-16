import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.layers")


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
        tokenized = self.tokenizer(prompts)
        if isinstance(tokenized, dict):
            input_ids = tokenized["input_ids"]
            attn_mask = tokenized.get("attention_mask", (input_ids != 0).long())
        else:
            input_ids = tokenized
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            pad_id = 0
            attn_mask = (input_ids != pad_id).long()
        self.register_buffer("ids", input_ids, persistent=False)      # [C, L]
        self.register_buffer("attn_mask", attn_mask, persistent=False)  # [C, L]
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
        # ctx: [C or 1, n_ctx, E]; prefix/suffix: [C,*,E]
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
        self.device = device  
        concept_arr = np.load(concept_feat_path, allow_pickle=True)

        if concept_arr.ndim == 2 and concept_arr.shape[1] >= 2:
            descriptions = concept_arr[:, 0].tolist()
            categories   = concept_arr[:, 1].tolist()
        else:
            descriptions = concept_arr.tolist()
            categories   = None
        self.raw_concepts = descriptions
        self.concept_categories = categories
        biomedclip_model, _ = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        biomedclip_model.float().eval().to(device)

        self.clip = biomedclip_model
        self.image_encoder = biomedclip_model.visual          
        self.logit_scale = biomedclip_model.logit_scale       
        self.dtype = biomedclip_model.text.transformer.dtype
        self.tokenizer = get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        if len(self.raw_concepts) > 0:
            with torch.no_grad():
                concept_tokens = self.tokenizer(self.raw_concepts)
                if isinstance(concept_tokens, dict):
                    concept_ids = concept_tokens["input_ids"]
                else:
                    concept_ids = concept_tokens
                    if concept_ids.dim() == 1:
                        concept_ids = concept_ids.unsqueeze(0)
                concept_ids = concept_ids.to(device)
                embed_concepts = biomedclip_model.encode_text(concept_ids)

            self.register_buffer("embed_concepts", embed_concepts.float())
            self.latent_dim = embed_concepts.shape[1]
        else:
            self.embed_concepts = None
            self.latent_dim = biomedclip_model.visual.output_dim
        self.concept_classifier = nn.Linear(len(self.raw_concepts), args.n_classes)
        self.classnames = getattr(args, "classnames", [f"class_{i}" for i in range(args.n_classes)])
        teacher_dim = getattr(args, "teacher_dim", 1024)
        self.teacher_proj = nn.Linear(teacher_dim, self.latent_dim)
        if (self.embed_concepts is not None) and (self.concept_categories is not None):
            all_cls_embeds = []
            cats = list(self.concept_categories)  
            for cname in self.classnames:
                idxs = [i for i, cat in enumerate(cats) if cat == cname]
                if len(idxs) == 0:
                    print(f"[FLAIRMultiLayer] WARNING: no concepts for class {cname} in concepts_raw.npy")
                    continue

                feats = self.embed_concepts[idxs]            # [N, D]
                feats = F.normalize(feats.float(), dim=-1)   
                N = feats.size(0)
                k = min(3, N)

                if N <= k:
                    selected = feats[:k]
                    if selected.size(0) < 3:
                        pad = selected[-1:].repeat(3 - selected.size(0), 1)
                        selected = torch.cat([selected, pad], dim=0)  # [3, D]
                else:
                    sel_idxs = kmeans_select_indices(feats, k=k)
                    selected = feats[sel_idxs]              # [k, D]
                    if selected.size(0) < 3:
                        pad = selected[-1:].repeat(3 - selected.size(0), 1)
                        selected = torch.cat([selected, pad], dim=0)  # [3, D]

                all_cls_embeds.append(selected.unsqueeze(0))  # [1,3,D]

            if len(all_cls_embeds) == len(self.classnames):
                teacher_text_embeds = torch.cat(all_cls_embeds, dim=0)  # [C,3,D]
                self.register_buffer("teacher_text_embeds", teacher_text_embeds)
                print("[FLAIRMultiLayer] teacher_text_embeds shape:", teacher_text_embeds.shape)
            else:
                print("[FLAIRMultiLayer] WARNING: teacher_text_embeds not built for all classes; KD logits will be disabled")
        else:
            print("[FLAIRMultiLayer] WARNING: embed_concepts or concept_categories is None; teacher_text_embeds disabled")
        class _BiomedTextWrapper(nn.Module):
            def __init__(self, clip_model, tokenizer):
                super().__init__()
                self.model = clip_model.text.transformer
                self.tokenizer = tokenizer  

        text_wrapper = _BiomedTextWrapper(self.clip, self.tokenizer)
        self.prompt_learner = PromptLearnerFL(
            cfg=args,
            classnames=self.classnames,
            flair_text_model=text_wrapper,
            flair_tokenizer=self.tokenizer,
            class_token_position=getattr(
                getattr(getattr(args, "TRAINER", None), "BIOMEDCOOP", None),
                "CLASS_TOKEN_POSITION",
                "end"
            )
        )

        self.text2latent = nn.Identity()
        if not hasattr(self, "logit_scale"):
            self.logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07), dtype=torch.float32))

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


    def forward(self, image):            
        logits, _, _ = self._student_logits_with_prompts(image)
        return logits

    def _encode_text_with_flair(self, prompts_embeds, attn_mask=None):
        

        C, L, _ = prompts_embeds.shape

        if attn_mask is not None:
            
            token_ids = attn_mask.to(dtype=torch.long, device=prompts_embeds.device)
        else:
            
            token_ids = torch.ones(C, L, dtype=torch.long, device=prompts_embeds.device)
        text_features = self.clip.encode_text(prompts_embeds, True, token_ids)

        return text_features

    def _student_logits_with_prompts(self, fundus_img: torch.Tensor):
        img_latent = self.image_encoder(fundus_img.type(self.dtype))  # [B, D]
        img_latent = img_latent.float()
        img_latent = F.normalize(img_latent, dim=-1)

        prompts, attn_mask = self.prompt_learner()                     # [C, L, emb_dim]
        text_feat = self._encode_text_with_flair(prompts, attn_mask)  # [C, D]
        text_feat = self.text2latent(text_feat)
        text_feat = F.normalize(text_feat, dim=-1)

        logit_scale = self.logit_scale.exp().to(img_latent.device)
        logits = logit_scale * (img_latent @ text_feat.t())           # [B, C]

        return logits, img_latent, text_feat

    def _fixed_text_prototypes(self):
        
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
            
            if getattr(self, "concept_categories", None) is None:
                center = self.embed_concepts.mean(dim=0, keepdim=True)    # [1, D]
                center = F.normalize(center, dim=-1)
                
                centers = center.expand(len(self.classnames), -1)
                return centers

            centers = []
            cats = list(self.concept_categories)  

            for cname in self.classnames:
                
                idxs = [i for i, cat in enumerate(cats) if cat == cname]
                if len(idxs) == 0:
                    
                    center_i = self.embed_concepts.mean(dim=0, keepdim=True)  # [1, D]
                else:
                    center_i = self.embed_concepts[idxs].mean(dim=0, keepdim=True)  # [1, D]

                center_i = F.normalize(center_i, dim=-1)
                centers.append(center_i)

            centers = torch.cat(centers, dim=0)   # [C, D]
            return centers

    def forward_train(
        self,
        fundus_img: torch.Tensor,
        label: torch.Tensor,
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

    
        concept_center = self._concept_center_proto().to(text_features.device).float()  # [C, D]

        cos = (text_features * concept_center).sum(dim=1)   # [C]
        loss_sccm = (1.0 - cos).mean() * float(sccm_lambda)

        if (teacher is not None) and (getattr(self, "teacher_text_embeds", None) is not None):
            with torch.no_grad():
            
                if hasattr(teacher, "forward_features"):
                    t_feat = teacher.forward_features(fundus_img.type(self.dtype))
                else:
                    t_feat = teacher(fundus_img.type(self.dtype))

                if t_feat.ndim == 3:
                    t_feat = t_feat.squeeze(1)
                t_feat = t_feat.float()
                t_feat = F.normalize(t_feat, dim=-1)        # [B, D_t]

            
                t_proj = self.teacher_proj(t_feat)          # [B, latent_dim]
                t_proj = F.normalize(t_proj, dim=-1)

                # teacher_text_embeds: [C, 3, D]
                text_proto = self.teacher_text_embeds.to(t_proj.device)
                C, K, D = text_proto.shape
                text_flat = text_proto.view(C * K, D)       # [C*K, D]

            
                sim = torch.matmul(t_proj, text_flat.t())   # [B, C*K]
                sim = sim.view(t_proj.size(0), C, K)        # [B, C, 3]
                teacher_logits = sim.mean(dim=-1)           

            
                logit_scale = self.logit_scale.exp().to(t_proj.device)
                teacher_logits = logit_scale * teacher_logits

            T = float(temperature)
            loss_kd = F.kl_div(
                F.log_softmax(logits / T, dim=1),
                F.log_softmax(teacher_logits / T, dim=1),
                reduction='batchmean',
                log_target=True
            )
            loss_kdsp = loss_kd * (T * T) * float(kdsp_lambda)
        else:
        
            loss_kdsp = torch.tensor(0.0, device=dev)

        return logits, loss_ce, loss_sccm, loss_kdsp


    def get_concepts_feat(self):  
        return self.embed_concepts
    
    

    
   

    