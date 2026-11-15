"""Geiru Keystone Attention.

Every token's attention is one pie that adds up to 100%: prompt, reference pictures, audio, the new frames,
the carried context (held frames from the last clip) and the keystone (clip 1's clean frame, grafted by
Geiru Keystone). This node leaves every slice exactly the size H3 would make it, except one: the slice
shared by the keystone and the carried context. Inside that slice only, `balance` moves attention from the
carried context to the keystone (balance > 1) or back (balance < 1). Prompt, references, audio and the new
frames keep exactly what they would normally get.

How it works (per attention call, per token, per head, exact):
- One normal attention pass, with two extra value channels that read out the keystone's share (pK) and the
  context's share (pC) of each token's attention.
- Two small passes over only the keystone keys and only the context keys give what each part contributes.
- Inside the shared slice S = pK + pC, the keystone gets S * b*pK / (b*pK + pC) instead of pK; the context
  gets the rest of S. The output is corrected by that shift. Everything outside S is untouched.

Needs Geiru Keystone in front (its av_latent goes into `latent`). Follows Geiru Context's
degradation_protection switch: off = pass through, like Keystone / Late Lock / Context Noise.
"""

import logging
import re
import types

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.patcher_extension
import comfy.quant_ops

WHO = "Geiru Keystone Attention"
KEY = "geiru_keystone_balance"

# The blocks found to carry brightness on H3 (block search, Nov 2026): balance on these 13 holds a chain's brightness.
LIGHTING_BLOCKS = "31-33, 36-44, 47"
BLOCK_MODES = ["lighting blocks", "custom blocks"]


def parse_blocks(text):
    """'31-33, 36-44, 47' or '31,32,33' or '31 - 33 ,47' -> sorted list of block numbers. Spaces don't matter."""
    text = re.sub(r"\s*-\s*", "-", str(text or ""))
    out = set()
    for part in re.split(r"[,;\s]+", text):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.update(range(min(a, b), max(a, b) + 1))
        elif part.isdigit():
            out.add(int(part))
        else:
            raise ValueError(f"[{WHO}] can't read '{part}' in blocks. Use numbers and ranges, like 31-33, 36-44, 47")
    return sorted(out)


def _ranges(blocks):
    out, i = [], 0
    while i < len(blocks):
        j = i
        while j + 1 < len(blocks) and blocks[j + 1] == blocks[j] + 1:
            j += 1
        out.append(str(blocks[i]) if i == j else f"{blocks[i]}-{blocks[j]}")
        i = j + 1
    return ", ".join(out)


def _held_video_latents(denoise_mask, latent_t):
    """bool [latent_t, H, W] (latent grid) - True where the video latent is held, or None."""
    if denoise_mask is None:
        return None
    m = denoise_mask[0, 0].to(torch.float32)
    if m.dim() != 3:
        return None
    if m.shape[0] == latent_t - 1:            # a back latent tacked on by Geiru Keystone's model patch: held
        m = torch.cat([m, torch.zeros_like(m[:1])], dim=0)
    if m.shape[0] != latent_t:
        return None
    return m < 0.5


class GeiruKeystoneAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "The model, after your other model patches (right before the guider)."}),
            "latent": ("LATENT", {"tooltip": "Geiru Keystone's av_latent output. Tells this node which latents are "
                                             "the keystone (and capstone) and whether degradation protection is on. "
                                             "Only read: the latent still goes from Keystone to the sampler as before."}),
            "balance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100000.0, "step": 0.01, "round": 0.01,
                "tooltip": "How the slice shared by the keystone and the carried context is split. 1 = normal. "
                           "2 = the keystone pulls twice as hard compared with the context, and the context gives up "
                           "exactly what the keystone gains. Every other slice (prompt, references, audio, new "
                           "frames) stays exactly as it would be."}),
            "block_mode": (BLOCK_MODES, {"default": "lighting blocks",
                "tooltip": "lighting blocks = the 13 blocks found to carry brightness (31-33, 36-44, 47). "
                           "custom blocks = type your own below."}),
            "blocks": ("STRING", {"default": LIGHTING_BLOCKS, "multiline": False,
                "tooltip": "Which transformer blocks (0-49) to apply it in, for custom blocks. Ranges and single "
                           "numbers, commas between: 31-33, 36-44, 47  or  31,32,33. Spaces don't matter."}),
            "enabled": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Geiru Context Suite"
    DESCRIPTION = ("Moves attention between the keystone and the carried context only, keeping every other part of "
                   "H3's attention exactly the same. Plug Geiru Keystone's av_latent into latent. Follows the "
                   "degradation_protection switch on Geiru Context. Works with any attention setup (plain, Kijai's low "
                   "VRAM or Sage patches): on its own blocks it does the balanced attention itself, only where the balance "
                   "applies, and leaves everything else to whatever attention is set.")

    def patch(self, model, latent, balance, block_mode, blocks, enabled, **_old_settings):
        # **_old_settings: start_step / end_step / take_from from older workflows are ignored
        if not enabled or abs(balance - 1.0) < 1e-6:
            return (model,)
        if isinstance(latent, dict) and latent.get("geiru_protection", True) is False:
            print(f"[{WHO}] degradation protection off - passing through")
            return (model,)
        slots = sorted(set(int(s) for s in (latent.get("clown_keyframe_slots") or []))) if isinstance(latent, dict) else []
        has_back = isinstance(latent, dict) and latent.get("clown_keyframe_back") is not None
        if not slots and not has_back:
            print(f"[{WHO}] no keystone on this shot (first shot, Keystone off or not in front) - passing through")
            return (model,)

        chosen = parse_blocks(LIGHTING_BLOCKS if block_mode == "lighting blocks" else blocks)
        if not chosen:
            print(f"[{WHO}] no blocks given - passing through")
            return (model,)
        block_set = frozenset(chosen)
        print(f"[{WHO}] balance x{float(balance):g} on {len(chosen)} blocks ({block_mode}): {_ranges(chosen)}")

        b = float(balance)
        stats = {"logged": False}
        m = model.clone()

        # 1) wrapper: per model call, find the keystone rows and the context rows (video rows only)
        def wrapper(executor, x, timestep, context, transformer_options={}, *args, **kwargs):
            state = {"key": None, "ctx": None, "cache": {}, "on": True}
            try:
                v = x[0] if isinstance(x, (list, tuple)) else x
                lt, lh, lw = v.shape[2], v.shape[3] + (v.shape[3] % 2), v.shape[4] + (v.shape[4] % 2)
                held = _held_video_latents(kwargs.get("denoise_mask"), lt)
                if held is not None:
                    held = F.pad(held.float(), (0, lw - held.shape[-1], 0, lh - held.shape[-2]), mode="replicate") > 0.5
                    held = held.reshape(lt, lh // 2, 2, lw // 2, 2).all(dim=4).all(dim=2)   # [lt, rows_h, rows_w]
                    anchor = torch.zeros(lt, dtype=torch.bool)
                    for s in slots:
                        if 0 <= s < lt:
                            anchor[s] = True
                    if has_back and held.shape[0] >= 1:
                        anchor[-1] = True
                    anchor &= held.reshape(lt, -1).all(dim=1).cpu()   # only fully held latents count as anchors
                    rows = held.reshape(lt, -1).cpu()
                    key = (anchor[:, None] & rows).reshape(-1)
                    ctx = (~anchor[:, None] & rows).reshape(-1)
                    if bool(key.any()) and bool(ctx.any()):
                        state["key"], state["ctx"] = key, ctx
            except Exception as exc:
                logging.warning("[%s] could not read the noise mask: %s", WHO, exc)
            transformer_options[KEY] = state
            return executor(x, timestep, context, transformer_options, *args, **kwargs)

        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEY, wrapper)

        def indices(topts, seq_len, device):
            state = topts.get(KEY)
            layout = topts.get("minimax_h3_layout")
            if not state or not state.get("on", True) or state["key"] is None or layout is None or getattr(layout, "seq_len", None) != seq_len:
                return None
            ck = (id(layout), str(device))
            if ck in state["cache"]:
                return state["cache"][ck]
            res = None
            for a, z, kind in layout.segments:
                if kind == "video" and state["key"].numel() == z - a:
                    ki = torch.nonzero(state["key"]).flatten() + a
                    ci = torch.nonzero(state["ctx"]).flatten() + a
                    res = (ki.to(device), ci.to(device))
                    break
            state["cache"][ck] = res
            return res

        # 2) the balanced attention, for one head group: q, k, v are [1, h, T, D]
        def balanced(qb, kb, vb, ki, ci, blk):
            d = qb.shape[-1]
            scale = d ** -0.5
            B, H, T, _ = qb.shape
            S_len = kb.shape[2]
            # full pass, two extra value channels = keystone share and context share
            ind = vb.new_zeros(S_len, 2)
            ind[ki, 0] = 1.0
            ind[ci, 1] = 1.0
            pad = (-(d + 2)) % 8
            v2 = torch.cat([vb, ind.view(1, 1, S_len, 2).expand(B, H, S_len, 2), vb.new_zeros(B, H, S_len, pad)], dim=-1)
            full = F.scaled_dot_product_attention(qb, kb, v2, scale=scale)
            out = full[..., :d]
            pK = full[..., d:d + 1].float()
            pC = full[..., d + 1:d + 2].float()
            del full, v2
            # what the keystone part and the context part each contribute (attention over only those keys)
            oK = F.scaled_dot_product_attention(qb, kb.index_select(2, ki), vb.index_select(2, ki), scale=scale)
            oC = F.scaled_dot_product_attention(qb, kb.index_select(2, ci), vb.index_select(2, ci), scale=scale)
            sh = pK + pC
            denom = b * pK + pC
            new_pK = torch.where(denom > 1e-12, sh * b * pK / denom.clamp_min(1e-12), pK)
            delta = new_pK - pK                                         # keystone gains delta, context loses delta
            out = (out.float() + delta * (oK.float() - oC.float())).to(out.dtype)
            if not stats["logged"]:
                stats["logged"] = True
                logging.info("[%s] balance x%.2f: keystone rows %d, context rows %d; block %s mean shares "
                             "keystone %.4f -> %.4f, context %.4f -> %.4f (rest unchanged %.4f)",
                             WHO, b, ki.numel(), ci.numel(), blk, float(pK.mean()), float(new_pK.mean()),
                             float(pC.mean()), float((sh - new_pK).mean()), float((1 - sh).mean()))
            return out.transpose(1, 2).reshape(B, T, H * d)

        # 3) this node's own attention forward on its blocks. It replaces whatever attention forward is there (H3's,
        #    Kijai's low VRAM or Sage patches, ...), but only acts on calls where the balance applies; every other call
        #    goes to that previous forward untouched. Takes the input as a tensor or as Kijai's one-item list.
        def make_forward(prev, blk):
            def forward(self, x, rope_freqs=None, transformer_options={}):
                topts = transformer_options if isinstance(transformer_options, dict) else {}
                state = topts.get(KEY)
                if not state or not state.get("on", True) or state.get("key") is None:
                    return prev(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
                xt = x[0] if isinstance(x, list) else x
                s_len = xt.shape[0]
                idx = indices(topts, s_len, xt.device)
                if idx is None:
                    return prev(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
                if isinstance(x, list):
                    x = x.pop()                                          # Kijai's low VRAM block hands h over in a list
                ki, ci = idx
                device, dtype = x.device, x.dtype
                q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
                del x
                v = v.view(s_len, self.heads, self.head_dim)
                if rope_freqs is not None:
                    q = q.view(1, s_len, self.heads, self.head_dim)
                    k = k.view(1, s_len, self.heads, self.head_dim)
                    qw = comfy.model_management.cast_to(self.q_norm.weight, device=device)
                    kw = comfy.model_management.cast_to(self.k_norm.weight, device=device)
                    rot = rope_freqs.shape[-3] * 2
                    if comfy.model_management.in_training:
                        q, k = comfy.quant_ops.ck.rms_rope_split_half(
                            q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
                    else:
                        comfy.quant_ops.ck.rms_rope_split_half_(
                            q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
                    q = q[0]
                    k = k[0]
                else:
                    q = self.q_norm(q.view(s_len, self.heads, self.head_dim))
                    k = self.k_norm(k.view(s_len, self.heads, self.head_dim))
                q = q.transpose(0, 1).unsqueeze(0)
                k = k.transpose(0, 1).unsqueeze(0)
                v = v.transpose(0, 1).unsqueeze(0)
                # head groups keep the memory down (heads are independent, so the result is the same);
                # follows Kijai's low VRAM head_chunks when it's set
                n = max(1, min(int(topts.get("minimax_head_chunks", 4) or 4), self.heads))
                out = torch.empty((s_len, self.heads * self.head_dim), dtype=dtype, device=device)
                hs = 0
                for size in [self.heads // n + (1 if i < self.heads % n else 0) for i in range(n)]:
                    he = hs + size
                    o = balanced(q[:, hs:he], k[:, hs:he], v[:, hs:he], ki, ci, blk)
                    out[:, hs * self.head_dim:he * self.head_dim] = o.squeeze(0)
                    hs = he
                del q, k, v
                return self.out_proj(out)
            return forward

        diffusion_model = m.get_model_object("diffusion_model")
        h3_blocks = getattr(diffusion_model, "blocks", None)
        if not h3_blocks or not hasattr(h3_blocks[0], "attn") or not hasattr(h3_blocks[0].attn, "qkv_proj"):
            print(f"[{WHO}] model doesn't look like MiniMax H3 - passing through")
            return (model,)
        others = set()
        for i in chosen:
            if i >= len(h3_blocks):
                continue
            attn = h3_blocks[i].attn
            key_name = f"diffusion_model.blocks.{i}.attn.forward"
            prev = m.object_patches.get(key_name)
            if prev is not None:
                others.add(getattr(getattr(prev, "__func__", prev), "__name__", "patch"))
            else:
                prev = types.MethodType(type(attn).forward, attn)
            m.add_object_patch(key_name, types.MethodType(make_forward(prev, i), attn))
        if others:
            print(f"[{WHO}] sits on top of the attention patch already on those blocks ({', '.join(sorted(others))}); "
                  f"it takes over only where the balance applies")
        return (m,)


NODE_CLASS_MAPPINGS = {"GeiruKeystoneAttention": GeiruKeystoneAttention}
NODE_DISPLAY_NAME_MAPPINGS = {"GeiruKeystoneAttention": "Geiru Keystone Attention"}
