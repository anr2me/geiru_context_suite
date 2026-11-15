"""Keystone attention for Geiru Hybrid Windows (the same balanced attention as Geiru Keystone Attention).

Inside each window's own steps, the keystone (window 1's finished frame 0, locked into the window's first slot) and
the overlap carried from the windows before (the context) are both locked latents. On the chosen blocks, the slice of
attention shared by those two is re-split so the keystone gets `balance` times its normal pull and the context gives
up exactly what the keystone gains. Everything else (prompt, references, audio, new frames) is untouched.
Off on the blend steps (no keystone there) and on window 1 (no keystone yet).
"""
import logging
import re
import types

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.patcher_extension
import comfy.quant_ops

WHO = "Geiru Hybrid Windows: keystone attention"
KEY = "geiru_hybrid_keystone_attention"


class _Skip(Exception):
    pass


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




def add_keystone_attention(m, balance, chosen, anchors_fn):
    """Put keystone attention on model patcher m (already a clone), on blocks `chosen`.
    anchors_fn(lt): the window-local latents that are keystones on this model call (or None = off for this call);
    anchors_fn(None) is asked first, as a quick on/off check."""
    b = float(balance)
    stats = {"logged": False}

    # 1) wrapper: per model call, find the keystone rows and the context rows (video rows only)
    def wrapper(executor, x, timestep, context, transformer_options={}, *args, **kwargs):
        state = {"key": None, "ctx": None, "cache": {}, "on": True}
        try:
            if anchors_fn(None) is None:              # nothing to balance on this call (blend steps, window 1, ...)
                raise _Skip()
            v = x[0] if isinstance(x, (list, tuple)) else x
            lt, lh, lw = v.shape[2], v.shape[3] + (v.shape[3] % 2), v.shape[4] + (v.shape[4] % 2)
            held = _held_video_latents(kwargs.get("denoise_mask"), lt)
            if held is not None:
                held = F.pad(held.float(), (0, lw - held.shape[-1], 0, lh - held.shape[-2]), mode="replicate") > 0.5
                held = held.reshape(lt, lh // 2, 2, lw // 2, 2).all(dim=4).all(dim=2)   # [lt, rows_h, rows_w]
                anchor = torch.zeros(lt, dtype=torch.bool)
                for s in (anchors_fn(lt) or []):
                    if -lt <= s < lt:
                        anchor[s] = True
                anchor &= held.reshape(lt, -1).all(dim=1).cpu()   # only fully held latents count as anchors
                rows = held.reshape(lt, -1).cpu()
                key = (anchor[:, None] & rows).reshape(-1)
                ctx = (~anchor[:, None] & rows).reshape(-1)
                if bool(key.any()) and bool(ctx.any()):
                    state["key"], state["ctx"] = key, ctx
        except _Skip:
            pass
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
        print(f"[{WHO}] model doesn't look like MiniMax H3 - keystone attention not added")
        return m
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
    return m


