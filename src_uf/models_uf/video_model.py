# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DCVC-UF style chunk-based video model.

This module is intentionally independent from the frame-by-frame DCVC-RT model. It provides the
main UF structural pieces: chunk coding, cross-frame interaction through a shared chunk latent,
cross-chunk context propagation, frame-specific decoders, and a streamlined entropy model that
uses single-step scale estimation for one arithmetic coding interaction over all y partitions.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .common_model import CompressionModel
from ..layers.layers import SubpelConv2x, DepthConvBlock, ResidualBlockUpsample, \
    ResidualBlockWithStride2
from ..layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE, round_and_to_int8, \
    bias_pixel_shuffle_8, bias_quant, replicate_pad

DEFAULT_CHUNK_SIZE = 8
DEFAULT_MODEL_SCALE = "ht-s"
qp_shift = [0, 8, 4]
extra_qp = max(qp_shift)

g_ch_recon = 320
g_ch_src_d = 3 * 8 * 8
g_ch_d = 256
g_ch_y = 128
g_ch_z = 128

@dataclass(frozen=True)
class UFModelConfig:
    name: str                   # name of the model preset（ht-s：轻量级, ht-l: 完全体, ld: 低延迟）
    chunk_size: int         # number of frames in a chunk
    encoder_blocks: int        # number of depth convolution blocks in the chunk encoder
    decoder_blocks: int        # number of depth convolution blocks in the chunk decoder
    context_blocks: int        # number of depth convolution blocks in the cross-chunk context generation
    frame_decoder_blocks: int   # number of depth convolution blocks in the frame-specific decoder


UF_MODEL_CONFIGS = {
    "ht-s": UFModelConfig("ht-s", 8, 6, 7, 11, 3),
    "ht-l": UFModelConfig("ht-l", 8, 7, 11, 12, 5),
    "ld": UFModelConfig("ld", 1, 3, 3, 9, 3),
}


class CrossChunkContextGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d*2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d)
        )
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d)
        )

    def forward(self, x, h):
        h1 = self.forward_part1(x, h)
        ctx = self.forward_part2(h1)
        return ctx , h1

    def forward_part1(self, x, h):
        h1 = self.conv1(torch.cat((x, h), dim=1))
        return h1

    def forward_part2(self, h1):
        ctx = self.conv2(h1)
        return ctx

#可能存在问题
class ChunkEncoder(nn.Module):
    def __init__(self, chunk_size=8, feature_ch=g_ch_d, context_ch=g_ch_d, y_ch=g_ch_y, block_num=6, heads=8):
        super().__init__()        
        self.frame_proj = nn.Linear(g_ch_src_d, g_ch_d)
        self.temporal_query = nn.Parameter(torch.randn(1, 1, g_ch_d))
        self.attn = nn.MultiheadAttention(embed_dim=g_ch_d,num_heads=heads,batch_first=True)
        
        self.crossframe_interaction = nn.Sequential(nn.Conv3d(g_ch_src_d, g_ch_d, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),  # 8 -> 4
                                                    nn.SiLU(inplace=True),
                                                    nn.Conv3d(g_ch_d, g_ch_d, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),  # 4 -> 2
                                                    nn.SiLU(inplace=True),
                                                    nn.Conv3d(g_ch_d, g_ch_d, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))  # 2 -> 1
                                                    )
        
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d)
        )
        self.conv3 = DepthConvBlock(g_ch_d, g_ch_d)
        self.down = nn.Conv2d(g_ch_d, g_ch_y, 3, stride=2, padding=1)


    def temporal_attention(self, feature):
        # feature: B,C,T,H,W
        b, c, t, h, w = feature.shape
        tokens = feature.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        tokens = self.frame_proj(tokens)  # B*H*W,T,g_ch_d
        query = self.temporal_query.expand(tokens.shape[0], -1, -1)
        feature, _ = self.attn(query, tokens, tokens, need_weights=False)
        feature = feature.squeeze(1)
        feature = feature.reshape(b, h, w, g_ch_d).permute(0, 3, 1, 2).contiguous()
        return feature


    def temporal_conv(self, feature):
        feature = self.crossframe_interaction(feature)

        if feature.shape[2] != 1:
            raise RuntimeError(f"expected temporal dim 1, got {feature.shape[2]}")

        return feature.squeeze(2)


    def forward(self, x, ctx, quant_step):
        B,T,C,H,W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = F.pixel_unshuffle(x, 8) # B*T, 8*8*C, 1, 1
        _,C1,H1,W1 = x.shape
        x = x.reshape(B, T, C1, H1, W1)
        feature = x.permute(0,2,1,3,4).contiguous() # B,C1,T,H1,W1
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(feature, ctx, quant_step)
        return self.forward_cuda(feature, ctx, quant_step)


    def forward_torch(self, feature, ctx, quant_step):
        #自注意力和卷积二选一
        #1
        #feature = self.temporal_attention(feature)
        #2
        feature = self.temporal_conv(feature)

        feature = self.conv2(torch.cat((feature, ctx), dim=1))
        feature = self.conv3(feature)
        feature = feature * quant_step
        feature = self.down(feature)
        return feature

    def forward_cuda(self, feature, ctx, quant_step):
        #feature = self.temporal_attention(feature)
        feature = self.temporal_conv(feature)

        feature = self.conv2(torch.cat((feature, ctx), dim=1))
        feature = self.conv3(feature, quant_step=quant_step)
        feature = self.down(feature)
        return feature


class ChunkDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = SubpelConv2x(g_ch_y, g_ch_d, 3, padding=1)
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d)
        )
        self.conv2 = nn.Conv2d(g_ch_d, g_ch_d, 1)

    def forward(self, x, ctx, quant_step,):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, ctx, quant_step)

        return self.forward_cuda(x, ctx, quant_step)

    def forward_torch(self, x, ctx, quant_step):
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = self.conv2(feature)
        feature = feature * quant_step
        return feature

    def forward_cuda(self, x, ctx, quant_step):
        feature = self.up(x, to_cat=ctx, cat_at_front=False)
        feature = self.conv1(feature)
        feature = F.conv2d(feature, self.conv2.weight)
        feature = bias_quant(feature, self.conv2.bias, quant_step)
        return feature


class FrameSpecificDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_d,     g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon)
        )
        self.head = nn.Conv2d(g_ch_recon, g_ch_src_d, 1)

    def forward(self, x):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x)

        return self.forward_cuda(x)

    def forward_torch(self, x):
        out = self.conv(x)
        out = self.head(out)
        out = F.pixel_shuffle(out, 8)
        out = torch.clamp(out, 0., 1.)
        return out

    def forward_cuda(self, x):
        out = self.conv[0](x)
        out = self.conv[1](out)
        out = self.conv[2](out)
        out = F.conv2d(out, self.head.weight)
        return bias_pixel_shuffle_8(out, self.head.bias)


class ParallelDepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, chunk_size, shortcut=False, force_adaptor=False):
        super().__init__()
        self.chunk_size = chunk_size
        self.out_ch = out_ch
        self.shortcut = shortcut

        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Conv2d(in_ch * chunk_size, out_ch * chunk_size, 1, groups=chunk_size)

        self.dc_0 = nn.Conv2d(out_ch * chunk_size, out_ch * chunk_size, 1, groups=chunk_size)
        self.dc_1 = nn.Conv2d(
            out_ch * chunk_size,
            out_ch * chunk_size,
            3,
            padding=1,
            groups=out_ch * chunk_size,
        )
        self.dc_2 = nn.Conv2d(out_ch * chunk_size, out_ch * chunk_size, 1, groups=chunk_size)
        self.ffn_0 = nn.Conv2d(out_ch * chunk_size, out_ch * 4 * chunk_size, 1, groups=chunk_size)
        self.ffn_1 = nn.Conv2d(out_ch * 2 * chunk_size, out_ch * chunk_size, 1, groups=chunk_size)

    @staticmethod
    def wsilu(x):
        return torch.sigmoid(4.0 * x) * x

    def wsilu_chunk_add(self, x):
        b, _, h, w = x.shape
        x = self.wsilu(x)
        x = x.reshape(b, self.chunk_size, self.out_ch * 4, h, w)
        x1, x2 = x.chunk(2, dim=2)
        return (x1 + x2).reshape(b, self.chunk_size * self.out_ch * 2, h, w)

    def forward(self, x):
        if self.adaptor is not None:
            x = self.adaptor(x)
        out = self.dc_2(self.dc_1(self.wsilu(self.dc_0(x)))) + x
        out = self.ffn_1(self.wsilu_chunk_add(self.ffn_0(out))) + out
        if self.shortcut:
            out = out + x
        return out


class FrameSpecificDecoders(nn.Module):
    def __init__(self, chunk_size=8):
        super().__init__()
        self.chunk_size = chunk_size
        self.conv = nn.Sequential(
            ParallelDepthConvBlock(g_ch_d, g_ch_recon, chunk_size),
            ParallelDepthConvBlock(g_ch_recon, g_ch_recon, chunk_size),
            ParallelDepthConvBlock(g_ch_recon, g_ch_recon, chunk_size),
        )
        self.head = nn.Conv2d(
            g_ch_recon * chunk_size,
            g_ch_src_d * chunk_size,
            1,
            groups=chunk_size,
        )

    def forward(self, feature):
        b, c, h, w = feature.shape
        out = feature.unsqueeze(1).expand(-1, self.chunk_size, -1, -1, -1)
        out = out.reshape(b, self.chunk_size * c, h, w)
        out = self.conv(out)
        out = self.head(out)
        out = out.reshape(b * self.chunk_size, g_ch_src_d, h, w)
        out = F.pixel_shuffle(out, 8)
        out = torch.clamp(out, 0., 1.)
        return out.reshape(b, self.chunk_size, 3, h * 8, w * 8)


class HyperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z)
        )

    def forward(self, x):
        return self.conv(x)


class HyperDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            DepthConvBlock(g_ch_z, g_ch_y)
        )

    def forward(self, x):
        return self.conv(x)


class PriorFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 3, 1),
        )

    def forward(self, x):
        return self.conv(x)


class RefChunk():
    def __init__(self):
        self.chunk = None
        self.feature = None
        self.hidden = None  # 保存上一 chunk 的 h1
        self.poc = None


class ChunkFeatureAdaptorI(nn.Module):
    def __init__(self, chunk_size=8):
        super().__init__()
        self.chunk_size = chunk_size
        self.temporal_fusion = nn.Sequential(
            nn.Conv3d(g_ch_src_d, g_ch_d, kernel_size=(3, 1, 1),
                      stride=(2, 1, 1), padding=(1, 0, 0)),  # 8 -> 4
            nn.SiLU(inplace=True),
            nn.Conv3d(g_ch_d, g_ch_d, kernel_size=(3, 1, 1),
                      stride=(2, 1, 1), padding=(1, 0, 0)),  # 4 -> 2
            nn.SiLU(inplace=True),
            nn.Conv3d(g_ch_d, g_ch_d, kernel_size=(3, 1, 1),
                      stride=(2, 1, 1), padding=(1, 0, 0)),  # 2 -> 1
        )
        self.refine = DepthConvBlock(g_ch_d, g_ch_d)

    def forward(self, chunk):
        # chunk: B,T,3,H,W
        b, t, c, h, w = chunk.shape
        assert t == self.chunk_size

        chunk = chunk.reshape(b * t, c, h, w)
        chunk = F.pixel_unshuffle(chunk, 8)

        _, patch_ch, patch_h, patch_w = chunk.shape
        chunk = chunk.reshape(b, t, patch_ch, patch_h, patch_w)
        chunk = chunk.permute(0, 2, 1, 3, 4).contiguous()  # B,192,T,H/8,W/8

        feature = self.temporal_fusion(chunk)              # B,256,1,H/8,W/8
        feature = feature.squeeze(2)                       # B,256,H/8,W/8
        feature = self.refine(feature)
        return feature


class StreamlinedEntropyModel(nn.Module):
    def __init__(self, y_ch, prior_ch):
        super().__init__()
        self.mean_prior_adaptors = nn.ModuleList([
            nn.Conv2d(prior_ch + y_ch, prior_ch, 1)
            for _ in range(3)
        ])

        self.mean_prior = nn.Sequential(
            DepthConvBlock(prior_ch, prior_ch, force_adaptor=True),
            DepthConvBlock(prior_ch, prior_ch),
            nn.Conv2d(prior_ch, y_ch, 1),
        )

    @staticmethod
    def separate_prior(common_params):
        q_dec, scales, mean0 = common_params.chunk(3, 1)
        q_dec = torch.clamp_min(q_dec, 0.5)
        return q_dec, scales, mean0

    def estimate_scales(self, common_params):
        _, scales, _ = self.separate_prior(common_params)
        return scales

    def mean_for_step(self, step, common_params, mean0, y_hat_so_far):
        if step == 0:
            return mean0

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        params = self.mean_prior_adaptors[step - 1](params)
        return self.mean_prior(params)

    @staticmethod
    def quantize_ste(x):
        x_quant = torch.round(x).clamp(-128.0, 127.0)
        return x + (x_quant - x).detach()

    def quantize(self, y, common_params, masks, force_zero_thres=None,
                 scale_min=None, scale_max=None):
        _, scales, mean0 = self.separate_prior(common_params)
        active_mask = None
        if force_zero_thres is not None:
            scales_for_mask = scales
            if scale_min is not None and scale_max is not None:
                scales_for_mask = scales_for_mask.clamp(scale_min, scale_max)
            active_mask = (scales_for_mask > force_zero_thres).to(dtype=y.dtype)

        y_q = torch.zeros_like(y)
        y_hat_so_far = torch.zeros_like(y)

        for step, mask in enumerate(masks):
            mean = self.mean_for_step(step, common_params, mean0, y_hat_so_far)
            curr_y_q = self.quantize_ste((y - mean) * mask)
            if active_mask is not None:
                curr_y_q = curr_y_q * active_mask
            curr_y_hat = (curr_y_q + mean) * mask

            y_q = y_q + curr_y_q
            y_hat_so_far = y_hat_so_far + curr_y_hat

        return y_q, scales, y_hat_so_far, None

    def restore(self, y_q, common_params, masks):
        _, scales, mean0 = self.separate_prior(common_params)

        y_hat_so_far = torch.zeros_like(y_q)

        for step, mask in enumerate(masks):
            mean = self.mean_for_step(step, common_params, mean0, y_hat_so_far)
            curr_y_hat = (y_q + mean) * mask
            y_hat_so_far = y_hat_so_far + curr_y_hat

        return y_hat_so_far, scales, None


class DCVCUF(CompressionModel):
    def __init__(self):
        super().__init__(z_channel=g_ch_z, extra_qp=extra_qp)
        self.qp_shift = qp_shift

        self.feature_adaptor_i = ChunkFeatureAdaptorI(chunk_size=8)
        self.feature_adaptor_p = nn.Conv2d(g_ch_d, g_ch_d, 1)
        self.hidden_adaptor_i = DepthConvBlock(g_ch_d, g_ch_d)

        self.context_extractor = CrossChunkContextGeneration()
        self.encoder = ChunkEncoder()
        self.decoder = ChunkDecoder()        
        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()
        self.temporal_prior_encoder = ResidualBlockWithStride2(g_ch_d, g_ch_y * 2)
        self.prior_fusion = PriorFusion()
        self.streamlinedentropy = StreamlinedEntropyModel(g_ch_y, g_ch_y * 3)#common.py里四叉树模型调用
        self.recon_generation_net = FrameSpecificDecoders()

        self.q_encoder = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        self.q_decoder = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        self.q_entropy = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
 
        self.dpb = []
        self.max_dpb_size = 1
        self.curr_poc = 0

    def add_ref_chunk(self, feature=None, chunk=None, hidden=None, increase_poc=True):
        if feature is not None and (feature.dim() != 4 or feature.shape[1] != g_ch_d):
            raise RuntimeError(
                f"expected propagated feature with shape B,{g_ch_d},H,W, got {tuple(feature.shape)}"
            )
        if chunk is not None and (
            chunk.dim() != 5 or chunk.shape[1] != self.feature_adaptor_i.chunk_size or chunk.shape[2] != 3
        ):
            raise RuntimeError(
                f"expected reference chunk with shape B,{self.feature_adaptor_i.chunk_size},3,H,W, "
                f"got {tuple(chunk.shape)}"
            )

        ref_chunk = RefChunk()
        ref_chunk.poc = self.curr_poc
        ref_chunk.feature = feature
        ref_chunk.chunk = chunk
        ref_chunk.hidden = hidden

        if len(self.dpb) >= self.max_dpb_size:
            self.dpb.pop(-1)
        self.dpb.insert(0, ref_chunk)
        if increase_poc:
            self.curr_poc += 1
            #self.curr_poc += self.chunk_size   后续回头看看

    def add_ref_key_chunk(self, x_hat, increase_poc=True):
        self.add_ref_chunk(feature=None, chunk=x_hat, hidden=None, increase_poc=increase_poc)

    def add_ref_key_feature(self, ref_feature, ref_chunk=None, increase_poc=True):
        self.add_ref_chunk(feature=ref_feature, chunk=ref_chunk, hidden=None, increase_poc=increase_poc)

    def apply_feature_adaptor(self):
        if len(self.dpb) == 0:
            raise RuntimeError("reference chunk is required before coding P chunk")
        ref = self.dpb[0]
        if ref.feature is None:
            if ref.chunk is None:
                raise RuntimeError("reference chunk or feature is required before coding P chunk")
            return self.feature_adaptor_i(ref.chunk)
        return self.feature_adaptor_p(ref.feature)

    def apply_hidden_adaptor(self, feature):
        if len(self.dpb) == 0:
            raise RuntimeError("reference chunk is required before coding P chunk")

        ref = self.dpb[0]

        # P chunk: 直接用前一个 chunk 保存下来的 h1
        if ref.hidden is not None:
            return ref.hidden

        # I chunk / key chunk: 没有历史 h，就初始化一个 h0
        return self.hidden_adaptor_i(feature)

    def res_prior_param_decoder(self, z_hat, ctx, q_entropy):
        hierarchical_params = self.hyper_decoder(z_hat)
        ctx = ctx * q_entropy
        temporal_params = self.temporal_prior_encoder(ctx)

        _, _, H, W = temporal_params.shape
        hierarchical_params = hierarchical_params[:, :, :H, :W].contiguous()

        params = self.prior_fusion(torch.cat((hierarchical_params, temporal_params), dim=1))
        return params

    def get_recon_and_feature(self, y_hat, ctx, q_decoder):
        feature = self.decoder(y_hat, ctx, q_decoder)
        x_hat = self.recon_generation_net(feature)
        return x_hat, feature

    def clear_dpb(self):
        self.dpb.clear()

    def set_curr_poc(self, poc):
        self.curr_poc = poc

    def reset_ref_feature(self):
        if len(self.dpb) > 0:
            self.dpb[0].feature = None

    def prepare_feature_adaptor_i(self, last_qp):
        if self.dpb[0].chunk is None:
            self.dpb[0].chunk = self.recon_generation_net(self.dpb[0].feature).clamp_(0, 1)
            self.reset_ref_feature()

    def compress(self, x, qp):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        device = x.device
        q_encoder = self.q_encoder[qp:qp+1, :, :, :]
        q_decoder = self.q_decoder[qp:qp+1, :, :, :]
        q_entropy = self.q_entropy[qp:qp+1, :, :, :]

        feature = self.apply_feature_adaptor()
        h0 = self.apply_hidden_adaptor(feature)
        h1 = self.context_extractor.forward_part1(feature, h0)
        ctx = self.context_extractor.forward_part2(h1)
        y = self.encoder(x, ctx, q_encoder)

        hyper_inp = self.pad_for_y(y)

        z = self.hyper_encoder(hyper_inp)
        z_hat, z_hat_write = round_and_to_int8(z)
        cuda_event_z_ready = torch.cuda.Event()
        cuda_event_z_ready.record()
        params = self.res_prior_param_decoder(z_hat, ctx, q_entropy)

        y_q, y_scales, y_hat = self.compress_prior_uf_quadtree_sem(y, params)

        cuda_event_y_ready = torch.cuda.Event()
        cuda_event_y_ready.record()
        feature = self.decoder(y_hat, ctx, q_decoder)

        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        with torch.cuda.stream(cuda_stream):
            self.entropy_coder.reset()
            cuda_event_z_ready.wait()
            self.bit_estimator_z.encode_z(z_hat_write, qp)
            cuda_event_y_ready.wait()
            self.gaussian_encoder.encode_y(y_q, y_scales)
            self.entropy_coder.flush()

        bit_stream = self.entropy_coder.get_encoded_stream()

        torch.cuda.synchronize(device=device)
        self.add_ref_chunk(feature, None, hidden=h1)
        return {
            'bit_stream': bit_stream,
        }

    def decompress(self, bit_stream, sps, qp):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        q_decoder = self.q_decoder[qp:qp+1, :, :, :]
        q_entropy = self.q_entropy[qp:qp+1, :, :, :]

        self.entropy_coder.set_use_two_entropy_coders(sps['ec_part'] == 1)
        self.entropy_coder.set_stream(bit_stream)
        z_size = self.get_downsampled_shape(sps['height'], sps['width'], 64)
        self.bit_estimator_z.decode_z(z_size, qp)

        feature = self.apply_feature_adaptor()
        h0 = self.apply_hidden_adaptor(feature)
        h1 = self.context_extractor.forward_part1(feature, h0)
        ctx = self.context_extractor.forward_part2(h1)
        z_hat = self.bit_estimator_z.get_z(z_size, device, dtype)
        params = self.res_prior_param_decoder(z_hat, ctx, q_entropy)

        infos = self.decompress_prior_uf_quadtree_sem_part1(params)

        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        with torch.cuda.stream(cuda_stream):

            y_hat = self.decompress_prior_uf_quadtree_sem_part2(params, infos)
            
            cuda_event = torch.cuda.Event()
            cuda_event.record()

        cuda_event.wait()
        x_hat, feature = self.get_recon_and_feature(y_hat, ctx, q_decoder)

        self.add_ref_chunk(feature, x_hat, hidden=h1)
        return {
            'x_hat': x_hat,
        }

    def shift_qp(self, qp, fa_idx):
        return qp + self.qp_shift[fa_idx]


def get_uf_model_config(name=DEFAULT_MODEL_SCALE):
    return UF_MODEL_CONFIGS[name]
