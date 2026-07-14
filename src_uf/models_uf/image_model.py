# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import torch
from torch import nn
import torch.nn.functional as F


from .common_model import CompressionModel
from ..layers.layers import DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2
from ..layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE, round_and_to_int8, \
    bias_pixel_shuffle_8

g_ch_src = 3 * 8 * 8
g_ch_d = 256
g_ch_enc_dec = 368


class IntraEncoder(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.chunk_enc = nn.Sequential(nn.Conv3d(g_ch_src, g_ch_d, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),  # 8 -> 4
                                       nn.SiLU(inplace=True),
                                       nn.Conv3d(g_ch_d, g_ch_d, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),  # 4 -> 2
                                       nn.SiLU(inplace=True),
                                       nn.Conv3d(g_ch_d, g_ch_d, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))  # 2 -> 1
                                       )
        self.enc_1 = DepthConvBlock(g_ch_d, g_ch_enc_dec)
        self.enc_2 = nn.Sequential(
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            nn.Conv2d(g_ch_enc_dec, N, 3, stride=2, padding=1),
        )

    def chunk_conv(self, feature):
        feature = self.chunk_enc(feature)

        if feature.shape[2] != 1:
            raise RuntimeError(f"expected temporal dim 1, got {feature.shape[2]}")

        return feature.squeeze(2)

    def forward(self, x, quant_step):
        B,T,C,H,W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = F.pixel_unshuffle(x, 8) # B*T, 8*8*C, 1, 1
        _,C1,H1,W1 = x.shape
        x = x.reshape(B, T, C1, H1, W1)
        feature = x.permute(0,2,1,3,4).contiguous() # B,C1,T,H1,W1
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(feature, quant_step)
        return self.forward_cuda(feature, quant_step)

    def forward_torch(self, out, quant_step):
        out = self.chunk_conv(out)
        out = self.enc_1(out)
        out = out * quant_step
        return self.enc_2(out)

    def forward_cuda(self, out, quant_step):
        out = self.chunk_conv(out)
        out = self.enc_1(out, quant_step=quant_step)
        return self.enc_2(out)


class IntraDecoder(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.dec_1 = nn.Sequential(
            ResidualBlockUpsample(N, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
        )
        self.dec_2 = DepthConvBlock(g_ch_enc_dec, g_ch_src)

    def forward(self, x, quant_step):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, quant_step)

        return self.forward_cuda(x, quant_step)

    def forward_torch(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step 
        out = self.dec_2(out) #特征feature
        return out

    def forward_cuda(self, x, quant_step):
        out = self.dec_1[0](x)
        out = self.dec_1[1](out)
        out = self.dec_1[2](out)
        out = self.dec_1[3](out)
        out = self.dec_1[4](out)
        out = self.dec_1[5](out)
        out = self.dec_1[6](out)
        out = self.dec_1[7](out)
        out = self.dec_1[8](out)
        out = self.dec_1[9](out)
        out = self.dec_1[10](out)
        out = self.dec_1[11](out)
        out = self.dec_1[12](out, quant_step=quant_step)
        out = self.dec_2(out)
        return out


class IntraSpecificFrame(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(g_ch_src, g_ch_src, 1)

    def forward(self, x):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x)
        return self.forward_cuda(x)

    def forward_torch(self, x):
        out = self.head(x)
        out = F.pixel_shuffle(out, 8)
        out = torch.clamp(out, 0., 1.)
        return out

    def forward_cuda(self, x):
        out = F.conv2d(x, self.head.weight)
        return bias_pixel_shuffle_8(out, self.head.bias)


class IntraSpecificFrames(nn.Module):
    def __init__(self, chunk_size=8):
        super().__init__()
        self.chunk_size = chunk_size
        self.head = nn.Conv2d(
            g_ch_src * chunk_size,
            g_ch_src * chunk_size,
            1,
            groups=chunk_size,
        )

    def forward(self, feature):
        b, c, h, w = feature.shape
        out = feature.unsqueeze(1).expand(-1, self.chunk_size, -1, -1, -1)
        out = out.reshape(b, self.chunk_size * c, h, w)
        out = self.head(out)
        out = out.reshape(b * self.chunk_size, g_ch_src, h, w)
        out = F.pixel_shuffle(out, 8)
        out = torch.clamp(out, 0., 1.)
        return out.reshape(b, self.chunk_size, 3, h * 8, w * 8)




class IntraStreamlinedEntropy(nn.Module):
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




class DCVCUFIntra(CompressionModel):
    def __init__(self, N=128, z_channel=128):
        super().__init__(z_channel=z_channel)

        self.enc = IntraEncoder(N)

        self.hyper_enc = nn.Sequential(
            DepthConvBlock(N, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
        )

        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(z_channel, z_channel),
            ResidualBlockUpsample(z_channel, z_channel),
            DepthConvBlock(z_channel, N),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(N, N * 3),
            DepthConvBlock(N * 3, N * 3),
            DepthConvBlock(N * 3, N * 3),
            nn.Conv2d(N * 3, N * 3, 1),
        )

        self.dec = IntraDecoder(N)
        self.recon_generation_net = IntraSpecificFrames()
        self.ref_feature_proj = nn.Conv2d(g_ch_src, g_ch_d, 1)
        self.streamlinedentropy = IntraStreamlinedEntropy(N, N * 3)

        self.q_scale_enc = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))

    @staticmethod
    def _quantize_ste(x):
        return x + (torch.round(x) - x).detach()

    @staticmethod
    def _probs_to_bits(probs):
        return -torch.log2(torch.clamp(probs, min=1e-9))

    def _get_index_tensor(self, qp, device):
        if isinstance(qp, torch.Tensor):
            qp = int(qp.item())
        return torch.tensor([qp], dtype=torch.int32, device=device)

    def _get_y_bits(self, y_q, scales):
        scales = torch.clamp(scales, min=0.11, max=16.0)
        gaussian = torch.distributions.normal.Normal(torch.zeros_like(scales), scales)
        probs = gaussian.cdf(y_q + 0.5) - gaussian.cdf(y_q - 0.5)
        return self._probs_to_bits(probs)

    def _get_z_bits(self, z_hat, qp):
        index = self._get_index_tensor(qp, z_hat.device)
        probs = self.bit_estimator_z.get_cdf(z_hat + 0.5, index) - \
            self.bit_estimator_z.get_cdf(z_hat - 0.5, index)
        return self._probs_to_bits(probs)

    def _chunk_mse(self, x, x_hat):
        mse = F.mse_loss(x_hat, x, reduction="none")
        return mse.flatten(1).mean(dim=1)

    def forward(self, x, qp):
        if isinstance(qp, torch.Tensor):
            qp = int(qp.item())

        curr_q_enc = self.q_scale_enc[qp:qp + 1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp + 1, :, :, :]

        y = self.enc(x, curr_q_enc)
        y_pad = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat = self._quantize_ste(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        _, _, y_h, y_w = y.shape
        params = params[:, :, :y_h, :y_w].contiguous()

        y_q, y_scales, y_hat = self.compress_prior_uf_quadtree_sem(y, params)
        feature = self.dec(y_hat, curr_q_dec)
        x_hat = self.recon_generation_net(feature)
        ref_feature = self.ref_feature_proj(feature)

        b, t, _, h, w = x.shape
        pixel_num = t * h * w
        bpp_y = torch.sum(self._get_y_bits(y_q, y_scales), dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(self._get_z_bits(z_hat, qp), dim=(1, 2, 3)) / pixel_num
        ssim = torch.zeros(b, device=x.device, dtype=x.dtype)

        return {
            "bpp": bpp_y + bpp_z,
            "x_hat": x_hat,
            "ref_chunk": x_hat,
            "ref_feature": ref_feature,
            "mse": self._chunk_mse(x, x_hat),
            "ssim": ssim,
        }

    def compress(self, x, qp):
        if isinstance(qp, torch.Tensor):
            qp = int(qp.item())
        device = x.device
        if device.type != "cuda":
            raise RuntimeError(
                "DCVCUFIntra.compress requires CUDA tensors because entropy coding uses CUDA streams."
            )
        if self.entropy_coder is None:
            self.update()

        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        y = self.enc(x, curr_q_enc)
        y_pad = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat, z_hat_write = round_and_to_int8(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()

        y_q, y_scales, y_hat = self.compress_prior_uf_quadtree_sem(y, params)

        cuda_event = torch.cuda.Event()
        cuda_event.record()
        feature = self.dec(y_hat, curr_q_dec)
        x_hat = self.recon_generation_net(feature)
        ref_feature = self.ref_feature_proj(feature)

        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        with torch.cuda.stream(cuda_stream):
            cuda_event.wait()
            self.entropy_coder.reset()
            self.bit_estimator_z.encode_z(z_hat_write, qp)
            self.gaussian_encoder.encode_y(y_q, y_scales)
            self.entropy_coder.flush()

        bit_stream = self.entropy_coder.get_encoded_stream()

        torch.cuda.synchronize(device=device)
        result = {
            "bit_stream": bit_stream,
            "x_hat": x_hat,
            "ref_chunk": x_hat,
            "ref_feature": ref_feature,
        }
        return result

    def decompress(self, bit_stream, sps, qp):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        if isinstance(qp, torch.Tensor):
            qp = int(qp.item())
        if device.type != "cuda":
            raise RuntimeError(
                "DCVCUFIntra.decompress requires a CUDA model because entropy decoding uses CUDA streams."
            )
        if self.entropy_coder is None:
            self.update()

        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        self.entropy_coder.set_use_two_entropy_coders(sps['ec_part'] == 1)
        self.entropy_coder.set_stream(bit_stream)
        z_size = self.get_downsampled_shape(sps['height'], sps['width'], 64)
        y_height, y_width = self.get_downsampled_shape(sps['height'], sps['width'], 16)
        self.bit_estimator_z.decode_z(z_size, qp)
        z_q = self.bit_estimator_z.get_z(z_size, device, dtype)
        z_hat = z_q

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = params[:, :, :y_height, :y_width].contiguous()

        y_hat = self.decompress_prior_uf_quadtree_sem(params)
        feature  = self.dec(y_hat, curr_q_dec)
        x_hat = self.recon_generation_net(feature)
        ref_feature = self.ref_feature_proj(feature)

        return {"x_hat": x_hat, "ref_chunk": x_hat, "ref_feature": ref_feature}
