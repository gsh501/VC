# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
from pytorch_msssim import MS_SSIM
from ..layers.cuda_inference import combine_for_reading_2x, \
    restore_y_2x, restore_y_2x_with_cat_after, add_and_multiply, \
    replicate_pad, restore_y_4x, clamp_reciprocal_with_quant
from .entropy_models import BitEstimator, GaussianEncoder, EntropyCoder
from torch.autograd import Function
import math

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None
    
class CompressionModel(nn.Module):
    def __init__(self, z_channel, extra_qp=0):
        super().__init__()

        self.z_channel = z_channel
        self.entropy_coder = None
        self.bit_estimator_z = BitEstimator(64 + extra_qp, z_channel)
        self.gaussian_encoder = GaussianEncoder()
        self.mse = nn.MSELoss(reduction='none')
        self.ssim = MS_SSIM(data_range=1.0, size_average=False)

        self.masks = {}
        self.cuda_streams = {}
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.xavier_normal_(m.weight, math.sqrt(2))
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)

    def get_cuda_stream(self, device, idx=0, priority=0):
        key = f"{device}_{priority}_{idx}"
        if key not in self.cuda_streams:
            self.cuda_streams[key] = torch.cuda.Stream(device, priority=priority)
        return self.cuda_streams[key]
    
    def add_noise(self, x):
        noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
        noise = noise.clone().detach()
        return x + noise
    
    @staticmethod
    def probs_to_bits(probs):
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = LowerBound.apply(bits, 0)
        return bits

    @staticmethod
    def get_qp_num():
        return 64

    @staticmethod
    def get_padding_size(height, width, p=64):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        padding_right = new_w - width
        padding_bottom = new_h - height
        return padding_right, padding_bottom

    @staticmethod
    def get_downsampled_shape(height, width, p):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        return int(new_h / p + 0.5), int(new_w / p + 0.5)
    
    def soft_clip(self, x, min_val, max_val):
        y = torch.sigmoid(x)           # 将输出压缩到(0,1)
        y = min_val + (max_val - min_val) * y  # 再缩放到[a,b]
        return y
    
    def quant(self, x, force_detach=False):
        if self.training or force_detach:
            n = torch.round(x) - x
            n = n.clone().detach()
            return x + n

        return torch.round(x)
    
    @staticmethod
    def get_index_tensor(q_index, device):
        if not isinstance(q_index, list):
            q_index = [q_index]
        return torch.tensor(q_index, dtype=torch.int32, device=device)
    
    def get_y_gaussian_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-2, 64)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return CompressionModel.probs_to_bits(probs)
    
    @staticmethod
    def get_y_gaussian_bits_safe(y, sigma, prob_clamp=1e-6, bin_size=1.0):
        # Ensure proper calculation precision
        @torch.no_grad()  # We don't need gradient through the assert check
        def check_sigma(s):
            assert s.min() > 0, f"Invalid sigma value: {s.min()}"
            return s

        y = y.float()
        sigma = check_sigma(sigma.float())  # Check sigma for validity
        mu = torch.zeros_like(sigma)
        gaussian = torch.distributions.normal.Normal(mu, sigma)

        # Safe log probability mass calculation
        def safe_log_prob_mass(dist, x, bin_size, prob_clamp):
            # Calculate probability mass: CDF(x+0.5) - CDF(x-0.5)
            prob_mass = dist.cdf(x + 0.5 * bin_size) - dist.cdf(x - 0.5 * bin_size)

            # Use approximation for numerical stability when probability mass is small
            log_prob = torch.where(
                prob_mass > prob_clamp,
                torch.log(torch.clamp(prob_mass, min=1e-10)),
                # Use log of PDF times bin size as approximation
                # For Gaussian distribution, this is a good approximation
                dist.log_prob(x) + math.log(bin_size)
            )
            return log_prob, prob_mass

        # Calculate log probability and probability mass
        log_probs, probs = safe_log_prob_mass(gaussian, y, bin_size, prob_clamp)

        # Convert from nats to bits but DO NOT sum
        bits = torch.clamp(-log_probs / math.log(2.0), 0, 50)  # Clamping to prevent extremely large bit values
        return bits

    def get_y_laplace_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def get_z_bits(self, z, bit_estimator, index):
        probs = bit_estimator.get_cdf(z + 0.5, index) - bit_estimator.get_cdf(z - 0.5, index)
        probs = probs.to(torch.float32)
        return CompressionModel.probs_to_bits(probs)
    
    @staticmethod
    def get_z_bits_safe(z, bit_estimator,index, prob_clamp=1e-6):
        probs = bit_estimator.get_cdf(z + 0.5, index) - bit_estimator.get_cdf(z - 0.5, index)
        probs = probs.to(torch.float32)
         # Calculate safe log probability
        def safe_log_probs(p, eps=1e-10):
            # Direct log calculation when probability is above threshold
            # Otherwise use Laplacian approximation for numerical stability
            log_p = torch.where(
                p > prob_clamp,
                torch.log(torch.clamp(p, min=eps)),
                # Using properties of Laplace distribution for approximation
                # Laplace distribution has exponential decay in tails, so linear approximation works well
                # This is consistent with the BitEstimator model characteristics
                torch.log(torch.tensor(prob_clamp, device=p.device, dtype=p.dtype)) + (p - prob_clamp) / prob_clamp
            )
            return log_p
        log_probs = safe_log_probs(probs)
        bits = -log_probs / math.log(2.0)  # Convert to base-2 logarithm (bits)
        # Limit extreme values to prevent gradient explosion
        bits = torch.clamp(bits, 0, 50)
        return bits


    def update(self, force_zero_thres=None):
        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder.update(self.entropy_coder, force_zero_thres=force_zero_thres)
        self.bit_estimator_z.update(self.entropy_coder)

    def set_use_two_entropy_coders(self, use_two_entropy_coders):
        self.entropy_coder.set_use_two_entropy_coders(use_two_entropy_coders)

    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_r, padding_b = self.get_padding_size(H, W, 4)
        y_pad = replicate_pad(y, padding_b, padding_r)
        return y_pad

    def separate_prior(self, params, is_video=False):
        if is_video:
            quant_step, scales, means = params.chunk(3, 1)
            quant_step = torch.clamp_min(quant_step, 0.5)
            q_enc = 1. / quant_step
            q_dec = quant_step
        else:
            q = params[:, :2, :, :]
            q_enc, q_dec = (torch.sigmoid(q) * 1.5 + 0.5).chunk(2, 1)
            scales, means = params[:, 2:, :, :].chunk(2, 1)
        return q_enc, q_dec, scales, means

   


    @staticmethod
    def separate_prior_for_video_encoding(params, y):
        q_dec, scales, means = params.chunk(3, 1)
        q_dec, y = clamp_reciprocal_with_quant(q_dec, y, 0.5)
        return y, q_dec, scales, means
    

    @staticmethod
    def separate_prior_for_video_encoding_t(params, y):
        q_dec, scales, means = params.chunk(3, 1)
        
         # 统计 q_dec 中小于 0.5 的元素个数
        # num_lt_05 = (q_dec < 0.5).sum().item()
        # print(f"Number of elements in q_dec less than 0.5: {num_lt_05}")
        q_dec = LowerBound.apply(q_dec, 0.5)
        q_enc = torch.reciprocal(q_dec)
        y = y * q_enc
        return y, q_dec, scales, means



    @staticmethod
    def separate_prior_for_video_decoding(params):
        quant_step, scales, means = params.chunk(3, 1)
        quant_step = torch.clamp_min(quant_step, 0.5)
        return quant_step, scales, means

    def process_with_mask(self, y, scales, means, mask):
        return self.gaussian_encoder.process_with_mask(y, scales, means, mask)
    
    def process_with_mask_t(self,y, scales, means, mask, force_zero_thres=None):

        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        if force_zero_thres is not None:
            cond = scales_hat > force_zero_thres
            y_q = y_q * cond
        # y_q = self.soft_clip(y_q, -128., 127.)
        # y_q = torch.clamp(y_q, -128., 127.)
        y_hat = y_q + means_hat
        

        return y_res, y_q, y_hat, scales_hat

    @staticmethod
    def get_one_mask(micro_mask, height, width, dtype, device):
        mask = torch.tensor(micro_mask, dtype=dtype, device=device)
        mask = mask.repeat((height + 1) // 2, (width + 1) // 2)
        mask = mask[:height, :width]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 0)
        return mask

    def get_mask_4x(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}_{width}_{height}_4x"
        with torch.no_grad():
            if curr_mask_str not in self.masks:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0 = self.get_one_mask(((1, 0), (0, 0)), height, width, dtype, device)
                m1 = self.get_one_mask(((0, 1), (0, 0)), height, width, dtype, device)
                m2 = self.get_one_mask(((0, 0), (1, 0)), height, width, dtype, device)
                m3 = self.get_one_mask(((0, 0), (0, 1)), height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    def get_mask_2x(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}_{width}_{height}_2x"
        with torch.no_grad():
            if curr_mask_str not in self.masks:
                assert channel % 2 == 0
                m = torch.ones((batch, channel // 2, height, width), dtype=dtype, device=device)
                m0 = self.get_one_mask(((1, 0), (0, 1)), height, width, dtype, device)
                m1 = self.get_one_mask(((0, 1), (1, 0)), height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1), dim=1)
                mask_1 = torch.cat((m * m1, m * m0), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    @staticmethod
    def single_part_for_writing_4x(x):
        x0, x1, x2, x3 = x.chunk(4, 1)
        return (x0 + x1) + (x2 + x3)

    @staticmethod
    def single_part_for_writing_2x(x):
        x0, x1 = x.chunk(2, 1)
        return x0 + x1

    def compress_prior_2x(self, y, common_params, y_spatial_prior):
        y, q_dec, scales, means = self.separate_prior_for_video_encoding(common_params, y)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_2x(B, C, H, W, dtype, device)

        _, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)
        cat_params = torch.cat((y_hat_0, common_params), dim=1)
        scales, means = y_spatial_prior(cat_params).chunk(2, 1)
        _, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat = add_and_multiply(y_hat_0, y_hat_1, q_dec)

        y_q_w_0 = self.single_part_for_writing_2x(y_q_0)
        y_q_w_1 = self.single_part_for_writing_2x(y_q_1)
        s_w_0 = self.single_part_for_writing_2x(s_hat_0)
        s_w_1 = self.single_part_for_writing_2x(s_hat_1)
        return y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat
    
    def compress_prior_2x_write(self, y, common_params, y_spatial_prior):
        y, q_dec, scales, means = self.separate_prior_for_video_encoding(common_params, y)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_2x(B, C, H, W, dtype, device)

        _, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)
        cat_params = torch.cat((y_hat_0, common_params), dim=1)
        scales, means = y_spatial_prior(cat_params).chunk(2, 1)
        _, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat = add_and_multiply(y_hat_0, y_hat_1, q_dec)

        y_q_w_0 = self.single_part_for_writing_2x(y_q_0)
        y_q_w_1 = self.single_part_for_writing_2x(y_q_1)
        s_w_0 = self.single_part_for_writing_2x(s_hat_0)
        s_w_1 = self.single_part_for_writing_2x(s_hat_1)
        return y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat
    
    def compress_prior_2x_t(self, y, common_params, y_spatial_prior):
        y, q_dec, scales, means = self.separate_prior_for_video_encoding(common_params, y)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_2x(B, C, H, W, dtype, device)

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask_t(y, scales, means, mask_0)
        cat_params = torch.cat((y_hat_0, common_params), dim=1)
        
        scales, means = y_spatial_prior(cat_params).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask_t(y, scales, means, mask_1)
        y_res = y_res_0 + y_res_1
        y_q = y_q_0 + y_q_1
        y_s = s_hat_0 + s_hat_1
        y_hat = y_hat_0 + y_hat_1
        y_hat = y_hat * q_dec

       
        return y_res,y_q,y_s, y_hat
    

    def compress_prior_2x_t_1(self, y, common_params,mask, y_spatial_prior):
        y, q_dec, scales, means = self.separate_prior_for_video_encoding_t(common_params, y)
        
        # dtype = y.dtype
        # device = y.device
        # B, C, H, W = y.size()
        # mask_0, mask_1 = self.get_mask_2x(B, C, H, W, dtype, device)
        mask_0 = mask
        mask_1 = torch.ones_like(mask_0) - mask_0

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask_t(y, scales, means, mask_0)
        
        cat_params = torch.cat((y_hat_0, common_params), dim=1)
        
        scales, means = y_spatial_prior(cat_params).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask_t(y, scales, means, mask_1)
        y_res = y_res_0 + y_res_1
        y_q = y_q_0 + y_q_1
        y_s = s_hat_0 + s_hat_1
        y_hat = y_hat_0 + y_hat_1
        y_hat = y_hat * q_dec

       
        return y_res,y_q,y_s, y_hat
    



    def decompress_prior_2x(self, common_params, y_spatial_prior):
        infos = self.decompress_prior_2x_part1(common_params)
        y_hat = self.decompress_prior_2x_part2(common_params, y_spatial_prior, infos)
        return y_hat

    def decompress_prior_2x_part1(self, common_params):
        q_dec, scales, means = self.separate_prior_for_video_decoding(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1 = self.get_mask_2x(B, C, H, W, dtype, device)

        scales_r = combine_for_reading_2x(scales, mask_0, inplace=False)
        indexes, skip_cond = self.gaussian_encoder.build_indexes_decoder(scales_r)
        self.gaussian_encoder.decode_y(indexes)
        infos = {
            "q_dec": q_dec,
            "mask_0": mask_0,
            "mask_1": mask_1,
            "means": means,
            "scales_r": scales_r,
            "skip_cond": skip_cond,
            "indexes": indexes,
        }
        return infos

    def decompress_prior_2x_part2(self, common_params, y_spatial_prior, infos):
        dtype = common_params.dtype
        device = common_params.device
        y_q_r = self.gaussian_encoder.get_y(infos["scales_r"].shape,
                                            infos["scales_r"].numel(),
                                            dtype, device,
                                            infos["skip_cond"], infos["indexes"])
        y_hat_0, cat_params = restore_y_2x_with_cat_after(y_q_r, infos["means"], infos["mask_0"],
                                                          common_params)
        scales, means = y_spatial_prior(cat_params).chunk(2, 1)
        scales_r = combine_for_reading_2x(scales, infos["mask_1"], inplace=True)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_1 = restore_y_2x(y_q_r, means, infos["mask_1"])

        y_hat = add_and_multiply(y_hat_0, y_hat_1, infos["q_dec"])
        return y_hat

    def compress_prior_4x(self, y, common_params, y_spatial_prior_reduction,
                          y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                          y_spatial_prior_adaptor_3, y_spatial_prior):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        q_enc, q_dec, scales, means = self.separate_prior(common_params, False)
        common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        y = y * q_enc

        _, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        _, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        _, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        _, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        y_hat = y_hat_so_far + y_hat_3
        y_hat = y_hat * q_dec

        y_q_w_0 = self.single_part_for_writing_4x(y_q_0)
        y_q_w_1 = self.single_part_for_writing_4x(y_q_1)
        y_q_w_2 = self.single_part_for_writing_4x(y_q_2)
        y_q_w_3 = self.single_part_for_writing_4x(y_q_3)
        s_w_0 = self.single_part_for_writing_4x(s_hat_0)
        s_w_1 = self.single_part_for_writing_4x(s_hat_1)
        s_w_2 = self.single_part_for_writing_4x(s_hat_2)
        s_w_3 = self.single_part_for_writing_4x(s_hat_3)
        return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, s_w_0, s_w_1, s_w_2, s_w_3, y_hat
    

    def compress_prior_4x_t(self, y, common_params, y_spatial_prior_reduction,
                          y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                          y_spatial_prior_adaptor_3, y_spatial_prior):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        q_enc, q_dec, scales, means = self.separate_prior(common_params, False)
        common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        y = y * q_enc

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask_t(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask_t(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_res_2, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask_t(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_res_3, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask_t(y, scales, means, mask_3)
        y_hat = y_hat_so_far + y_hat_3
        y_hat = y_hat * q_dec
        y_q = y_q_0 + y_q_1 + y_q_2 + y_q_3
        y_s = s_hat_0 + s_hat_1 + s_hat_2+ s_hat_3
        y_res = y_res_0 + y_res_1 + y_res_2 + y_res_3

        return y_res,y_q,y_s, y_hat

    def decompress_prior_4x(self, common_params, y_spatial_prior_reduction,
                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                            y_spatial_prior_adaptor_3, y_spatial_prior):
        _, quant_step, scales, means = self.separate_prior(common_params, False)
        common_params = y_spatial_prior_reduction(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        scales_r = self.single_part_for_writing_4x(scales * mask_0)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_0)
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        scales_r = self.single_part_for_writing_4x(scales * mask_1)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        scales_r = self.single_part_for_writing_4x(scales * mask_2)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_2)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        scales_r = self.single_part_for_writing_4x(scales * mask_3)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_3)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat = y_hat_so_far * quant_step

        return y_hat
