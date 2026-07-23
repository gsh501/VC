
######额外测试

  CUDA_VISIBLE_DEVICES=0 /home/admin1/anaconda3/envs/gsh/bin/python \
  /home/admin1/gsh/VC/compress_reconstruct/c_r_phase_1.py \
  /home/admin1/Data/data/testdata \
  --filelist-dir /home/admin1/Data/data/testdata_filelists \
  --checkpoint /home/admin1/gsh/VC/pretrained_uf/DCVCUFIntra/4/checkpoint_best_loss_uf_phase_1.pth.tar \
  --output-dir /home/admin1/gsh/VC/compress_reconstruct/codec_outputs/my_phase1_test  \
  --qps 20

  CUDA_VISIBLE_DEVICES=1 /home/admin1/anaconda3/envs/gsh/bin/python \
  /home/admin1/gsh/VC/compress_reconstruct/c_r_phase_1.py \
  /home/admin1/Data/data/vimeo_septuplet \
  --filelist /home/admin1/gsh/VC/datafiles/trian_phase_1/test_filelist.txt \
  --checkpoint /home/admin1/gsh/VC/pretrained_uf/DCVCUF/4/checkpoint_best_loss_uf_phase_2_intra.pth.tar \
  --output-dir /home/admin1/gsh/VC/compress_reconstruct/codec_outputs/vimeo_phase2_test \
  --log-interval 10 \
  --qps 20

  CUDA_VISIBLE_DEVICES=1 /home/admin1/anaconda3/envs/gsh/bin/python \
  /home/admin1/gsh/VC/compress_reconstruct/c_r_phase_1.py \
  /home/admin1/Data/data/vimeo_septuplet \
  --filelist /home/admin1/gsh/VC/datafiles/trian_phase_1/test_filelist.txt \
  --checkpoint /home/admin1/gsh/VC/pretrained_uf/DCVCUFIntra/4/checkpoint_best_loss_uf_phase_1.pth.tar \
  --output-dir /home/admin1/gsh/VC/compress_reconstruct/codec_outputs/vimeo_phase1_test \
  --qps 20

  cd /home/admin1/gsh/VC

CUDA_VISIBLE_DEVICES=7 /home/admin1/anaconda3/envs/gsh/bin/python \
  /home/admin1/gsh/VC/compress_reconstruct/c_r_phase_2.py \
  /home/admin1/Data/data/testdata \
  --filelist-dir /home/admin1/Data/data/testdata_filelists \
  --filelist /home/admin1/gsh/VC/datafiles/train_phase_2/test_filelist_phase2.txt \
  --filelist-root /home/admin1/Data/data/data \
  --dataset-name phase2_test \
  --checkpoint /home/admin1/gsh/VC/pretrained_uf/DCVCUF/4/checkpoint_best_loss_uf_phase_2.pth.tar \
  --output-dir /home/admin1/gsh/VC/compress_reconstruct/codec_outputs/phase2_all_for_plot \
  --qps 20