#1.从github下下载相关代码
网址：git@github.com:gsh501/VC.git

更改提交：
git add .
git commit -m "×××××"
git push origin

下载：
git status -sb
git pull --ff-only origin gsh
git log --oneline -5
git status -sb


#2.配置环境，在dcvc-rt环境基础上加入imageio
pip install imageio


#3.第一阶段训练（8帧训练DCVCUFIntra）：采用Partvimeo_7数据进行
##3.1划分数据集make_filelist_phase1.py
python /home/admin1/gsh/VC/tools/make_filelist_phase1.py

##3.2开始训练第一阶段train_phase_1.py
Windows:
smoke test:
python .\train_uf\train_phase_1.py `
  --train-root D:\Drivers\Desktop\VC-gsh\Partvimeo_32\sequences `
  --train-filelist D:\Drivers\Desktop\VC-gsh\Partvimeo_32\train_filelist.txt `
  --val-dataset D:\Drivers\Desktop\VC-gsh\Partvimeo_32\sequences `
  --val-filelist D:\Drivers\Desktop\VC-gsh\Partvimeo_32\val_filelist.txt `
  --epochs 4 --batch-size 1 --val-batch-size 1 `
  --num-workers 0 --max-samples 5 --log-interval 1

Ubuntu：
tmux ls
tumx new -s gsh         #创建gsh的会话
tmux attach -t gsh      #接入gsh的会话
tmux detach -t gsh      #终端直接分离会话（后台挂起，退出 SSH 不终止程序）    
按下 Ctrl + b松开两个键，再按d屏幕会提示类似：[detached (from session ...)]然后你就回到了普通的 shell，而训练仍然在后台继续运行。
tmux kill-session -t gsh   # 删除gsh会话
tmux rename-session -t 旧名 新名   #重命名会话

cd /home/admin1/gsh/VC

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/admin1/anaconda3/envs/dcvc/bin/python -m torch.distributed.run \
  --nproc_per_node=4 \
  train_uf/train_phase_1.py \
  --train-filelist /home/admin1/Data/data/vimeo_septuplet/train_filelist.txt \
  --val-filelist /home/admin1/Data/data/vimeo_septuplet/val_filelist.txt \
  --output-dir ./pretrained_uf \
  --quality-level 4 \
  --epochs 120 \
  --batch-size 4 \
  --val-batch-size 1 \
  --num-workers 4

##3.3测试第一阶段训练结果test_phase_1.py

cd /home/admin1/gsh/VC

/home/admin1/anaconda3/envs/dcvc/bin/python test_uf/test_phase_1.py \
  --checkpoint /home/admin1/gsh/VC/pretrained_uf/DCVCUFIntra/2/checkpoint_best_loss_uf_phase_1.pth.tar \
  --device cuda:5 \
  --qps 20


#4.第二阶段训练（32帧训练DCVCUF）
##4.1检查第一阶段保存训练模型VC-gsh\pretrained_uf\DCVCUFIntra\2\checkpoint_best_loss_uf_phase_1.pth.tar

##4.2划分数据集make_filelist_phase2.py
python /home/admin1/gsh/VC/tools/make_filelist_phase2.py

##4.3开始训练第二阶段train_phase_2.py
Windows:
smoke test:
python train_uf\train_phase_2.py `
  --train-filelist D:\Drivers\Desktop\VC-gsh\Partvimeo_32\train_filelist_phase2.txt `
  -td_l D:\Drivers\Desktop\VC-gsh\Partvimeo_32\val_filelist_phase2.txt `
  --epochs 5 `
  --batch-size 1 `
  --test-batch-size 1 `
  --num-workers 0 `
  --patch-size 128 128 `
  --log-interval 1

Ubuntu：
cd /home/admin1/gsh/VC

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_uf/train_phase_2.py \
  --phase1-checkpoint /home/admin1/gsh/VC/pretrained_uf/DCVCUFIntra/2/checkpoint_best_loss_uf_phase_1.pth.tar \
  --train-filelist /home/admin1/gsh/VC/datafiles/train_phase_2/train_filelist_phase2.txt \
  --val-filelist /home/admin1/gsh/VC/datafiles/train_phase_2/val_filelist_phase2.txt \
  --output-dir /home/admin1/gsh/VC/pretrained_uf \
  --quality-level 4 \
  --epochs 180 \
  --batch-size 4 \
  --val-batch-size 1 \
  --num-workers 4 \
  --patch-size 256 256 \
  --log-interval 500
  
##4.4测试第二阶段训练结果test_phase_2.py
cd /home/admin1/gsh/VC

联合权重测试：
/home/admin1/anaconda3/envs/dcvc/bin/python test_uf/test_phase_2.py \
  --checkpoint /home/admin1/gsh/VC/pretrained_uf/DCVCUF/4/checkpoint_best_loss_uf_phase_2.pth.tar \
  --device cuda:4 \
  --qps 20

分离权重测试：
/home/admin1/anaconda3/envs/dcvc/bin/python test_uf/test_phase_2.py \
  --video-checkpoint /home/admin1/gsh/VC/pretrained_uf/DCVCUF/4/checkpoint_best_loss_uf_phase_2_video.pth.tar \
  --intra-checkpoint /home/admin1/gsh/VC/pretrained_uf/DCVCUF/4/checkpoint_best_loss_uf_phase_2_intra.pth.tar \
  --device cuda:4 \
  --qps 20