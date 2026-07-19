#1.从github下下载相关代码
网址：git@github.com:gsh501/VC.git

#2.配置环境，在dcvc-rt环境基础上加入imageio
pip install imageio

#3.第一阶段训练（8帧训练DCVCUFIntra）：采用Partvimeo_7数据进行
##3.1划分数据集make_filelist_phase1.py

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


#4.第二阶段训练（32帧训练DCVCUF）
##4.1检查第一阶段保存训练模型VC-gsh\pretrained_uf\DCVCUFIntra\2\checkpoint_best_loss_uf_phase_1.pth.tar

##4.2划分数据集make_filelist_phase2.py

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


  