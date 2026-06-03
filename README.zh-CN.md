# ReGenNet：面向人体动作-反应合成

![](./assets/teaser.png)

<p align="left">
  <a href='https://arxiv.org/abs/2403.11882'>
    <img src='https://img.shields.io/badge/Arxiv-2403.11882-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://arxiv.org/pdf/2403.11882.pdf'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
  <a href='https://liangxuy.github.io/ReGenNet/'>
  <img src='https://img.shields.io/badge/Project-Page-pink?style=flat&logo=Google%20chrome&logoColor=pink'></a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=liangxuy.ReGenNet&left_color=gray&right_color=orange">
  </a>
</p>

本仓库包含以下论文对应的代码内容：
> ReGenNet: Towards Human Action-Reaction Synthesis <br>[Liang Xu](https://liangxuy.github.io/)<sup>1,2</sup>, [Yizhou Zhou](https://scholar.google.com/citations?user=dHBNmSkAAAAJ&hl=zh-CN)<sup>3</sup>, [Yichao Yan](https://daodaofr.github.io/)<sup>1</sup>,  [Xin Jin](http://home.ustc.edu.cn/~jinxustc/)<sup>2</sup>, Wenhan Zhu, [Fengyun Rao](https://scholar.google.com/citations?user=38dACd4AAAAJ&hl=en)<sup>3</sup>, [Xiaokang Yang](https://scholar.google.com/citations?user=yDEavdMAAAAJ&hl=zh-CN)<sup>1</sup>, [Wenjun Zeng](https://scholar.google.com/citations?user=_cUfvYQAAAAJ&hl=en)<sup>2</sup><br>
> <sup>1</sup> 上海交通大学 <sup>2</sup> 东方理工高等研究院（宁波） <sup>3</sup> 腾讯微信


## 更新
- [2024.07.14] 发布训练代码、评估代码以及预训练模型。
- [2024.03.18] 发布论文与项目主页。


## 方法框架
![](./assets/framework.png)


## 安装
1. 首先使用以下命令克隆仓库：
    ```
    git clone https://github.com/liangxuy/ReGenNet.git
    cd ReGenNet
    ```

2. 配置运行环境
    <details>
      <summary>1. 使用以下命令创建 conda 环境：</summary>

      * 安装 ffmpeg（如果尚未安装）
        ```
        sudo apt update
        sudo apt install ffmpeg
        ```
      * 创建 conda 环境
        ```
        conda env create -f environment.yml
        conda activate regennet
        python -m spacy download en_core_web_sm
        pip install git+https://github.com/openai/CLIP.git
        ```
      * 安装 mpi4py（多 GPU 训练时需要）
        ```
        sudo apt-get install libopenmpi-dev openmpi-bin
        pip install mpi4py
        ```
    </details>

    如果你希望自行构建 Docker 环境，我们也提供了一个 Dockerfile：`docker/Dockerfile`。

3. 下载其他必需文件

    * 你可以从 [Google Drive](https://drive.google.com/drive/folders/1UtIIB67cZyWAfaw1ZKYjerp_pEAk_XTc?usp=sharing) 下载预训练模型，并将其放到 `save` 文件夹下，以复现实验结果。

    * 你需要从 [Google Drive](https://drive.google.com/drive/folders/1oi1LCNMz3bQoiOktUEeyZRVb6VXEFohP?usp=sharing) 下载动作识别模型，并将其放到 `recognition_training` 文件夹下，用于评估。

    * 从 [SMPL 官网](https://smpl.is.tue.mpg.de/) 下载 SMPL 中性模型，从 [SMPL-X 官网](https://smpl-x.is.tue.mpg.de/) 下载 SMPL-X 模型，然后分别放到 `body_models/smpl` 和 `body_models/smplx`。为方便使用，我们也提供了一份备份文件，见 [这里](https://drive.google.com/drive/folders/1OSLli1j7EBk79tvWk0Ep_adbCM20k8ZV?usp=sharing)。

## 数据准备

### NTU RGB+D 120
由于 NTU RGB+D 120 数据集的许可证不允许我们公开分发其数据和标注，因此我们无法公开发布处理后的 NTU RGB+D 120 数据集。如果你对处理后的数据感兴趣，请发送邮件联系我。

### Chi3D

你可以在 [这里](https://ci3d.imar.ro/download) 下载原始数据集，并在 [这里](https://drive.google.com/file/d/1OvdOGgH1JpVL7viTgPOKHzJSVRB3NczN/view?usp=sharing) 下载 actor-reactor 顺序标注。

你也可以从 [Google Drive](https://drive.google.com/drive/folders/1wPStrZgZaOa42ilADZRvv-_U7gBNjQEr?usp=sharing) 下载处理后的数据集，并将其放到 `dataset/chi3d` 目录下。

### InterHuman

你可以在 [这里](https://tr3e.github.io/intergen-page/) 下载原始数据集，并在 [这里](https://drive.google.com/file/d/10nLfK4uYNblHUhFXKHZWIbFvnRr7G805/view?usp=sharing) 下载 actor-reactor 顺序标注，然后将它们放到 `dataset/interhuman` 目录下。


## 训练
我们提供了在 `NTU120-AS` 数据集上进行人体动作-反应合成的 `online` 和 `unconstrained` 设置下的训练脚本。你也可以通过自定义 `--arch`、`--unconstrained` 和 `--dataset` 来适配不同设置。

* 单 GPU 训练：

  ```
  # NTU RGB+D 120 数据集
  python -m train.train_mdm --setting cmdm --save_dir save/cmdm/ntu_smplx --dataset ntu --cond_mask_prob 0 --num_person 2 --layers 8 --num_frames 60 --arch online --overwrite --pose_rep rot6d --body_model smplx --data_path PATH/TO/xsub.train.h5 --train_platform_type TensorboardPlatform --vel_threshold 0.03 --unconstrained
  ```

  ```
  # Chi3D 数据集
  python -m train.train_mdm --setting cmdm --save_dir save/cmdm/chi3d_smplx --dataset chi3d --cond_mask_prob 0 --num_person 2 --layers 8 --num_frames 150 --arch online --overwrite --pose_rep rot6d --body_model smplx --data_path PATH/TO/chi3d_smplx_train.h5 --train_platform_type TensorboardPlatform --vel_threshold 0.01 --unconstrained
  ```

* 多 GPU 训练（示例使用 4 张 GPU）：

  ```
  mpiexec -n 4 --allow-run-as-root python -m train.train_mdm --setting cmdm --save_dir save/cmdm/ntu_smplx --dataset ntu --cond_mask_prob 0 --num_person 2 --layers 8 --num_frames 60 --arch online --overwrite --pose_rep rot6d --body_model smplx --data_path PATH/TO/xsub.train.h5 --train_platform_type TensorboardPlatform --vel_threshold 0.03 --unconstrained
  ```


## 评估
关于动作识别模型，你可以：

1. 直接从 [这里](https://drive.google.com/drive/folders/1oi1LCNMz3bQoiOktUEeyZRVb6VXEFohP?usp=sharing) 下载训练好的动作识别模型；

2. 或者自行训练动作识别模型：

    动作识别模型的训练代码基于 [ACTOR](https://github.com/Mathux/ACTOR) 仓库。
    <details>
      <summary>训练你自己的动作识别模型的命令：</summary>

      ```python
      cd actor-x;
      # 训练前，你需要先设置好 `dataset` 和 `SMPL-X models` 目录
      ### NTU RGB+D 120 ###
      python -m src.train.train_stgcn --dataset ntu120_2p_smplx --pose_rep rot6d --num_epochs 100 --snapshot 10 --batch_size 64 --lr 0.0001 --num_frames 60 --sampling conseq --sampling_step 1 --glob --translation --folder recognition_training/ntu_smplx --datapath dataset/ntu120/smplx/conditioned/xsub.train.h5 --num_person 2 --body_model smplx

      ### Chi3D ###
      python -m src.train.train_stgcn --dataset chi3d --pose_rep rot6d --num_epochs 100 --snapshot 10 --batch_size 64 --lr 0.0001 --num_frames 150 --sampling conseq --sampling_step 1 --glob --translation --folder recognition_training/chi3d_smplx --datapath dataset/chi3d/smplx/conditioned/chi3d_smplx_train.h5 --num_person 2 --body_model smplx
      ```
    </details>

下面的脚本将评估训练好的模型 `PATH/TO/model_XXXX.pt`，其中 `rec_model_path` 是动作识别模型路径。结果会写入 `PATH/TO/evaluation_results_XXXX_full.yaml`。我们使用 `ddim5` 来加速评估过程。

```
python -m eval.eval_cmdm --model PATH/TO/model_XXXX.pt --eval_mode full --rec_model_path PATH/TO/checkpoint_0100.pth.tar --use_ddim --timestep_respacing ddim5
```

如果你想生成带均值与置信区间的表格，可以使用以下脚本：

```
python -m eval.easy_table PATH/TO/evaluation_results_XXXX_full.yaml
```

## 动作生成与可视化

1. 生成结果，结果将保存到 `results.npy`。

    ```
    python -m sample.cgenerate --model_path PATH/TO/model_XXXX.pt --action_file assets/action_names_XXX.txt --num_repetitions 10 --dataset ntu --body_model smplx --num_person 2 --pose_rep rot6d --data_path PATH/TO/xsub.test.h5 --output_dir XXX
    ```

2. 渲染结果

    安装额外依赖：
    ```
    pip install trimesh
    pip install pyrender
    pip install imageio-ffmpeg
    ```

    ```
    python -m render.crendermotion --data_path PATH/TO/results.npy --num_person 2 --setting cmdm --body_model smplx
    ```

## 待办事项
- [x] 发布训练代码、评估代码与预训练模型。
- [x] 发布标注结果。



## 致谢

感谢以下项目与贡献者，本仓库代码基于它们构建：

[ACTOR](https://github.com/Mathux/ACTOR), [motion diffusion model](https://github.com/GuyTevet/motion-diffusion-model), [guided diffusion](https://github.com/openai/guided-diffusion), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [HumanML3D](https://github.com/EricGuo5513/HumanML3D)


## 许可证
本代码基于 [MIT LICENSE](https://github.com/liangxuy/ReGenNet/blob/main/LICENSE) 分发。

请注意，本项目依赖其他库，包括 CLIP、SMPL、SMPL-X、PyTorch3D，并且使用的数据集也各自具有独立许可证，使用时必须同时遵守这些许可证要求。


## 引用
如果你觉得 ReGenNet 对你的研究有帮助，请引用我们的工作：

```
@inproceedings{xu2024regennet,
  title={ReGenNet: Towards Human Action-Reaction Synthesis},
  author={Xu, Liang and Zhou, Yizhou and Yan, Yichao and Jin, Xin and Zhu, Wenhan and Rao, Fengyun and Yang, Xiaokang and Zeng, Wenjun},
  booktitle={CVPR},
  pages={1759--1769},
  year={2024}
}
```
