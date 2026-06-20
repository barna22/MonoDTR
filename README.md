# MonoDTR

**MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer** (CVPR 2022) [[paper](https://arxiv.org/abs/2203.10981)]\
Kuan-Chih Huang, Tsung-Han Wu, Hung-Ting Su, Winston H. Hsu.

<img src="resources/example.jpg" alt="vis" style="zoom:50%;" />

## Update 

The code for the KITTI-360 dataset is now available in the [kitti360 branch](https://github.com/KuanchihHuang/MonoDTR/tree/kitti360), and the results can be viewed on the [KITTI-360 leaderboard](https://www.cvlibs.net/datasets/kitti-360/eval_bbox_detect_detail.php?benchmark=bbox3d&result=f5508c2c6753b33341f66f1d965f9df51d8671a1).


## Setup

Please refer to [INSTALL.md](INSTALL.md) for installation and to [DATA.md](DATA.md) for data preparation.

## Starting a training run with the modified configuration

Navigate to the root MonoDTR folder, and activate the Conda environment.
The Conda environment can be accessed by running the following command:
```sh
 conda activate monodtr
```
Once the conda environment is activated, a training run can be started with the following command (if running in a multi-GPU environment like DGX, please make sure that the selected GPU is not already occuppied):
```sh
 ./launcher/train.sh config/config.py $GPU_NUMBER $EXP_NAME
```
Rerunning an existing experiment will work without any additional setup, however, for new experiments, a depth map precomputation needs to be run as well. Refer to [DATA.md](DATA.md) for instructions.

## Train

Move to root and train the network with `$EXP_NAME`:

```sh
 cd #MonoDTR_ROOT
 ./launcher/train.sh config/config.py 0 $EXP_NAME
```

Note: this repo only supports single GPU training. Also, the training randomness for monocular 3D object detection may cause the variance of ±1 AP3D.

## Eval


To evaluate on the validation set using checkpoint `$CHECKPOINT_PATH`:

```sh
 ./launcher/eval.sh config/config.py 0 $CHECKPOINT_PATH validation
```

We provide a good checkpoint for the car category on train/val split [here](https://drive.google.com/file/d/1mQzqjnosra5JE1sJKzSRzdn3R010G0Gh/view?usp=sharing).

## Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{huang2022monodtr,
    author = {Kuan-Chih Huang and Tsung-Han Wu and Hung-Ting Su and Winston H. Hsu},
    title = {MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer},
    booktitle = {CVPR},
    year = {2022}    
}
 ```

## Acknowlegment

Our codes are mainly based on [visualDet3D](https://github.com/Owen-Liuyuxuan/visualDet3D), and also benefits from [CaDDN](https://github.com/TRAILab/CaDDN), [MonoDLE](https://github.com/xinzhuma/monodle), and [LoFTR](https://github.com/zju3dv/LoFTR). Thanks for their contributions!

## License

This project is released under the MIT License.

