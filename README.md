

# 3D Reconstruction of an object from a Single Image and a Text Prompt

This project combines GroundingDINO, Segment Anything, ZoeDepth and Multiview Compressive Coding for 3D reconstruction to reconstruct 3D model of the prompted object from a single image.

## Demo
|Input|Mask|Depth|Output|
|--|--|--|--|
|![Input](https://github.com/AkshitPareek/3D-reconstruction-of-an-object-from-a-Single-Image-and-a-Text-Prompt/blob/daea412dff07e1098786d32dc0dfc9168aff77c6/assets/birdie.jpeg)	`prompt: bird`| ![Mask](https://github.com/AkshitPareek/3D-reconstruction-of-an-object-from-a-Single-Image-and-a-Text-Prompt/blob/9472e5a745f026af084c40e27888c8f4b997bd34/assets/asset1.png) | ![Depth](https://github.com/AkshitPareek/3D-reconstruction-of-an-object-from-a-Single-Image-and-a-Text-Prompt/blob/9472e5a745f026af084c40e27888c8f4b997bd34/assets/asset2.png) |![Output](https://github.com/AkshitPareek/3D-reconstruction-of-an-object-from-a-Single-Image-and-a-Text-Prompt/blob/53faafeb27e40f0a5c8990036ead96df597a2579/assets/birdie.gif)|
|![Input](https://github.com/AkshitPareek/3D-reconstruction-of-an-object-from-a-Single-Image-and-a-Text-Prompt/blob/28be1c05920295b1b86acee2f041463c37df7a52/assets/airplane.png)	`prompt: airplane`|![Mask](https://github.com/AkshitPareek/3D-reconstruction-of-an-object-from-a-Single-Image-and-a-Text-Prompt/blob/28be1c05920295b1b86acee2f041463c37df7a52/assets/airplane_mask.png)|![Depth](https://github.com/AkshitPareek/3D-reconstruction-of-an-object-from-a-Single-Image-and-a-Text-Prompt/blob/28be1c05920295b1b86acee2f041463c37df7a52/assets/airplane_depth.png)|![Output](https://github.com/AkshitPareek/3D-reconstruction-of-an-object-from-a-Single-Image-and-a-Text-Prompt/blob/28be1c05920295b1b86acee2f041463c37df7a52/assets/airplane.gif)|

## Installation

Clone Repositories.

```bash
git clone https://github.com/AkshitPareek/3D-reconstruction-of-an-object-from-a-Single-Image-and-a-Text-Prompt.git

git clone https://github.com/facebookresearch/segment-anything.git

git clone https://github.com/IDEA-Research/GroundingDINO.git

git clone https://github.com/facebookresearch/MCC.git
```

Download Checkpoints.

-   [GroundingDino checkpoint](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth)
-   [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
-   [MCC checkpoint](https://dl.fbaipublicfiles.com/MCC/co3dv2_all_categories.pth)

Place them in a new checkpoints/ folder.

```
mkdir checkpoints
```

Create a new conda environment.

```bash
conda create -n btp python=3.9

conda activate btp
```

Install pytorch3d dependencies:

```
conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia 
conda install -c fvcore -c iopath -c conda-forge fvcore iopath

conda install jupyter 
pip install scikit-image matplotlib imageio plotly opencv-python

```

Build [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#building--installing-from-source) from source.

Install other dependencies.

```bash
pip install h5py omegaconf submitit
```

```
pip install -e GroundingDINO 
pip install -e SegmentAnything
```

Add the conda environment to jupyter notebook:

```bash
python -m ipykernel install --user --name=btp
```

**NOTE:** If pytorch3d gives import module error, choose your environment kernel in jupyter notebook

## Usage

To use this project, you need to put your input image in the input folder.

```bash
mkdir input
```
```
mkdir output
```

You can perform the reconstruction using

```
python reconstruction.py --image_name 'name_of_your_image.extension' --caption 'your_text_prompt'
```

The above script will generate a .html file (output using plotly) and a .npz file (which stores the prediction point arrays)

To generate a .ply output

```
python save_pointcloud.py --image_name 'name_of_your_image' --threshold 'select_between_[0, 1]'
```


## Citation

```
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@inproceedings{ShilongLiu2023GroundingDM,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author={Shilong Liu and Zhaoyang Zeng and Tianhe Ren and Feng Li and Hao Zhang and Jie Yang and Chunyuan Li and Jianwei Yang and Hang Su and Jun Zhu and Lei Zhang},
  year={2023}
}


@misc{bhat2023zoedepth,
  title={ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth},
  author={Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and Müller, Matthias},
  year={2023},
  publisher={arXiv},
  url={https://arxiv.org/abs/2302.12288},
  doi={10.48550/ARXIV.2302.12288},
  keywords={Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences}
}


@article{wu2023multiview,
  author    = {Wu, Chao-Yuan and Johnson, Justin and Malik, Jitendra and Feichtenhofer, Christoph and Gkioxari, Georgia},
  title     = {Multiview Compressive Coding for 3{D} Reconstruction},
  journal   = {arXiv:2301.08247},
  year      = {2023},
}

```
