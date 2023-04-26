# 3D-reconstruction-of-an-object-from-a-Single-Image-and-a-Text-Prompt
Combining GroundingDINO, Segment Anything, ZoeDepth and Multiview Compressive Coding for 3D reconstruction to reconstruct 3D model of the prompted object from a single image.

git clone btp

git clone https://github.com/facebookresearch/segment-anything.git

git clone https://github.com/IDEA-Research/GroundingDINO.git

git clone https://github.com/facebookresearch/MCC.git

#checkpoint for GroundingDino, put in a new folder checkpoints
mkdir checkpoints
https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth

#checkpoint for SAM, put in checkpoints folder
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

#checkpoint for MCC, put in MCC folder
https://dl.fbaipublicfiles.com/MCC/co3dv2_all_categories.pth

#create new environment

conda create -n btp python=3.9

conda activate btp

conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath


conda install jupyter
pip install scikit-image matplotlib imageio plotly opencv-python

#install pytorch3d from one of the source links
https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#building--installing-from-source

pip install h5py omegaconf submitit

pip install -e GroundingDINO
pip install -e SegmentAnything

#add conda env to jupyter notebook

python -m ipykernel install --user --name=btp

now run jupyter-notebook from the active environment, if pytorch3d fails to load, create a new kernel and choose btp instead of ipython

#put your input image in the input folder, and follow the steps on the notebook to perform the reconstruction
mkdir input
mkdir output

