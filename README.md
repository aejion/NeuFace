# NeuFace: Realistic 3D Neural Face Rendering from Multi-view Images

###  [Paper]() | [Data]()


Official code for CVPR 2023 paper NeuFace: Realistic 3D Neural Face Rendering from Multi-view Images.

The paper presents a novel 3D face rendering model, namely **NeuFace**, to learn accurate and physically-meaningful underlying 3D representa
tions by neural rendering techniques.
NeuFace naturally incorporates the neural BRDFs into physically based rendering, capturing facial geometry and complex appearance properties in a collaborative manner. NeuFace has a decent generalization ability to common objects as well.


## Installation Requirmenets
The code is compatible with python 3.6.13 and pytorch 1.9.1.
You can create an anaconda environment called `neuface` with the required dependencies by running:

```
conda create -n neuface python==3.6.13
conda activate neuface
pip install -r requirement.txt
```

## Usage
### Data and shape prior
For face dataset, We use the authorized data from 3 individuals for model evaluation from<a href="https://facescape.nju.edu.cn/" target="_blank">FaceScape Dataset</a>. We use the detailed 3D mesh to generate mask of each image's face area, please refer to <a href="https://arxiv.org/abs/2203.14510" target="_blank">ImFace</a> to for more information about the data-preprocessing of 3D mesh.
For common objects, we use the <a href="https://www.dropbox.com/sh/5tam07ai8ch90pf/AADniBT3dmAexvm_J1oL__uoaDTU" target="_blank"> DTU dataset</a> for model evaluation.
As ImFace serves as the shape prior, the pretrained model can be download at <a href="" target="_blank">pretrained-model</a>
### Train on Facescape
For training NeuFace on Facescape dataset, run:
```
python scripts/train_pl.py
```
Please check pathes in your config file are both correct. Results can be found in `{out_dir}/{expname}`.
Our trained model can be downloaded as follow:

| Trained Model            | Description  |
|-------------------|-------------------------------------------------------------|
| <a href="" target="_blank">NeuFace_212_id_1_exp</a> | train on 212 id with neutral expression of Facescape dataset|

### Evaluation on Facescape
For evaluating the novel view metrics, run:
```
 python scripts/eval_pl.py --ckpt [ckpt_path] --out_dir [our_dir]
```
Results can be found in `{our_dir}/test/{expname}`.

### Train on DTU
For training NeuFace on DTU dataset, run:
```
cd common_object
python training/exp_runner.py --conf ./confs/dtu_fixed_cameras.conf --scan_id [scan_id] --gpu [GPU_ID]
```
Please check [dataset.data_dir] in your config file are both correct. Results can be found in `common_object/exps/{train.expname}/{timestamp}`.
Our trained model can be downloaded as follow:

| Trained Model            | Description  |
|-------------------|-------------------------------------------------------------|
| <a href="" target="_blank">NeuFace_DTU_65</a> | train on 65 scan of Facescape dataset|

### Evaluation on DTU
For evaluating the training view metrics, run:
```
cd common_object
python evaluation/eval.py  --conf ./confs/dtu_fixed_cameras.conf --scan_id [SCAN_ID] --eval_rendering --gpu 0
```
Results can be found in `common_object/evals/{train.expname}/rendering`..
## Citation
If you find our work useful in your research, please consider citing:

	@inproceedings{
	}

