# {J, T}-Linkage for Vanishing Point Estimation

This is an implementation of J-Linkage [[1]](#references) and T-Linkage [[2]](#references) for vanishing point 
estimation from line segments extracted via LSD [[3]](#references). 

This implementation was used in our CONSAC paper [[4]](#references), so please cite the paper if you use this code:
```
@inproceedings{kluger2020consac,
  title={CONSAC: Robust Multi-Model Fitting by Conditional Sample Consensus},
  author={Kluger, Florian and Brachmann, Eric and Ackermann, Hanno and Rother, Carsten and Yang, Michael Ying and Rosenhahn, Bodo},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
``` 

## Setup
Assuming that you are using Anaconda.

Get the code:
 ```
 git clone --recurse-submodules https://github.com/fkluger/vp-linkage.git
 cd vp-linkage
 git submodule update --init --recursive
 ```
Prepare environment:
```
conda env create -f environment.yml
source activate vp_linkage
cd datasets/nyu_vp/lsd
python setup.py build_ext --inplace
cd ../../yud_plus/lsd
python setup.py build_ext --inplace
cd ../../..
```

## Datasets
### NYU-VP
The vanishing point labels and pre-extracted line segments for the 
[NYU dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) are fetched automatically via the *nyu_vp* 
submodule. In order to use the original RGB images as well, you need to obtain the original 
[dataset MAT-file](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) and convert it to a 
*version 7* MAT-file in MATLAB so that we can load it via scipy:
```
load('nyu_depth_v2_labeled.mat')
save('nyu_depth_v2_labeled.v7.mat','-v7')
```

### YUD and YUD+
Pre-extracted line segments and VP labels are fetched automatically via the *yud_plus* submodule. RGB images and camera 
calibration parameters, however, are not included. Download the original York Urban Dataset from the 
[Elder Laboratory's website](http://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/) and 
store it under the ```datasets/yud_plus/data``` subfolder. 

## Demo
...coming soon(ish)

## Run
To compute the AUC metric over the YUD test set, run:
```
python linkage.py --dataset yud --dataset_path ./datasets/yud_plus/data/ 
```
For YUD+:
```
python linkage.py --dataset yud+ --dataset_path ./datasets/yud_plus/data/ 
```
For NYU-VP:
```
python linkage.py --dataset nyu --dataset_path ./datasets/nyu_vp/data/ --mat_file_path nyu_depth_v2_labeled.v7.mat
```

Add the option ```--tlinkage``` in order to switch from J-Linkage to T-Linkage. 
See ```python linkage.py --help``` for available options.

## References
[1] Roberto Toldo and Andrea Fusiello. 
Robust multiple structures estimation with j-linkage. ECCV 2008.

[2] Luca Magri and Andrea Fusiello. 
T-linkage: A continuous relaxation of j-linkage for multi-model fitting. CVPR 2014.

[3] Rafael Grompone Von Gioi, Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall. 
Lsd: A fast line segment detector with a false detection control. TPAMI 2008.

[4] Florian Kluger, Eric Brachmann, Hanno Ackermann, Carsten Rother, Michael Ying Yang, and Bodo Rosenhahn. 
CONSAC: Robust Multi-Model Fitting by Conditional Sample Consensus. CVPR 2020