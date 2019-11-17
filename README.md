# {J, T}-Linkage for Vanishing Point Estimation

This is an implementation of J-Linkage [[1]](#references) and T-Linkage [[2]](#references) for vanishing point 
estimation from line segments extracted via LSD [[3]](#references). 

### Setup
Assuming that you are using Anaconda.

Get the code:
 ```
 git clone --recurse-submodules https://github.com/fkluger/vp-linkage.git
 cd vp-linkage
 ```
Prepare environment:
```
conda env create -f environment.yml
source activate vp_linkage
cd lsd
python setup.py build_ext --inplace
cd ..
```

Download the [York Urban Dataset (YUD)](http://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/).

### Run
To compute the AUC metric over the YUD test set, run:
```
python linkage.py --dataset_path PATH_TO_YUD
```
Add the option ```--tlinkage``` in order to switch from J-Linkage to T-Linkage. 
See ```python linkage.py --help``` for available options.

### References
[1] Roberto Toldo and Andrea Fusiello. 
Robust multiple structures estimation with j-linkage. ECCV 2008.

[2] Luca Magri and Andrea Fusiello. 
T-linkage: A continuous relaxation of j-linkage for multi-model fitting. CVPR 2014.

[3] Rafael Grompone Von Gioi, Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall. 
Lsd: A fast line segment detector with a false detection control. TPAMI 2008.
