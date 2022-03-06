# Social-Implicit

## General 
- On the first run of testing or training code, the code will save a pickled version of the processed data for faster production in the next time. 

- We used pipreqs to generate the minimum needed dependcies ro tun the code. The necessary packages are in requirements.txt, you can install it by running:

```
pip3 install -r requirements.txt
```
## Code Structure
- train.py for training the code
- test_amd_amv_kde.py to test the models and report the AMD/AMV/KDE metrics
- test_ade_fde.py to test the models and report the ADE/FDE metrics 
- CFG.py contains model configratuin parameters 
- metrics.py contains ADE/FDE implementation 
- amd_amv_kde.py contains the AMD/AMV/KDE implementation 
- trajectory_augmenter.py contains the augmentations techniques for trajectories 
- utils.py general utils used by the code 
- pkls folder: stores the data pickles 
- datasets folder: contains the ETH/UCY raw data
- checkpoint folder: contains the trained models

## Testing using pretarined models
### To report the AMD/AMV/KDE 
```
python3 test_amd_amv_kde.py
```
Note that the code will take a while to run becuase the GMM fit version we use is not vecotrized version. 
### To report the ADE/FDE
```
python3 test_ade_fde.py
```

## Training from scratch 
Simply, run: 
```
train.sh
```
## Visualization 
The visualization script compares our model with prior models. 
The visualization data is precomputed and can be downloaded using
```
Visualization/downloadVisualData.sh
```
Then you will need to run the notebook
```
Align.ipynb
```
In order to generate a suitable visualization pkls aligning the raw outputs of the models. Aftewards, 
you can visualize everything using either the zoomed or the aspect ratio constarined visualization notebooks. 
```
Visualize.ipynb
VisualizeZoomed.ipynb
```

