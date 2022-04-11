# Presentation
This repository contains the code of our [paper](https://arxiv.org/abs/2201.01047): *DIAL: Deep Interactive and Active Learning for Semantic Segmentation in Remote Sensing.*

# To use

## Prepare data
The training datasets should be stored in a folder *MyDataset* organized as follows:
 - a folder named `imgs` containing the RGB images.
 - a folder named `gts` containing the ground-truths.

:warning: Ground-truth files must have the same names than their associated image.

#### Example for [ISPRS Potsdam dataset](http://www2.isprs.org/commissions/comm3/wg4/data-request-form2.html).

```Shell
cd <PotsdamDataset>
sudo apt install rename
cd gts; rename 's/_label//' *; cd ../imgs; rename 's/_RGB//' *
```
The ground-truth maps have to be one-hot encoded (i.e. not in a RGB format):
```Shell
cd ICSS
python preprocess/format_gt.py -n 6 -d <PathToMyDataset>/gts
```

## To train:
All parameters (including the activation of DISIR and DISCA) can be set in a config file similar to `configs/some_config.yml`.

In `src/train.py`: Train a model on the train set and test it on the evaluation set (with N clicks simulations if DISIR is enabled). It is possible to skip the training with a pretrained model.


## Active learning
### Pixelwise AL
```Shell
python -m src.active_learning.pixelwise_al -d /data/gaston/Potsdam -c configs/some_config.yml  -p data/models/my_model.pt
```

### Patchwise AL
```Shell
python -m src.active_learning.patchwise_al -d /data/gaston/Potsdam -c configs/some_config.yml  -p data/models/my_model.pt
```
# Licence

Code is released under the MIT license for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

See [LICENSE](./LICENSE) for more details.

# Acknowledgements

This work has been jointly conducted at [Alteia](https://alteia.com/)  and [ONERA-DTIS](https://www.onera.fr/en/dtis).
