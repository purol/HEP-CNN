# nurion4hep/HEP-CNN
Modified code from HEP-CNN

## How to use
### Making image file(.h5)
In the `scripts` directory of the main directory, you can find `makeDetectorImage_224.py`, `makeDetectorImage_256256.py`, and `makeDetectorImage_280280.py`.
These three codes can make image file(.h5) from the root file. `makeDetectorImage_224.py`, `makeDetectorImage_256256.py`, and `makeDetectorImage_280280.py` can be used for 2x224x224, 4x256x256, and 4x280x280 root file, respectively. You can find more information about it from https://github.com/purol/dual-readout.

Do

    python makeMyDetectorImage_224.py input <directory of input root file> -o <the name of output h5 file> -n <the number of image in each h5 file>

, which would produce h5 files. For instance,

    python makeMyDetectorImage_224.py input ./* -o qqbar_images.h5 -n 1000

would produce `qqbar_images_0.h5`, `qqbar_images_1.h5`, `qqbar_images_2.h5`···.

### Run CNN
In the `run` directory of the main directory, you can find `train_labelByUser_224_single.py`, `train_labelByUser_224_multi.py`, `train_labelByUser_256256_single.py`, and `train_labelByUser_256256_multi.py`. `train_labelByUser_224_single.py` and `train_labelByUser_224_multi.py` can be used for 2x224x224 h5 file. `train_labelByUser_256256_single.py` and `train_labelByUser_256256_multi.py` can be used for 4x256x256 h5 file.

`train_labelByUser_224_single.py` and `train_labelByUser_256256_single.py` can be used if you have only two classes. On the other hand, you should use `train_labelByUser_224_multi.py` or `train_labelByUser_256256_multi.py` if you have 3 or more classes.
