# nurion4hep/HEP-CNN
Modified code from HEP-CNN

## How to use
### Making image file(.h5)
In the `scripts` directory of the main directory, you can find `makeDetectorImage_224.py`, `makeDetectorImage_256256.py`, and `makeDetectorImage_280280.py`.
These three codes can make image file(.h5) from the root file. `makeDetectorImage_224.py`, `makeDetectorImage_256256.py`, and `makeDetectorImage_280280.py` can be used for 2x224x224, 4x256x256, and 4x280x280 root file, respectively. You can find more information about it from https://github.com/purol/dual-readout.

Do

    python <the name of python program> input <directory of input root file> -o <the name of output h5 file> -n <the number of event in each h5 file>

, which would produce h5 files. For instance,

    python makeMyDetectorImage_224.py input ./* -o qqbar_images.h5 -n 1000

would produce `qqbar_images_0.h5`, `qqbar_images_1.h5`, `qqbar_images_2.h5`···.

### Running CNN
In the `run` directory of the main directory, you can find `train_labelByUser_224_single.py`, `train_labelByUser_224_multi.py`, `train_labelByUser_256256_single.py`, and `train_labelByUser_256256_multi.py`.

`train_labelByUser_224_single.py` and `train_labelByUser_224_multi.py` can be used for 2x224x224 h5 file. `train_labelByUser_256256_single.py` and `train_labelByUser_256256_multi.py` can be used for 4x256x256 h5 file.

`train_labelByUser_224_single.py` and `train_labelByUser_256256_single.py` can be used if you have only two classes. On the other hand, you should use `train_labelByUser_224_multi.py` or `train_labelByUser_256256_multi.py` if you have 3 or more classes.

You also can find `config_multi.yaml` and `config_single.yaml` files in the `run` directory. It is configuration file when you run CNN. Inside the `config_single.yaml`, you can find

    samples:
      - name: Z2tau2pipizeropizeronu
        label: 1
        path: ../data/Z2pipizeropizeronu_images/*.h5
        xsec: 1
        ngen: 1
      - name: Z2tau2qqbar
        label: 0
        path: ../data/Z2qqbar_images/*.h5
        xsec: 1
        ngen: 1
    training:
        randomSeed1: 123456
        nDataLoaders: 4
. `path` is the path of the h5 file. I do not care about `xsec` and `ngen`. I just set them to be one. In this case, weights for each events becomes 1. If you want to change the weight, you need to change `xsec` or `ngen`.

Do

    python <the name of python program> --batch <batch number> --device <device number> --model <the name of model you use> -o <path of output files> -c <path of configuration file> --epoch <the number of epoch>

, which train CNN and produce output file. The name of model can be found in `train_labelByUser` code.

For instance,

    python train_labelByUser_256256_multi.py --batch 32 --device 1 --model defaultnorm1 -o ../result/ -c ./config_multi.yaml --epoch 50

, which would train CNN. It also makes `history_0.csv`, `model.pth`, `resourceByCP_0.csv`, `resourceByTime_0.csv`, `summary.txt`, and `weight_0.pth`.

`history_0.csv` saves loss and accuracy. `model.pth` and `weight_0.pth` saves the trained CNN. `summary.txt` saves a configuration of CNN. 

### Making a plot about testing sample
We need to obtain the performance of trained CNN, using testing model.
In the `run` directory of the main directory, you can find `eval_torch_single.py` and `eval_torch_multi.py`. `eval_torch_single.py` is used when the number of classes is 2. `eval_torch_multi.py` is used when the number of classes is 3 or larger than 3.

Do

    python <the name of program> --input <path of model.pth and weight_0.pth> -c <path of config file> --batch <batch number> --device <device number>

For instance,

    python eval_torch_multi.py --input ../result/CPlargekernelnorm1_multi_input_again/ -c ./config_multi.yaml --batch 32 --device 0

, which would evaluate a trained CNN in the `../result/CPlargekernelnorm1_multi_input_again/` directory, using `./config_multi.yaml`.

If you use `eval_torch_single.py`, you can obtain images of CNN response and roc curve:
![Alt text](/img/response.png "response")
![Alt text](/img/roc.png "roc")

On the other hand, you can obtain a confusion matrix if you use `eval_torch_multi.py`:
![Alt text](/img/conf2.png "conf")

### Making a plot of loss and accuracy
You can make a plot by using `history_0.csv`. You can find `drawLossCurve.py` in the `run` directory. 

Do

    python drawLossCurve.py <path of history.csv file>
    
For instance

    python drawLossCurve.py ../result/defaultnorm1_no_strange_norm_multi/history_0.csv

, which would make a plot about loss and accuracy:
![Alt text](/img/loss.png "loss")

### Making a plot of h5 file and root file
You can train CNN and estimate it with above steps. However, you sometimes want to inspect the h5 or root file.

You can find `view_h5.py` in the `script` directory of the main directory. That code makes you to inspect h5 file.

Do

    python view_h5.py <name of h5 file>
    
, which would produce png image.

For instance,

    python view_h5.py qqbar_images_0.h5
    
would make a visible png file:
![Alt text](/img/view.png "view")

`show_rootfile_224.py`, `show_rootfile_256.py`, and `show_rootfile_280.py` make png image file, using root file.

`show_rootfile_224.py`, `show_rootfile_256.py`, and `show_rootfile_280.py` is used for 2x224x224, 4x256x256, and 4x280x280 root file, respectively. You can find more information about it from https://github.com/purol/dual-readout.

Do

    python <name of program> <path of root file>
    
, which would produce png image.

For instance,

    python show_rootfile_224.py ./20210520_Z2tau2pipipipizero_CNN_out_160.root
    
