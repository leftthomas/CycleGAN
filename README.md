# CycleGAN

A PyTorch implementation of CycleGAN based on ICCV 2017 paper [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593).

![Network Architecture](result/structure.png)

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch
```

## Dataset

[monet2photo](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip)
dataset is used in this repo, you could download this dataset from official website. The data 
directory structure is shown as follows:

 ```
├──data_root
   ├── train
       ├── A (domain A images)
       ├── B (domain B images) 
   ├── test
       ├── A (domain A images)
       ├── B (domain B images) 
```

## Usage

```
python main.py --epochs 300 --lr 0.0001
optional arguments:
--data_root                   Datasets root path [default value is 'monet2photo']
--batch_size                  Number of images in each mini-batch [default value is 1]
--epochs                      Number of epochs over the data to train [default value is 200]
--lr                          Initial learning rate [default value is 0.0002]
--decay                       Epoch to start linearly decaying lr to 0 [default value is 100]
--save_root                   Result saved root path [default value is 'result']
```

## Results

The model is trained on one NVIDIA GTX TITAN (12G) GPU. Here are some examples, on the left is the original image, and
on the right is the generated image.

![vis](result/vis.png)