# MNISST_GAN
Reproduct of GAN for MNIST.
Just for learn the basic knowledge about Generative Adversarial Networks (GAN).

## Results
Epoch 2500 17900 30200 40000

![alt 2500](./images/2500.png) ![alt 17900](./images/17900.png) ![alt 30200](./images/30200.png) ![alt 40000](./images/40000.png)

## Basic Thoughts
GAN has a *generator* generates fake instances from noises, and a *discriminator* 
distinguish the fake instances and real instances.

Training both two networks reach a Nash Equilibrium the GAN model is converged.

## Structure
The generator is a U-Net-like backbone with two downsampling and 
two upsampling. The discriminator is a VGG-like backbone an average pooling after two
convolutions and global average pooling before fully connected.

## Usage
Simple:
```python 
python mnist_gan.py
```

Costume:
```python
python mnist_gan.py \
--learning_rate 0.01 \
--batch_szie 256 \
--epochs 40000 \
--smooth 0.9 \
--dataset_path /path/to/your/MNIST/dir/mnist.npz \
--model_dir /path/to/your/saved/model/dir \
--log_dir /path/to/your/saved/log/dir
``` 

## Tips
This GAN project is build up following these [advices](https://github.com/soumith/ganhacks).