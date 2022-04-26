# Extensions to Spline-PINN for HIFU Simulation

The work in this repository is an extension of Spline-PINN, with the aim of adapting it for HIFU simulation. The relevant publications for that work are listed below.

## Related Publications

Comprehensive background information is provided in our paper:  
[Spline-PINN: Approaching PDEs without Data using Fast, Physics-Informed Hermite-Spline CNNs](https://arxiv.org/abs/2109.07143)  
*Nils Wandel, Michael Weinmann, Michael Neidlin, Reinhard Klein*, AAAI, 2022 ([Supplemental video](https://www.youtube.com/watch?v=QC98LCtCZn0), [1 min introduction video](https://www.youtube.com/watch?v=C5IAfCfcyDQ), [20 min presentation](https://www.youtube.com/watch?v=H0g6Tm1zio8))

This physics-*informed* method builds up on previous work that relies on a physics-*constrained* loss based on a finite difference marker and cell grid. Respective publications for 2D and 3D fluid simulations can be found here:

2D: 
[Learning Incompressible Fluid Dynamics from Scratch - Towards Fast, Differentiable Fluid Models that Generalize](https://arxiv.org/abs/2006.08762)  
*Nils Wandel, Michael Weinmann, Reinhard Klein*, ICLR, 2021 ([Code](https://github.com/aschethor/Unsupervised_Deep_Learning_of_Incompressible_Fluid_Dynamics),[Video](https://www.youtube.com/watch?v=EU3YuUNVsXQ),[Spotlight presentation](https://www.youtube.com/watch?v=wIvFkhsIaRA))

3D: 
[Teaching the Incompressible Navier Stokes Equations to Fast Neural Surrogate Models in 3D](https://arxiv.org/abs/2012.11893)  
*Nils Wandel, Michael Weinmann, Reinhard Klein*, Physics of Fluids, 2021 ([Code](https://github.com/aschethor/Teaching_Incompressible_Fluid_Dynamics_to_3D_CNNs),[Video](https://www.youtube.com/watch?v=tKcYJaJtHJE))

## Installation

First, create a new Conda-environment:

```
conda create --name my_cool_fluid_env python=3.7  
conda activate my_cool_fluid_env
```

Now, install the following packages:

```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch  
conda install matplotlib statsmodels natsort tensorboard pyevtk  
pip install opencv-python
```

The installation was tested on Ubuntu 18.04, but other operating systems should work as well. This code works best on machines with CUDA capable GPUs.

### Wave demo

First of all, to see the meaning of all of the script parameters, you can run the following:

```
python wave_train.py --help
```

A pretrained model for the 1D damped wave equation is provided. To start the wave simulation, call:

```
python wave_test.py --net=Wave_model --stiffness=0.1 --damping=0.1 --load_date_time='2022-04-04 16:34:23' --hidden_size=20 --orders_z=2 --load_index=195
```

To update the stiffness throughout the domain and the position of the interface, you can change lines 16-20 in wave\_test.py. You can also change the type of source term on line 36 of wave\_test.py

## Train your own PDE models

You can train a wave model on domains with varying stiffnesses using the following command 

```
python wave_train.py --net=Wave_model --hidden_size=20 --damping=0.1 --loss_v=1 --loss_wave=0.1 --loss_bound=10 --average_sequence_length=200
```

Here, loss_v, loss_wave and loss_bound correspond to alpha, beta and gamma in equation 13 of the paper on Spline-PINNs (Wandel et al. 2022) respectively. 
You can get more information about training parameters and by launching tensorboard using the following commands:

```
python wave_train.py --help  
tensorboard --logdir=Logger/tensorboard
```

## Documentation

Most documentation is auto-generated. Use comments within the wave_train.py and wave_test.py scripts as documentations for those files.
