# AffordanceVAED

## [Affordance Learning for End-to-End Visuomotor Robot Control](TODO)

In [Affordance Learning for End-to-End Visuomotor Robot Control](TODO), we introduced a modular deep neural network structure, 
that detects a container on a table, and inserts a ball into it.

We showed that our system performs its task successfully in zero-shot sim-to-real transfer manner.
Each part of our system was fully trained either with synthetic data or in a simulation.
The system was invariant to, e.g., distractor objects and textures.

We have didvided our work into the following code blocks:

* [AffordanceVAED](https://github.com/gamleksi/AffordanceVAED) extracts affordance information from an observation image, and represents it as a latent space vector. 
Figure 1 shows the structure of the model.
* [BlenderDomainRandomizer](https://github.com/gamleksi/BlenderDomainRandomizer) generates  a domain randomized dataset for VAED.
* [TrajectoryVAE](https://github.com/gamleksi/TrajectoryVAE) represents trajectories in a low-dimensional latent space, and generates a trajectory based on a given latent vector.
* [affordance_gym](https://github.com/gamleksi/affordance_gym) generates training data for TrajectoryVAE, and combines VAED and TrajectoryVAE together to perform desired trajectories based on an observation.

## Model

This reprository inroduces the variational affordance encoder-decoder structure:

![the VAED structure](images/affo.png?raw=true)

## Setup

Install required depedencies To install ```pip install -r requirements.txt```.

## Blender VAED

1) Generate training data with [BlenderDomainRandomizer](https://github.com/gamleksi/BlenderDomainRandomizer).
3) Run ```python blender_main.py -h``` to see how to include the generated training data and rest of the options.

## UMD VAED

1) Load [the UMD-dataset](http://users.umiacs.umd.edu/~amyers/part-affordance-dataset/)
2) Run ```python umd_main.py -h``` to see how to include the loaded UMD dataset.
