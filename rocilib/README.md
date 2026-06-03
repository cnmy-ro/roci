# `rocilib`

`rocilib` aims to be the Keras of computational imaging.


Core constructs:
1. `Representation`: observations and latents
2. `ForwardModel`: Model of the sensor, represents the data generation process of the observation `Representation` from the underlying latent `Representation`
3. `InverseInferer`: Inference algorithm for inverting the observation `Representation` through the `ForwardModel` to obtain the latent `Representation`
4. `ActiveSampler`: Feedback from latents to determine how to sample the next observation.


Usage:
1. Define the representation and the forward model
2. Invert