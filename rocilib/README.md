# `rocilib`

`rocilib` aims to be the Keras of computational imaging.


Core abstractions:
1. `Observation`: Observed sample
2. `ForwardModel`: Parametric model of the sensing apparatus. Represents the generative process of the `Observation` from the underlying latent parameters. Probabilistic, differentiable, and composable.
3. `InverseInferer`: Inference algorithm for inverting the `Observation` through the `ForwardModel` to obtain the latent parameters.


Usage:
1. Define the forward model
2. Define and initialize the observation
3. Invert


Additional features:
- Numerical tools to analyze the forward model's theoretical properties, e.g., identifiability and estimability
- Active sampler based on a feedback loop