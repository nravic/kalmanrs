# kalmanrs 

This library directly implements the algorithm found [here](http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf). kalmanrs is build using nalgebra, and is designed to be as dimensionally generic as possible to make it easy to implement in whatever control structure or system you're working with. 

## Setup and Usage
Add `kalmanrs` to your Cargo.toml under dependencies.

The design of `kalmanrs` necessitates creating a wrapper for the two structs that comprise a Kalman Filter in `kalmanrs`, `LinearKalman` and `KalmanState`. What allows for dimensional genericity is the macro `kalmanrs` supplies, `lkf_builder`; which implements the predict and update methods for your Kalman Filter. 

A self-explanatory example is detailed in `examples/lkf.rs`. An important note to make is that because rust currently does not support parametrization over integer values, dimensions are simulated using types; which are defined in the root module of nalgebra. If you wanted to use a 2x3 dimensional matrix for example, you would have to `use na::{U2, U3}`. 

## Todo
- Add the Extended Kalman Filter (EKF).
- Add the Unscented Kalman Filter (UKF). 
- Better testing/test coverage


