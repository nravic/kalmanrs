#![allow(non_snake_case)] // rust complains about capitalized matrix names 

extern crate nalgebra as na;
use na::{DefaultAllocator, Scalar, Dim, VectorN, MatrixMN};
use na::allocator::Allocator;

// Kalman Filters in Rust.
// NB: Until const generics are a thing in Rust (see: https://github.com/rust-lang/rust/issues/44580), I'm forced to use a pretty ugly way of making the Kalman state structs generic. 

#[derive(Debug)]
pub struct LinearKalman<T: Scalar, D1: Dim, D2: Dim, D3: Dim> // (d1: n, d2: m, d3: l)
where DefaultAllocator: Allocator<T, D1, D1> // allocator for nxn matrix
    + Allocator<T, D2, D1> // allocator for (mxn) matrix
    + Allocator<T, D1, D2> // allocator for (nxm) matrix 
    + Allocator<T, D1, D3> { // allocator for (nxl) matrix  
    pub Q: MatrixMN<T, D1, D1>, // Process noise covariance matrix (nxn)
    pub A: MatrixMN<T, D1, D1>, // Prev state -> current state mapping matrix (nxn)
    pub B: MatrixMN<T, D1, D3>, // Control input -> state mapping matrix (nxl)
    pub P: MatrixMN<T, D1, D1>, // Estimate error covariance matrix
    pub K: MatrixMN<T, D1, D2>, // Kalman gain matrix (nxm)
    pub R: MatrixMN<T, D1, D1>, // Measurement noise covariance matrix
    pub H: MatrixMN<T, D2, D1>, // Matrix relating state x to measurement z (mxn) 
    pub I: MatrixMN<T, D1, D1>, // Identity matrix. Stored because of ndarray generic shenanigans
}

// #[derive(Debug)]
// struct UnscentedKalman<T: Scalar, D: Dim>
// where DefaultAllocator: Allocator<T, D, D> {}

#[derive(Debug)]
pub struct KalmanState<T: Scalar, D1: Dim, D2: Dim, D3: Dim>
where DefaultAllocator: Allocator<T, D1>
    + Allocator<T, D2>
    + Allocator<T, D3> {
    pub x: VectorN<T, D1>, // state vector (nx1)
    pub z: VectorN<T, D2>, // measurement vector (mx1)
    pub u: VectorN<T, D3>, // control input vector (lx1) 
}

// -- Macro to build the Linear Kalman Filter -- 
// Because the macro implements a generic type, and the impl isn't defined in this file, the user has to define the implementation. An issue then arises because in Rust it is not possible to impl a type outside of where it is defined.
// $lkf_struct and $lkf_state hence are structs that need to be created by the user that respectively encapsulate the LinearKalman and KalmanState structs.

#[macro_export]
    macro_rules! lkf_tuple {
    ($lkf_struct: ty, $lkf_state: ty, $scalar: ty, $dim_n: ty, $dim_m: ty, $dim_l: ty) => {
        struct Filter($lkf_struct<$scalar, $dim_n, $dim_m, $dim_l>,
            $lkf_state<$scalar, $dim_n, $dim_m, $dim_l>);
    }
}

#[macro_export]
macro_rules! lkf_builder {
    ($lkf_struct: ty, $lkf_state: ty, $scalar: ty, $dim_n: ty, $dim_m: ty, $dim_l: ty) => {
        impl $lkf_struct<$scalar, $dim_n, $dim_m, $dim_l> {
                
            fn predict(&mut self, state: &mut $kstate<$scalar, $dim_n, $dim_m, $dim_l>) {
                // projects state ahead 
               state.x = self.A * state.x + self.B*state.u;
                // project error covariance ahead
                self.P = self.A * self.P * self.A.transpose() + self.Q;
            }
            
            fn update(&mut self, state: &mut $kstate<$scalar, $dim_n, $dim_m, $dim_l>) {
                // compute Kalman gain
                let mut inner_val = self.H * self.P * self.H.transpose() + self.R;
                inner_val.try_inverse_mut();
                self.K = self.P * self.H.transpose() * inner_val;

                // update estimate with measurement z
                state.x = state.x + self.K*(state.z - self.H * state.x);

                // update error covariance
                self.P = (self.I - self.K * self.H) * self.P;
            }
        }
    }
}
