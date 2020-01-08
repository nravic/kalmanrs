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
    pub H: MatrixMN<T, D2, D1>, // Matrix relating state x to measurement z (mxn )asi
    pub I: MatrixMN<T, D1, D1>, // Identity matrix. Stored because of ndarray generic shenanigans
}

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
// Because the macro implements a generic type, and the impl isn't defined in this file, the user has to define the implementation using a wrapper struct.

#[macro_export]
macro_rules! lkf_builder {
    ($lkf_wrapper: ty) => {
        impl $lkf_wrapper {                
            fn predict(&mut self) {
                // projects state ahead 
               self.state.x = self.lk.A * self.state.x + self.lk.B*self.state.u;
                // project error covariance ahead
                self.lk.P = self.lk.A * self.lk.P * self.lk.A.transpose() + self.lk.Q;
            }
            
             fn update(&mut self) {
                // compute Kalman gain
                let mut inner_val = self.lk.H * self.lk.P * self.lk.H.transpose() + self.lk.R;
                inner_val.try_inverse_mut();
                self.lk.K = self.lk.P * self.lk.H.transpose() * inner_val;

                // update estimate with measurement z
                self.state.x = self.state.x + self.lk.K*(self.state.z - self.lk.H * self.state.x);

                // update error covariance
                self.lk.P = (self.lk.I - self.lk.K * self.lk.H) * self.lk.P;
            }
        }
    }
}
