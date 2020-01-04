 // examples/lkf.rs
// Simple example for instantiating and using a Linear Kalman Filter
extern crate kalmanrs as krs;
extern crate nalgebra as na;   
use krs::{KalmanState, LinearKalman, lkf_builder};
use na::{U3, Matrix3, Vector3};

struct KalmanWrapper {
    lk: LinearKalman<f64, U3, U3, U3>, 
    state: KalmanState<f64, U3, U3, U3>
}
 
lkf_builder! {KalmanWrapper}

fn main() {
    let example_kalman = LinearKalman {
        Q: Matrix3::new_random(),
        A: Matrix3::new_random(),
        B: Matrix3::new_random(),
        P: Matrix3::new_random(),
        K: Matrix3::new_random(),
        R: Matrix3::new_random(),
        H: Matrix3::new_random(),
        I: Matrix3::new_random()
    };
 
    let kalman_state = KalmanState {
        x: Vector3::new_random(),
        u: Vector3::new_random(),
        z: Vector3::new_random()                            
    };

    let mut filter = KalmanWrapper {
        lk: example_kalman,
        state: kalman_state
    };

    let _dt = 0.1; 
    let t_max = 30;
    for _t in (0..t_max).step_by(1) {
        filter.predict();
        filter.update();
        println!("{:?}", filter.state)
    };
}
