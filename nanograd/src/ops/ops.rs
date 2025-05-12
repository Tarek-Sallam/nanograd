use crate::ops::types::{GradFn, OpFn};
use crate::tensor::{Tensor, create_tensor};

// Op struct containing the forward and backward function pointers
pub struct Op {
    pub forward: OpFn,
    pub backward: GradFn,
}

// Add forward pass implementation
pub fn add_forward(inputs: &[Tensor]) -> Tensor {
    if inputs.len() != 2 {
        panic!("Add operation requires exactly 2 inputs");
    }
    let a = &inputs[0];
    let b = &inputs[1];

    // Validate shapes match
    if a.shape() != b.shape() {
        panic!(
            "Cannot add tensors with different shapes: {:?} and {:?}",
            a.shape(),
            b.shape()
        );
    }

    let result_data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| x + y)
        .collect();

    create_tensor(result_data, a.shape().to_vec(), true)
}

// Add backward pass implementation
pub fn add_backward(inputs: &[Tensor], grad_output: &Tensor) -> Vec<Tensor> {
    if inputs.len() != 2 {
        panic!("Add operation requires exactly 2 inputs");
    }
    let a = &inputs[0];
    let b = &inputs[1];

    // Validate shapes match
    if a.shape() != b.shape() || a.shape() != grad_output.shape() {
        panic!("Shape mismatch in add backward pass");
    }

    let grad_a = if a.track_grad() {
        grad_output.clone()
    } else {
        create_tensor(vec![], vec![], false)
    };

    let grad_b = if b.track_grad() {
        grad_output.clone()
    } else {
        create_tensor(vec![], vec![], false)
    };

    vec![grad_a, grad_b]
}
