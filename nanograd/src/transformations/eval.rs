use crate::ops::ops::add_impl;
use crate::tensor::{Tensor, TensorKernel};
use std::rc::Rc;

/// Evaluate a tensor, computing its value
pub fn eval(tensor: &Tensor) -> Vec<f32> {
    match &*tensor.0 {
        TensorKernel::Data { data, .. } => data.data().to_vec(),
        TensorKernel::Add(a, b) => add_impl(&Tensor(Rc::clone(a)), &Tensor(Rc::clone(b))),
    }
}

/// Get the value at the given index in a tensor
pub fn get(tensor: &Tensor, index: &[usize]) -> f32 {
    let data = eval(tensor);
    let strides = match &*tensor.0 {
        TensorKernel::Data { strides, .. } => strides.clone(),
        _ => {
            let shape = tensor.shape();
            let mut strides = vec![1; shape.len()];
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            strides
        }
    };

    let mut flat_index = 0;
    for (i, &idx) in index.iter().enumerate() {
        assert!(idx < tensor.shape()[i]);
        flat_index += idx * strides[i];
    }
    data[flat_index]
}
