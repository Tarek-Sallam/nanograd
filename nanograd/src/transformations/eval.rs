use crate::ops::ops::add_impl;
use crate::tensor::{Tensor, TensorKernel};
use std::rc::Rc;

/// Evaluate a tensor, computing its value and returning a new Tensor
pub fn eval(tensor: &Tensor) -> Tensor {
    match &*tensor.0 {
        TensorKernel::Data { .. } => tensor.clone(),
        TensorKernel::Add(a, b) => add_impl(&Tensor(Rc::clone(a)), &Tensor(Rc::clone(b))),
    }
}

/// Get the value at the given index in a tensor
pub fn get(tensor: &Tensor, index: &[usize]) -> f32 {
    let evaluated = eval(tensor);
    let strides = match &*evaluated.0 {
        TensorKernel::Data { strides, .. } => strides.clone(),
        _ => {
            let shape = evaluated.shape();
            let mut strides = vec![1; shape.len()];
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            strides
        }
    };

    let mut flat_index = 0;
    for (i, &idx) in index.iter().enumerate() {
        assert!(idx < evaluated.shape()[i]);
        flat_index += idx * strides[i];
    }
    match &*evaluated.0 {
        TensorKernel::Data { data, .. } => data.data()[flat_index],
        _ => unreachable!(),
    }
}
