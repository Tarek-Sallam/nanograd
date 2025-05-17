use crate::tensor::{Tensor, TensorKernel};
use crate::transformations::eval::eval;
use std::rc::Rc;

/// Internal implementation of tensor addition
pub(crate) fn add_impl(a: &Tensor, b: &Tensor) -> Tensor {
    let a_eval = eval(a);
    let b_eval = eval(b);
    let a_shape = a_eval.shape();
    let a_data = match &*a_eval.0 {
        TensorKernel::Data { data, .. } => data.data(),
        _ => unreachable!(),
    };
    let b_data = match &*b_eval.0 {
        TensorKernel::Data { data, .. } => data.data(),
        _ => unreachable!(),
    };
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x + y)
        .collect();
    Tensor::new(a_shape, result)
}
