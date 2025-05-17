use crate::tensor::{Tensor, TensorKernel};
use crate::transformations::eval::eval;
use std::rc::Rc;

/// Internal implementation of tensor addition
pub(crate) fn add_impl(a: &Tensor, b: &Tensor) -> Vec<f32> {
    let a_data = eval(a);
    let b_data = eval(b);
    a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x + y)
        .collect()
}
