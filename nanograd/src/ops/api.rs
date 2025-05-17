use crate::tensor::{Tensor, TensorKernel};
use std::rc::Rc;

/// Add two tensors together (public API)
pub fn add(a: Tensor, b: Tensor) -> Tensor {
    Tensor(Rc::new(TensorKernel::Add(a.0, b.0)))
}
