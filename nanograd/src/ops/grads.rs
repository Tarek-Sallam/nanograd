use crate::ops::ops::{Op, OpType};
use crate::tensor::Tensor;
use std::collections::HashMap;

pub trait Grad<T> {
    fn grad(
        &self,
        output: &Tensor<T>,
        grad_out: &[T],
        grads: &mut HashMap<usize, Vec<T>>,
        stack: &mut Vec<Tensor<T>>,
    );
}

pub 