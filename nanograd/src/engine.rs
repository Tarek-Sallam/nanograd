use crate::tensor::Tensor;
use std::collections::HashMap;

pub struct Engine;

impl Engine {
    pub fn backward<T>(output: &Tensor<T>, wrt: &Tensor<T>) -> Vec<T>
    where:
        T: Copy + Default + std::ops::Add<Output = T> + From<f32>
}