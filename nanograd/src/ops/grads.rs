use crate::tensor::Tensor;

pub type GradFn<T> = Box<dyn Fn(Option<&Tensor<T>>, &[Tensor<T>])>;

pub fn add_grad<T>(output_grad: Option<&Tensor<T>>, inputs: &[Tensor<T>]) {}
