use crate::tensor::Tensor;
use crate::tensor::TensorKernel;

use std::sync::Arc;

// operation type enumeration
pub enum OpType {
    Add,
    // Add more operations as needed
}

// operation struct
pub struct Op<T> {
    op_type: OpType,
    inputs: Vec<Tensor<T>>,
}

// methods for the operation struct
impl<T> Op<T> {
    // creates a new operation
    pub fn new(op_type: OpType, inputs: Vec<Tensor<T>>) -> Self {
        Op { op_type, inputs }
    }
}

// tensor operations trait for tensor
pub trait TensorOps<T> {
    fn add(&self, other: &Tensor<T>) -> Tensor<T>
    where
        T: std::ops::Add<Output = T> + Copy;
}

// tensor operations methods
impl<T> TensorOps<T> for Tensor<T> {
    // add method
    fn add(&self, other: &Tensor<T>) -> Tensor<T>
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        // construct the data from the two input tensors
        let data: Vec<T> = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(&a, &b)| a + b)
            .collect();

        // construct the new operation with references to parent tensors
        let op = Arc::new(Op::new(OpType::Add, vec![self.clone(), other.clone()]));
        let track_grad = self.track_grad() || other.track_grad();
        // construct and return the new tensor kernel as the result of adding the two tensors
        Arc::new(TensorKernel::new(
            data,
            self.shape().to_vec(),
            track_grad,
            Some(op),
        ))
    }
}
