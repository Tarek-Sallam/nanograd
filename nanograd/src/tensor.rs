use crate::ops::Op;
use std::rc::Rc;

pub enum DataType {
    Int32,
    Int64,
    UInt32,
    UInt64,
    Float32,
    Float64,
}

// Tensor struct
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<u32>,
    op: Option<Rc<Op<T>>>,
}

// TensorView struct which provides a non-contiguous view of a Tensor
pub struct TensorView<'a, T> {
    tensor_ref: &'a Tensor<T>,
    shape: Vec<u32>,
    strides: Vec<usize>,
    offset: usize,
}

// Tensor struct implementations
impl<T> Tensor<T> {
    // create a Tensor
    pub fn new(data: Vec<T>, shape: Vec<u32>, op: Option<Rc<Op<T>>>) -> Self {
        Tensor { data, shape, op }
    }
    // return the shape
    pub fn shape(&self) -> &[u32] {
        &self.shape[..]
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn get<'a>(&'a self, start: &[u32], stop: &[u32]) -> Option<TensorView<'a, T>> {
        if start.len() != stop.len() {
            return None;
        }
        // Need to implement this function that calculates strides + offset and returns view
    }
}
