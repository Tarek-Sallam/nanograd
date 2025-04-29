use crate::ops::Op;
use std::sync::Arc;

// data for the tensor as a struct
pub struct TensorData<T> {
    data: Vec<T>,
}

// methods for the data struct
impl<T> TensorData<T> {
    pub fn new(data: Vec<T>) -> Self {
        TensorData { data: data }
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }
}

// core tensor kernel
pub struct TensorKernel<T> {
    data: TensorData<T>,
    shape: Vec<usize>,
    op: Option<Arc<Op<T>>>,
}

// tensor is a reference counter of the tensor kernel
pub type Tensor<T> = Arc<TensorKernel<T>>;

pub fn tensor<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
    Arc::new(TensorKernel::new(data, shape, None))
}

// methods for the tensor kernel
impl<T> TensorKernel<T> {
    // creates a new tensor kernel
    pub fn new(data: Vec<T>, shape: Vec<usize>, op: Option<Arc<Op<T>>>) -> Self {
        TensorKernel {
            data: TensorData::new(data),
            shape: shape,
            op: op,
        }
    }

    // returns the data from the tensor kernel
    pub fn data(&self) -> &[T] {
        self.data.data()
    }

    // returns the shape of the tensor kernel
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}
