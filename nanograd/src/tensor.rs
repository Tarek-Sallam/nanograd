use crate::ops::Op;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    id: usize,
    data: TensorData<T>,
    shape: Vec<usize>,
    track_grad: bool,
    op: Option<Arc<Op<T>>>,
}

// tensor is a reference counter of the tensor kernel
pub type Tensor<T> = Arc<TensorKernel<T>>;

pub fn tensor<T>(data: Vec<T>, shape: Vec<usize>, track_grad: bool) -> Tensor<T> {
    Arc::new(TensorKernel::new(data, shape, track_grad, None))
}

// methods for the tensor kernel
impl<T> TensorKernel<T> {
    // creates a new tensor kernel
    pub fn new(data: Vec<T>, shape: Vec<usize>, track_grad: bool, op: Option<Arc<Op<T>>>) -> Self {
        TensorKernel {
            id: AtomicUsize::new(0).fetch_add(1, Ordering::SeqCst),
            data: TensorData::new(data),
            shape,
            track_grad,
            op,
        }
    }

    // returns the data from the tensor kernel
    pub fn data(&self) -> &[T] {
        self.data.data()
    }

    pub fn track_grad(&self) -> bool {
        self.track_grad
    }

    // returns the shape of the tensor kernel
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn op(&self) -> Option<&Arc<Op<T>>> {
        self.op.as_ref()
    }
}
