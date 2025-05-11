use std::ops::Deref;
use std::rc::Rc;

// Data for the tensor as a struct
pub struct TensorData {
    data: Vec<f32>,
}

// Methods for the data struct
impl TensorData {
    pub fn new(data: Vec<f32>) -> Self {
        TensorData { data }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

// Core tensor kernel
pub struct TensorKernel {
    data: TensorData,
    shape: Vec<usize>,
    track_grad: bool,
}

// Tensor is a wrapper around Rc<TensorKernel>
#[derive(Clone)]
pub struct Tensor(Rc<TensorKernel>);

// Public Tensor Factory Function
pub fn create_tensor(data: Vec<f32>, shape: Vec<usize>, track_grad: bool) -> Tensor {
    // Create a new tensor without an operation
    Tensor(Rc::new(TensorKernel::new(data, shape, track_grad)))
}

// Methods for the tensor kernel
impl TensorKernel {
    /// Creates a new tensor kernel
    pub fn new(data: Vec<f32>, shape: Vec<usize>, track_grad: bool) -> Self {
        TensorKernel {
            data: TensorData::new(data),
            shape,
            track_grad,
        }
    }

    /// Returns the raw tensor data
    pub fn data(&self) -> &[f32] {
        self.data.data()
    }

    /// Returns the tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Whether to track gradients
    pub fn track_grad(&self) -> bool {
        self.track_grad
    }
}

// Implement Deref for Tensor to automatically dereference to TensorKernel
impl Deref for Tensor {
    type Target = TensorKernel;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
