use std::fmt::Debug;
use std::rc::Rc;

/// Raw tensor data storage
#[derive(Debug, Clone)]
pub struct TensorData {
    data: Vec<f32>,
}

impl TensorData {
    // Construct new tensor data from a vector of floats
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    // Construct new tensor data from a scalar value
    pub fn scalar(value: f32) -> Self {
        Self { data: vec![value] }
    }

    // Get the data as a slice
    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

/// The core tensor implementation that handles data and operations
#[derive(Debug, Clone)]
pub enum TensorKernel {
    // Leaf nodes (actual data)
    Data {
        data: Rc<TensorData>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
    // Operation nodes
    Add(Rc<TensorKernel>, Rc<TensorKernel>),
}

/// A reference-counted tensor
#[derive(Clone)]
pub struct Tensor(pub Rc<TensorKernel>);

impl Tensor {
    /// Create a new tensor from raw data
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), shape.iter().product());

        // Calculate strides for efficient indexing
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        Self(Rc::new(TensorKernel::Data {
            data: Rc::new(TensorData::new(data)),
            shape,
            strides,
        }))
    }

    /// Create a new tensor from a scalar value
    pub fn scalar(value: f32) -> Self {
        Self::new(vec![1], vec![value])
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> Vec<usize> {
        match &*self.0 {
            TensorKernel::Data { shape, .. } => shape.clone(),
            TensorKernel::Add(a, _) => Tensor(Rc::clone(a)).shape(),
        }
    }
}
