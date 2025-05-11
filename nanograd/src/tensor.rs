use crate::ir::ir::{IRExpr, with_builder_mut};
use crate::ops::ops::Op;
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
    id: usize,
    data: TensorData,
    shape: Vec<usize>,
    track_grad: bool,
    op: Option<Rc<Op>>,
}

// Tensor is a reference counter of the tensor kernel
pub type Tensor = Rc<TensorKernel>;

// Public Tensor Factory Functio
pub fn tensor(data: Vec<f32>, shape: Vec<usize>, track_grad: bool) -> Tensor {
    // Get the next ID for the tensor
    let id = with_builder_mut(|builder| {
        let next_id = builder.next_id();
        builder.record(IRExpr::Input(next_id));
        next_id
    });

    // Return a new TensorKernel wrapped in an Rc
    Rc::new(TensorKernel::new(id, data, shape, track_grad, None))
}

// Methods for the tensor kernel
impl TensorKernel {
    /// Creates a new tensor kernel
    pub fn new(
        id: usize,
        data: Vec<f32>,
        shape: Vec<usize>,
        track_grad: bool,
        op: Option<Rc<Op>>,
    ) -> Self {
        TensorKernel {
            id,
            data: TensorData::new(data),
            shape,
            track_grad,
            op,
        }
    }

    /// Returns the unique ID
    pub fn id(&self) -> usize {
        self.id
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

    /// Operation that created the tensor
    pub fn op(&self) -> Option<Rc<Op>> {
        self.op.clone()
    }
}
