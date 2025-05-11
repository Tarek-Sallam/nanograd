use crate::nano::{Nanode, with_builder_mut};
use crate::tensor::{Tensor, create_tensor};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

// Type of operation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpType {
    Add,
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpType::Add => write!(f, "Add"),
        }
    }
}

// Type for operation function
type OpFn = fn(&[Tensor]) -> Tensor;

// Type for gradient function
type GradFn = fn(&[Tensor], &Tensor) -> Vec<Tensor>;

// Represents an operation in the computation graph
pub struct Op {
    pub op_type: OpType,
    pub forward: OpFn,
    pub backward: GradFn,
}

impl Op {
    pub fn new(op_type: OpType, forward: OpFn, backward: GradFn) -> Self {
        Op {
            op_type,
            forward,
            backward,
        }
    }

    pub fn apply(&self, inputs: &[Tensor]) -> Tensor {
        (self.forward)(inputs)
    }

    pub fn compute_gradients(&self, inputs: &[Tensor], grad_output: &Tensor) -> Vec<Tensor> {
        (self.backward)(inputs, grad_output)
    }
}

// Static forward and backward functions for addition
fn add_forward(inputs: &[Tensor]) -> Tensor {
    if inputs.len() != 2 {
        panic!("Add operation requires exactly 2 inputs");
    }
    let a = &inputs[0];
    let b = &inputs[1];

    // Validate shapes match
    if a.shape() != b.shape() {
        panic!(
            "Cannot add tensors with different shapes: {:?} and {:?}",
            a.shape(),
            b.shape()
        );
    }

    let result_data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| x + y)
        .collect();

    create_tensor(result_data, a.shape().to_vec(), true)
}

fn add_backward(inputs: &[Tensor], grad_output: &Tensor) -> Vec<Tensor> {
    if inputs.len() != 2 {
        panic!("Add operation requires exactly 2 inputs");
    }
    let a = &inputs[0];
    let b = &inputs[1];

    // Validate shapes match
    if a.shape() != b.shape() || a.shape() != grad_output.shape() {
        panic!("Shape mismatch in add backward pass");
    }

    let grad_a = if a.track_grad() {
        grad_output.clone()
    } else {
        create_tensor(vec![], vec![], false)
    };

    let grad_b = if b.track_grad() {
        grad_output.clone()
    } else {
        create_tensor(vec![], vec![], false)
    };

    vec![grad_a, grad_b]
}

// Static map of operations
static OPERATIONS: Lazy<HashMap<OpType, Arc<Op>>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert(
        OpType::Add,
        Arc::new(Op::new(OpType::Add, add_forward, add_backward)),
    );
    m
});

// Get an operation by type
pub fn get_op(op_type: OpType) -> Option<Arc<Op>> {
    OPERATIONS.get(&op_type).cloned()
}

// Execute the operation and record it in the IR
pub fn apply_op(op: Arc<Op>, inputs: &[Tensor], track_grad: bool) -> Tensor {
    // Get the result from applying the operation
    let result = op.apply(inputs);

    // Record the operation in the IR
    let _ = with_builder_mut(|builder| {
        builder.record(Nanode::Op(
            op.op_type.clone(),
            vec![Nanode::Input; inputs.len()],
        ))
    });

    // Create a new tensor with the operation result
    let result_data = result.data().to_vec();
    let result_shape = result.shape().to_vec();

    create_tensor(result_data, result_shape, track_grad)
}

// Add operation implementation
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let op = OPERATIONS
        .get(&OpType::Add)
        .expect("Add operation not found in operations map")
        .clone();
    let inputs = vec![a.clone(), b.clone()];
    apply_op(op, &inputs, true)
}
