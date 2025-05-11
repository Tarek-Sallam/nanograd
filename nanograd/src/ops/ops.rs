use crate::ir::ir::{IR, IRExpr, with_builder, with_builder_mut};
use crate::tensor::{Tensor, TensorKernel, tensor};
use std::fmt;
use std::rc::Rc;

/// Type of operation
#[derive(Debug, Clone, PartialEq)]
pub enum OpType {
    Add,
    // Other operations can be added later
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpType::Add => write!(f, "Add"),
            // Other operations can be added later
        }
    }
}

/// Type alias for operation function
type OpFn = fn(&[Tensor]) -> Tensor;
/// Type alias for gradient function
type GradFn = fn(&[Tensor], &Tensor) -> Vec<Tensor>;

/// Represents an operation in the computation graph
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

impl fmt::Debug for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Op")
            .field("op_type", &self.op_type)
            .finish()
    }
}

// Factory function to create operation
pub fn create_op(op_type: OpType, forward: OpFn, backward: GradFn) -> Rc<Op> {
    Rc::new(Op::new(op_type, forward, backward))
}

// Execute the operation and record it in the IR
pub fn apply_op(op: Rc<Op>, inputs: &[Tensor], track_grad: bool) -> Tensor {
    // Get the result from applying the operation
    let result = op.apply(inputs);

    // Record the operation in the IR
    let input_ids: Vec<usize> = inputs.iter().map(|t| t.id()).collect();
    let result_id =
        with_builder_mut(|builder| builder.record(IRExpr::Op(op.op_type.clone(), input_ids)));

    // Create a new tensor with the operation result and link to the op
    let result_data = result.data().to_vec();
    let result_shape = result.shape().to_vec();

    Rc::new(TensorKernel::new(
        result_id,
        result_data,
        result_shape,
        track_grad,
        Some(op),
    ))
}

// Add operation implementation
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let forward: OpFn = |inputs| {
        let a = &inputs[0];
        let b = &inputs[1];

        // Element-wise addition
        let result_data: Vec<f32> = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| x + y)
            .collect();

        tensor(result_data, a.shape().to_vec(), true)
    };

    let backward: GradFn = |inputs, grad_output| {
        let a = &inputs[0];
        let b = &inputs[1];

        // For addition, gradients are simply passed through
        // ∂L/∂a = ∂L/∂output * ∂output/∂a = ∂L/∂output * 1 = ∂L/∂output
        // ∂L/∂b = ∂L/∂output * ∂output/∂b = ∂L/∂output * 1 = ∂L/∂output
        let grad_a = if a.track_grad() {
            grad_output.clone()
        } else {
            tensor(vec![], vec![], false)
        };

        let grad_b = if b.track_grad() {
            grad_output.clone()
        } else {
            tensor(vec![], vec![], false)
        };

        vec![grad_a, grad_b]
    };

    let op = create_op(OpType::Add, forward, backward);
    let inputs = vec![a.clone(), b.clone()];
    apply_op(op, &inputs, true)
}
