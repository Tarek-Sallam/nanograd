use crate::ir::ir::{NanoGraph, Nanode, with_builder, with_builder_mut};
use crate::ops::ops::{Op, OpType, add, create_op};
use crate::tensor::{Tensor, create_tensor};
use std::rc::Rc;

/// Represents the computational graph for gradient computation
pub struct GradGraph {
    graph: NanoGraph,
}

impl GradGraph {
    /// Create a new gradient graph from a computation graph
    pub fn new(graph: NanoGraph) -> Self {
        GradGraph { graph }
    }

    /// Backward pass to compute gradients
    pub fn backward(&self, output: &Tensor, seed_grad: f32) -> Vec<Tensor> {
        // Initialize output gradient with seed value
        let output_shape = output.shape().to_vec();
        let seed_data = vec![seed_grad; output_shape.iter().product::<usize>()];
        let seed_tensor = create_tensor(seed_data, output_shape, true);

        // Process nodes in reverse order
        let nodes: Vec<Nanode> = self.graph.nodes.iter().cloned().collect();
        let mut grads = Vec::new();

        for node in nodes.iter().rev() {
            if let Some(grad) = self.process_node(node, &seed_tensor) {
                grads.push(grad);
            }
        }

        grads
    }

    // Process a single node and return its gradient
    fn process_node(&self, node: &Nanode, grad_output: &Tensor) -> Option<Tensor> {
        match node {
            Nanode::Op(op_type, inputs) => {
                // For addition, gradients are simply passed through
                if *op_type == OpType::Add && inputs.len() == 2 {
                    // Record gradient operation
                    with_builder_mut(|builder| {
                        builder.record(Nanode::GradOp(op_type.clone(), inputs.clone()));
                    });

                    // Return the gradient
                    Some(grad_output.clone())
                } else {
                    None
                }
            }
            Nanode::Input => {
                // Return the gradient for input tensors
                Some(grad_output.clone())
            }
            Nanode::GradOp(_, _) => {
                // These are recorded during backward pass, no action needed here
                None
            }
        }
    }
}

/// Compute gradients for a given output tensor
pub fn compute_gradients(output: &Tensor, seed_grad: f32) -> Vec<Tensor> {
    // Build the computation graph
    let graph = with_builder(|builder| builder.build());

    // Create gradient graph and compute gradients
    let grad_graph = GradGraph::new(graph);
    grad_graph.backward(output, seed_grad)
}
