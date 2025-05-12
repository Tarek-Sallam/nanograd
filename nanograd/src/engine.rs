use crate::nano::{NanoGraph, Nanode, with_builder, with_builder_mut};
use crate::ops::types::OpType;
use crate::tensor::{Tensor, create_tensor};
use std::collections::HashMap;

// Methods for the computation graph
impl NanoGraph {
    /// Backward pass to compute gradients
    pub fn backward(&self, output: &Tensor, seed_grad: f32) -> Vec<Tensor> {
        // Initialize output gradient with seed value
        let output_shape = output.shape().to_vec();
        let seed_data = vec![seed_grad; output_shape.iter().product::<usize>()];
        let seed_tensor = create_tensor(seed_data, output_shape, true);

        // Process nodes in reverse order
        let nodes: Vec<Nanode> = self.nodes.iter().cloned().collect();
        let mut grad_map: HashMap<usize, Tensor> = HashMap::new();

        for (idx, node) in nodes.iter().enumerate().rev() {
            if let Some(grad) = self.process_node(node, &seed_tensor, &grad_map) {
                // Accumulate gradients for the same input
                if let Nanode::Input = node {
                    grad_map.insert(idx, grad);
                }
            }
        }

        // Convert gradient map to vector in order of nodes
        nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| {
                if node.is_input() {
                    grad_map.get(&idx).cloned()
                } else {
                    None
                }
            })
            .collect()
    }

    // Process a single node and return its gradient
    fn process_node(
        &self,
        node: &Nanode,
        grad_output: &Tensor,
        grad_map: &HashMap<usize, Tensor>,
    ) -> Option<Tensor> {
        match node {
            Nanode::Op(op_type, inputs) => {
                // Record gradient operation
                let _ = with_builder_mut(|builder| {
                    builder.record(Nanode::GradOp(op_type.clone(), inputs.clone()));
                });

                match op_type {
                    OpType::Add => {
                        if inputs.len() == 2 {
                            // For addition, gradients are passed through
                            Some(grad_output.clone())
                        } else {
                            None
                        }
                    }
                }
            }
            Nanode::Input => {
                // Return the gradient for input tensors
                Some(grad_output.clone())
            }
            Nanode::GradOp(op_type, inputs) => {
                // Handle gradient operations
                match op_type {
                    OpType::Add => {
                        if inputs.len() == 2 {
                            // For addition, gradients are passed through
                            Some(grad_output.clone())
                        } else {
                            None
                        }
                    }
                }
            }
        }
    }
}

// Compute gradients for a given output tensor
pub fn compute_gradients(output: &Tensor, seed_grad: f32) -> Vec<Tensor> {
    // Build the computation graph
    let graph = with_builder(|builder| builder.build());

    // Compute gradients
    graph.backward(output, seed_grad)
}
