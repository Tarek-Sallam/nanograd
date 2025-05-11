use crate::ir::ir::{IR, IRExpr, with_builder, with_builder_mut};
use crate::tensor::{Tensor, tensor};
use std::collections::HashMap;

/// Represents the computational graph for gradient computation
pub struct GradGraph {
    ir: IR,
    grads: HashMap<usize, Tensor>,
}

impl GradGraph {
    /// Create a new gradient graph from an IR
    pub fn new(ir: IR) -> Self {
        GradGraph {
            ir,
            grads: HashMap::new(),
        }
    }

    /// Get the gradient for a given tensor id
    pub fn get_grad(&self, id: usize) -> Option<&Tensor> {
        self.grads.get(&id)
    }

    /// Set the gradient for a given tensor id
    pub fn set_grad(&mut self, id: usize, grad: Tensor) {
        self.grads.insert(id, grad);
    }

    /// Backward pass to compute gradients
    pub fn backward(&mut self, seed_grad: f32) {
        let output_id = self.ir.output_id;

        // Initialize output gradient with seed value
        let output_shape = self.find_tensor_shape(output_id);
        let seed_data = vec![seed_grad; output_shape.iter().product::<usize>()];
        let seed_tensor = tensor(seed_data, output_shape, true);
        self.set_grad(output_id, seed_tensor);

        // Topologically sorted list of operations (reverse order)
        let topo_sorted = self.topological_sort();

        // Perform backward pass
        for &expr_id in topo_sorted.iter().rev() {
            if let Some((_, expr)) = self.ir.exprs.iter().find(|(id, _)| *id == expr_id) {
                match expr {
                    IRExpr::Op(_, input_ids) => {
                        // Extract all data we need before any mutable operations
                        let grad_output = self.get_grad(expr_id).cloned();
                        let tensor_with_op = self.find_tensor_with_op(expr_id);
                        let input_ids = input_ids.clone(); // Clone to avoid borrowing issues

                        // Process with the extracted data
                        if let Some(grad_output) = grad_output {
                            if let Some(tensor) = tensor_with_op {
                                if let Some(op) = tensor.op() {
                                    // Get input tensors from their IDs
                                    let inputs: Vec<Tensor> = input_ids
                                        .iter()
                                        .filter_map(|&id| self.find_tensor_with_op(id))
                                        .collect();

                                    // Compute gradients for inputs
                                    let input_grads = op.compute_gradients(&inputs, &grad_output);

                                    // Accumulate gradients for each input
                                    for (i, input_id) in input_ids.iter().enumerate() {
                                        if i < input_grads.len() {
                                            let grad_to_accumulate = input_grads[i].clone();
                                            self.accumulate_grad(*input_id, grad_to_accumulate);
                                        }
                                    }

                                    // Record gradient operation in IR
                                    with_builder_mut(|builder| {
                                        builder.record(IRExpr::GradOp(expr_id, input_ids));
                                    });
                                }
                            }
                        }
                    }
                    IRExpr::Input(_) | IRExpr::Const(_) => {
                        // Leaf nodes, no gradient computation needed
                    }
                    IRExpr::GradOp(_, _) => {
                        // These are recorded during backward pass, no action needed here
                    }
                }
            }
        }
    }

    // Accumulate gradient for a given tensor id
    fn accumulate_grad(&mut self, id: usize, grad: Tensor) {
        // Get and clone the existing gradient if it exists
        let existing_grad_opt = self.grads.get(&id).cloned();

        if let Some(existing_grad) = existing_grad_opt {
            // Add the new gradient to the existing one
            let shape = existing_grad.shape().to_vec();
            let new_data: Vec<f32> = existing_grad
                .data()
                .iter()
                .zip(grad.data().iter())
                .map(|(&a, &b)| a + b)
                .collect();

            self.set_grad(id, tensor(new_data, shape, true));
        } else {
            self.set_grad(id, grad);
        }
    }

    // Find the tensor with a given id in all expressions
    fn find_tensor_with_op(&self, id: usize) -> Option<Tensor> {
        // This is a simplified implementation
        // In a real-world scenario, you would maintain a mapping of id -> tensor
        // For demonstration purposes, we'll return None as we can't access tensors directly from IR
        None
    }

    // Find the shape of a tensor with the given id
    fn find_tensor_shape(&self, id: usize) -> Vec<usize> {
        // For simplicity, we'll return a default shape
        // In a real implementation, you would maintain a mapping of id -> tensor shape
        vec![1]
    }

    // Topologically sort the operations in the IR
    fn topological_sort(&self) -> Vec<usize> {
        let mut visited = HashMap::new();
        let mut topo_order = Vec::new();

        // Visit all expressions
        for (id, _) in &self.ir.exprs {
            self.visit(*id, &mut visited, &mut topo_order);
        }

        topo_order
    }

    // DFS visit for topological sort
    fn visit(&self, id: usize, visited: &mut HashMap<usize, bool>, topo_order: &mut Vec<usize>) {
        // If already completely visited, skip
        if let Some(&complete) = visited.get(&id) {
            if complete {
                return;
            }
        }

        // Mark as being visited
        visited.insert(id, false);

        // Visit all dependencies
        if let Some((_, expr)) = self.ir.exprs.iter().find(|(expr_id, _)| *expr_id == id) {
            match expr {
                IRExpr::Op(_, input_ids) => {
                    for &input_id in input_ids {
                        self.visit(input_id, visited, topo_order);
                    }
                }
                IRExpr::GradOp(_, input_ids) => {
                    for &input_id in input_ids {
                        self.visit(input_id, visited, topo_order);
                    }
                }
                _ => {}
            }
        }

        // Mark as completely visited and add to topo order
        visited.insert(id, true);
        topo_order.push(id);
    }
}

/// Compute gradients for a given output tensor
pub fn compute_gradients(output: &Tensor, seed_grad: f32) -> GradGraph {
    // Build the IR
    let ir = with_builder(|builder| builder.build(output.id()));

    // Create gradient graph
    let mut grad_graph = GradGraph::new(ir);

    // Compute gradients
    grad_graph.backward(seed_grad);

    grad_graph
}

/// A simplified API to get the gradient of a specific tensor
pub fn grad(output: &Tensor, input: &Tensor, seed_grad: f32) -> Option<Tensor> {
    let grad_graph = compute_gradients(output, seed_grad);
    grad_graph.get_grad(input.id()).cloned()
}
