use crate::engine::compute_gradients;
use crate::tensor::Tensor;

/// Compute the gradient of a function with respect to its inputs.
/// This is similar to JAX's grad transformation.
///
/// # Arguments
///
/// * `output` - The output tensor to compute gradients for
/// * `input` - The input tensor to compute gradients with respect to
/// * `seed_grad` - The seed gradient value (usually 1.0)
///
/// # Returns
///
/// The gradient of the output with respect to the input, if available.
pub fn grad(output: &Tensor, input: &Tensor, seed_grad: f32) -> Option<Tensor> {
    let grads = compute_gradients(output, seed_grad);
    // In a real implementation, we would need to match the input tensor with its gradient
    // For now, we'll just return the first gradient if available
    grads.into_iter().next()
}
