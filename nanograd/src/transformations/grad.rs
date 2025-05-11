use crate::engine::compute_gradients;
use crate::tensor::Tensor;

pub fn grad(output: &Tensor, input: &Tensor, seed_grad: f32) -> Option<Tensor> {
    let grads = compute_gradients(output, seed_grad);
    // In a real implementation, we would need to match the input tensor with its gradient
    // For now, we'll just return the first gradient if available
    grads.into_iter().next()
}
