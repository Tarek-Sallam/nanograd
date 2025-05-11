use crate::engine::compute_gradients;
use crate::tensor::Tensor;

pub fn grad(output: &Tensor, input: &Tensor, seed_grad: f32) -> Option<Tensor> {
    let grads = compute_gradients(output, seed_grad);
    grads.into_iter().next()
}
