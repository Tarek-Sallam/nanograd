use crate::tensor::Tensor;

pub struct Engine;

impl Engine {
    pub fn backward<T>(output: &Tensor<T>)
    where
        T: Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + From<f32>
            + 'static,
    {
        let mut stack = vec![output.clone()];

        while let Some(tensor) = stack.pop() {
            let output_grad = tensor.get_grad();

            if let Some(op) = tensor.op() {
                let new_inputs = op.grad_fn(output_grad);
            }
        }
    }
}
