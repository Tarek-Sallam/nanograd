use crate::tensor::Tensor;

pub type GradFn<T> = Box<dyn Fn(Option<&Tensor<T>>, &mut [Tensor<T>]) -> () + 'static>;

pub fn add_grad<T>(output_grad: Option<&Tensor<T>>, inputs: &mut [Tensor<T>])
where
    T: Copy + std::ops::AddAssign + 'static,
{
    if let Some(out_grad) = output_grad {
        let grad_vals = out_grad.data();

        for input in inputs {
            if input.track_grad() {
                match &mut input.get_grad() {
                    Some(grad) => {
                        for (x, y) in grad.data().iter_mut().zip(grad_vals.iter()) {
                            *x += *y;
                        }
                    }
                    None => {
                        // If no gradient exists, initialize it with the output gradient
                        input.set_grad(Some(Tensor::new(
                            grad_vals.to_vec(),
                            input.shape.clone(),
                            input.track_grad,
                        )));
                    }
                }
            }
        }
    }
}
