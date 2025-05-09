use crate::tensor::{Tensor, TensorKernel};
use std::rc::Rc;

pub type GradFn<T> = Box<dyn Fn(Option<&Tensor<T>>, &mut [Tensor<T>]) -> () + 'static>;

pub fn add_grad<T>(output_grad: &Tensor<T>, inputs: &[Tensor<T>]) -> Vec<Tensor<T>>
where
    T: Copy + std::ops::AddAssign + 'static,
{
    let mut new_inputs = Vec::new();
    for input in inputs {
        if input.track_grad() {
            let new_grad = Rc::new(TensorKernel::new(
                output_grad.data().to_vec(),
                output_grad.shape().to_vec(),
                output_grad.track_grad(),
                None,
                None,
            ));
            let new_input = Rc::new(TensorKernel::new(
                input.data().to_vec(),
                input.shape().to_vec(),
                input.track_grad(),
                input.op(),
                Some(new_grad),
            ));
            new_inputs.push(new_input);
        } else {
            new_inputs.push(input.clone());
        }
    }
    new_inputs
}
