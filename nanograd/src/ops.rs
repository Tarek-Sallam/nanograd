use crate::tensor::Tensor;
use std::rc::Rc;

enum OpType {
    Add,
}

pub struct Op<T> {
    op_type: OpType,
    inputs: Vec<Rc<Tensor<T>>>,
}

impl<T> Op<T> {
    pub fn new(op_type: OpType, inputs: Vec<Rc<Tensor<T>>>) -> Self {
        Op { op_type, inputs }
    }
}

impl<T> std::ops::Add for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, other: &Tensor<T>) -> Tensor<T> {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Tensors must have the same shape"
        );

        let data: Vec<i32> = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| *a + *b)
            .collect();

        Tensor::new(
            data,
            self.shape().clone(),
            Some(Rc::new(Op::new(
                OpType::Add,
                vec![Rc::clone(&self), Rc::clone(&other)],
            ))),
        )
    }
}
