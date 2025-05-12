use crate::ops::ops::*;

use crate::tensor::Tensor;
use once_cell::sync::Lazy;
use std::collections::HashMap;

// Type of operation, displayable, cloneable, comparable, hashable
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpType {
    Add,
}

// Type alias for operation function
pub type OpFn = fn(&[Tensor]) -> Tensor;

// Type alias for gradient function
pub type GradFn = fn(&[Tensor], &Tensor) -> Vec<Tensor>;

// Static map of operations
pub static OPERATIONS: Lazy<HashMap<OpType, Op>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert(
        OpType::Add,
        Op {
            forward: add_forward,
            backward: add_backward,
        },
    );
    m
});
