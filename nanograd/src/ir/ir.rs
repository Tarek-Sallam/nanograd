use crate::ops::ops::OpType;
use std::cell::RefCell;

// Thread-local IR builder, uses RefCell to allow mutability
thread_local! {
    static IR_BUILDER: RefCell<IRBuilder> = RefCell::new(IRBuilder::new());
}

// Public function to access the IR builder
pub fn with_builder<F, R>(f: F) -> R
where
    F: FnOnce(&IRBuilder) -> R,
{
    IR_BUILDER.with(|builder| f(&builder.borrow()))
}

// Function to access the IR builder mutably
pub fn with_builder_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut IRBuilder) -> R,
{
    IR_BUILDER.with(|builder| f(&mut builder.borrow_mut()))
}

// Represents the IR expressions, contains the operation, input tensor IDs, and output tensor ID
#[derive(Debug, Clone)]
pub enum IRExpr {
    Input(usize),
    Const(f32),
    Op(OpType, Vec<usize>),
    GradOp(usize, Vec<usize>),
}

// Represents the IR, contains all IR expressions and the output ID
#[derive(Debug, Clone)]
pub struct IR {
    pub exprs: Vec<(usize, IRExpr)>,
    pub output_id: usize,
}

// Represents the IR builder, contains the next ID and all IR expressions
#[derive(Clone)]
pub struct IRBuilder {
    next_id: usize,
    exprs: Vec<(usize, IRExpr)>,
}

impl IRBuilder {
    pub fn new() -> Self {
        IRBuilder {
            next_id: 0,
            exprs: vec![],
        }
    }

    pub fn next_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn record(&mut self, expr: IRExpr) -> usize {
        let id = self.next_id();
        self.exprs.push((id, expr));
        id
    }

    pub fn build(&self, output_id: usize) -> IR {
        IR {
            exprs: self.exprs.clone(),
            output_id,
        }
    }
}
