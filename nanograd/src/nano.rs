use crate::ops::types::OpType;
use once_cell::unsync::Lazy;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

// Represents a node in the computation graph
#[derive(Clone, Debug)]
pub enum Nanode {
    Input,
    Op(OpType, Vec<Nanode>),
    GradOp(OpType, Vec<Nanode>),
}

impl Nanode {
    // Helper method to check if a node is an input
    pub fn is_input(&self) -> bool {
        matches!(self, Nanode::Input)
    }

    // Helper method to get the operation type if the node is an Op or GradOp
    pub fn op_type(&self) -> Option<&OpType> {
        match self {
            Nanode::Op(op, _) | Nanode::GradOp(op, _) => Some(op),
            Nanode::Input => None,
        }
    }

    // Helper method to get the children of a node
    pub fn children(&self) -> &[Nanode] {
        match self {
            Nanode::Op(_, children) | Nanode::GradOp(_, children) => children,
            Nanode::Input => &[],
        }
    }
}

// Represents the computation graph
#[derive(Clone, Debug)]
pub struct NanoGraph {
    pub nodes: Vec<Nanode>,
}

// Builder for constructing the computation graph
#[derive(Clone)]
pub struct NanoForge {
    nodes: Vec<Nanode>,
}

// Methods for the forge
impl NanoForge {
    // Create a new forge
    pub fn new() -> Self {
        NanoForge { nodes: Vec::new() }
    }

    // Record a node in the graph
    pub fn record(&mut self, node: Nanode) {
        self.nodes.push(node);
    }

    // Build the graph
    pub fn build(&self) -> NanoGraph {
        NanoGraph {
            nodes: self.nodes.clone(),
        }
    }
}

// Thread-local graph builder, we use RefCell to allow mutable access
thread_local! {
    static NANO_FORGE: RefCell<NanoForge> = RefCell::new(NanoForge::new());
}

// Execute a closure with read-only access to the graph builder
pub fn with_builder<F, R>(f: F) -> R
where
    F: FnOnce(&NanoForge) -> R,
{
    NANO_FORGE.with(|builder| f(&builder.borrow()))
}

// Execute a closure with mutable access to the graph builder
pub fn with_builder_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut NanoForge) -> R,
{
    NANO_FORGE.with(|builder| f(&mut builder.borrow_mut()))
}
