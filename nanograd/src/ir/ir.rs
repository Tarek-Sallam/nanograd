use crate::ops::ops::OpType;
use std::cell::RefCell;
use std::rc::Rc;

/// Represents a node in the computation graph
#[derive(Clone, Debug)]
pub enum Nanode {
    Input,
    Op(OpType, Vec<Nanode>),
    GradOp(OpType, Vec<Nanode>),
}

/// Represents the computation graph
#[derive(Clone, Debug)]
pub struct NanoGraph {
    pub nodes: Vec<Nanode>,
}

/// Builder for constructing the computation graph
pub struct NanoForge {
    nodes: Vec<Nanode>,
}

impl NanoForge {
    pub fn new() -> Self {
        NanoForge { nodes: Vec::new() }
    }

    pub fn record(&mut self, node: Nanode) {
        self.nodes.push(node);
    }

    pub fn build(&self) -> NanoGraph {
        NanoGraph {
            nodes: self.nodes.clone(),
        }
    }
}

// Thread-local graph builder
thread_local! {
    static NANO_FORGE: RefCell<NanoForge> = RefCell::new(NanoForge::new());
}

/// Execute a closure with read-only access to the graph builder
pub fn with_builder<F, R>(f: F) -> R
where
    F: FnOnce(&NanoForge) -> R,
{
    NANO_FORGE.with(|builder| f(&builder.borrow()))
}

/// Execute a closure with mutable access to the graph builder
pub fn with_builder_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut NanoForge) -> R,
{
    NANO_FORGE.with(|builder| f(&mut builder.borrow_mut()))
}
