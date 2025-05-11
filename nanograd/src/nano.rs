use crate::ops::OpType;
use std::cell::RefCell;

// Represents a node in the computation graph
#[derive(Clone, Debug)]
pub enum Nanode {
    Input,
    Op(OpType, Vec<Nanode>),
    GradOp(OpType, Vec<Nanode>),
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
    pub fn record(&self, node: Nanode) -> Self {
        let mut nodes = self.nodes.clone();
        nodes.push(node);
        NanoForge { nodes }
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

// Execute a closure with the graph builder, returning a new builder
pub fn with_builder_mut<F>(f: F) -> NanoForge
where
    F: FnOnce(&NanoForge) -> NanoForge,
{
    NANO_FORGE.with(|builder| {
        let current = builder.borrow();
        let new = f(&current);
        *builder.borrow_mut() = new.clone();
        new
    })
}
