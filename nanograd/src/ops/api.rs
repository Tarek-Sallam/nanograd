use crate::nano::{Nanode, with_builder_mut};
use crate::ops::types::OPERATIONS;
use crate::ops::types::OpType;
use crate::tensor::{Tensor, create_tensor};
// Public add function to add the two tensors
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let op = OPERATIONS
        .get(&OpType::Add)
        .expect("Add operation not found in operations map");

    // Get the result from applying the operation
    let inputs = vec![a.clone(), b.clone()];
    let result = (op.forward)(&inputs);

    // Record the operation in the IR
    let _ = with_builder_mut(|builder| {
        builder.record(Nanode::Op(
            OpType::Add, // We can use OpType::Add directly since we know it's an add operation
            vec![Nanode::Input; inputs.len()],
        ))
    });

    // Create a new tensor with the operation result
    let result_data = result.data().to_vec();
    let result_shape = result.shape().to_vec();

    create_tensor(result_data, result_shape, true)
}
