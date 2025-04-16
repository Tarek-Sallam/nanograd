use nanograd::tensor::DataType;
use nanograd::tensor::Tensor;

fn create_tensor() {
    let data = vec![1, 2, 3, 4, 5];
    let shape = vec![5];
    let _t = Tensor::new(data, shape, DataType::Int64);
}
