use nanograd::tensor::DataType;
use nanograd::tensor::Tensor;

#[test]
fn create_tensor() {
    let data = vec![1, 2, 3, 4, 5];
    let shape = vec![5];
    let t = Tensor::new(data, shape, DataType::Int64);
    println!("{:?}", t.shape())
}
