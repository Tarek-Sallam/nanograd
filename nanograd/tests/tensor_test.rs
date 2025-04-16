use nanograd::tensor::DataType;
use nanograd::tensor::Tensor;

#[test]
fn create_tensor() {
    let data = vec![1, 2, 3, 4, 5];
    let shape = vec![5];
    let t = Tensor::new(data, shape, DataType::Int64);
    println!("{:?}", t.shape())
}

fn add_tensor() {
    let data1 = vec![1, 2, 3, 4, 5];
    let shape = vec![5];
    let data2 = vec![5, 4, 3, 2, 1];
    let t = Tensor::new(data, shape, None);
    let t2 = Tensor::new(data2, shape, None);
}
