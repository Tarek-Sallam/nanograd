use nanograd::ops::TensorOps;
use nanograd::tensor;

#[test]

fn add_tensor() {
    let data1 = vec![1, 2, 3, 4];
    let data2 = vec![3, 2, 1, 0];
    let tensor1 = tensor::tensor(data1, vec![4]);
    let tensor2 = tensor::tensor(data2, vec![4]);
    let tensor3 = tensor1.add(&tensor2);
    println!("`{:?}`", tensor3.data());
    println!("`{:?}`", tensor1.data());
    println!("`{:?}`", tensor2.data());
}
