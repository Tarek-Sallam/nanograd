use nanograd::ops::api::add;
use nanograd::tensor::Tensor;
use nanograd::transformations::eval::eval;

#[test]
fn test_tensor_add() {
    let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let c = add(a, b);
    let result = eval(&c);
    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
}
