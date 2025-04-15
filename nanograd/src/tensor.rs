pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        Tensor { data, shape }
    }
}
