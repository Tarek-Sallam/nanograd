pub enum DataType {
    Int32,
    Int64,
    UInt32,
    UInt64,
    Float32,
    Float64,
}

pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<u32>,
    dtype: DataType,
}

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<u32>, dtype: DataType) -> Self {
        Tensor { data, shape, dtype }
    }

    pub fn shape(&self) -> &[T] {
        &self.shape[..]
    }
}
