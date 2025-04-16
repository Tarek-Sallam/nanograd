enum DataType {
    Int32,
    Int64,
    UInt32,
    UInt64,
    Float32,
    Float64,
}

struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<u32>,
    dtype: DataType,
}

impl Tensor<T> {
    fn new(data: Vec<T>, shape: Vec<u32>, dtype: DataType) -> Self {
        Tensor { data, shape, dtype }
    }
}
