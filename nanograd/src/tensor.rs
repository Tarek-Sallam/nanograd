enum DataType {
    Int32,
    Int64,
    UInt32,
    UInt64,
    Float32,
    Float64,
}

struct Tensor<T> {
    dtype: DataType,
    data: Vec<T>,
    shape: Vec<u32>,
}
