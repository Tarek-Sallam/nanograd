pub enum DataType {
    Int32,
    Int64,
    UInt32,
    UInt64,
    Float32,
    Float64,
}

// Tensor struct
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<u32>,
    dtype: DataType,
}

// TensorView struct which provides a non-contiguous view of a Tensor
pub struct TensorView<'a, T> {
    tensor_ref: &'a Tensor,
    shape: Vec<u32>,
    strides: Vec<usize>,
    offset: usize,
}

// Tensor struct implementations
impl<T> Tensor<T> {
    // create a Tensor
    pub fn new(data: Vec<T>, shape: Vec<u32>, dtype: DataType) -> Self {
        Tensor { data, shape, dtype }
    }
    // return the shape
    pub fn shape(&self) -> &[u32] {
        &self.shape[..]
    }
    // return the data type
    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    pub fn get<'a>(&'a self, start: &[u32], stop: &[u32]) -> Option<TensorView<'a, T>> {
        if start.len() != stop.len() {
            return None;
        }
        // Need to implement this function that calculates strides + offset and returns view
    }
}
