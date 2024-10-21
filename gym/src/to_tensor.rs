use burn::{
    prelude::*,
    tensor::{BasicOps, Element, TensorData},
};

/// A trait for converting items to tensors
///
/// Commonly implemented for `Vec<T>` to convert batches of `T` to a tensor of dimension `D`
pub trait ToTensor<B: Backend, const D: usize, K: BasicOps<B>> {
    fn to_tensor(self, device: &B::Device) -> Tensor<B, D, K>;
}

impl<B, E, K> ToTensor<B, 1, K> for Vec<E>
where
    B: Backend,
    E: Element,
    K: BasicOps<B, Elem = E>,
{
    fn to_tensor(self, device: &<B as Backend>::Device) -> Tensor<B, 1, K> {
        let len = self.len();
        Tensor::from_data(TensorData::new(self, [len]), device)
    }
}

impl<B, E, K, const A: usize> ToTensor<B, 2, K> for Vec<[E; A]>
where
    B: Backend,
    E: Element,
    K: BasicOps<B, Elem = E>,
{
    fn to_tensor(self, device: &B::Device) -> Tensor<B, 2, K> {
        let len = self.len();
        let data = TensorData::new(self.into_iter().flatten().collect::<Vec<_>>(), [len * A]);
        Tensor::<B, 1, K>::from_data(data, device).reshape([-1, A as i32])
    }
}

// impl<B, E, K> ToTensor<B, 2, K> for Vec<Vec<E>>
// where
//     B: Backend,
//     E: Element,
//     K: BasicOps<B, Elem = E>,
// {
//     fn to_tensor(self, device: &B::Device) -> Tensor<B, 2, K> {
//         let len = self.len();
//         let data = TensorData::new(
//             self.into_iter().flatten().collect::<Vec<_>>(),
//             [len * self[0].len()],
//         );
//         Tensor::<B, 1, K>::from_data(data, device).reshape([-1, self[0].len() as i32])
//     }
// }

#[cfg(test)]
mod tests {
    use burn::backend::{ndarray::NdArrayDevice, NdArray as B};

    use super::*;

    #[test]
    fn vec_impl() {
        let device = NdArrayDevice::Cpu;
        let x = vec![1f32, 2.0, 3.0];
        let t1: Tensor<B, 1> = x.to_tensor(&device);

        let t2: Tensor<B, 1> = vec![1f32, 2.0, 3.0].to_tensor(&device);
        assert!(
            t1.equal(t2).all().into_scalar(),
            "valid tensor constructed from `Vec<E>`"
        );
    }

    #[test]
    fn vec_arr_impl() {
        let device = NdArrayDevice::Cpu;
        let x = vec![[1f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let t1: Tensor<B, 2> = x.to_tensor(&device);

        let t2: Tensor<B, 2> = vec![[1f32, 2.0, 3.0], [4.0, 5.0, 6.0]].to_tensor(&device);
        assert!(
            t1.equal(t2).all().into_scalar(),
            "valid tensor constructed from `Vec<[E; A]>`"
        );
    }
}
