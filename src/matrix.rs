// #![allow(incomplete_features)]
// #![feature(generic_const_exprs, adt_const_params, const_generics, const_evaluatable_checked)]

#[repr(usize)]
enum MatrixSize {
	_2X2 = 2,
	_3X3 = 3,
	_4X4 = 4,
}

enum MatrixFill<T, const N: usize> {
	Array([[T; N]; N]),
	Single(T),
}

#[derive(Debug)]
struct Matrix<T, const N: usize> {
	data: [[T; N]; N],
}

impl<T: Default + Copy, const N: usize> Matrix<T, N> {
	const CHECK: () = assert!(N > 1 && N < 4);

	pub fn new(values: Option<[[T; N]; N]>) -> Matrix<T, N> {
		let mut data = [[T::default(); N]; N];

		if let Some(values) = values {
			for (i, row) in values.iter().enumerate() {
				data[i] = *row;
			}
		}

		Matrix { data }
	}

	pub fn set(&mut self, row: usize, col: usize, value: T) {
		self.data[row][col] = value;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_new_matrix() {
		let fill_array: [[f32; 2]; 2] = [
			[-3., 5.],
			[1., -2.]
		];
		let matrix2x2_filled = Matrix::new(Some(fill_array));
		assert_eq!(matrix2x2_filled.data, fill_array);

		let fill_array: [[f32; 3]; 3] = [
			[-3., 5., 0.],
			[1., -2., 7.],
			[1., -2., 1.]
		];
		let matrix3x3_filled = Matrix::new(Some(fill_array));
		assert_eq!(matrix3x3_filled.data, fill_array);

		let fill_array: [[f32; 4]; 4] = [
			[1., 2., 3., 4.],
			[5.5, 6.5, 7.5, 8.5],
			[9., 10., 11., 12.],
			[13.5, 14.5, 15.5, 16.5]
		];
		let matrix4x4_filled = Matrix::new(Some(fill_array));
		assert_eq!(matrix4x4_filled.data, fill_array);
	}
}
