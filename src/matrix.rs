// #![allow(incomplete_features)]
// #![feature(generic_const_exprs, adt_const_params, const_generics, const_evaluatable_checked)]

#[allow(dead_code)]
#[repr(usize)]
enum MatrixSize {
	_2X2 = 2,
	_3X3 = 3,
	_4X4 = 4,
}

#[allow(dead_code)]
enum MatrixFill<T, const N: usize> {
	Array([[T; N]; N]),
	Single(T),
}

#[derive(Debug)]
struct Matrix<T, const N: usize> {
	data: [[T; N]; N],
}

impl<T: Default + Copy, const N: usize> Matrix<T, N> {
	#[allow(dead_code)]
	const CHECK: () = assert!(N > 1 && N < 4);

	#[allow(dead_code)]
	pub fn new(values: Option<MatrixFill<T, N>>) -> Matrix<T, N> {
		let mut data = [[T::default(); N]; N];

		if let Some(fill) = values {
			match fill {
				MatrixFill::Array(values) => {
					for (i, row) in values.iter().enumerate() {
						data[i] = *row;
					}
				}
				MatrixFill::Single(value) => {
					for row in &mut data {
						for elem in row.iter_mut() {
							*elem = value;
						}
					}
				}
			}
		}

		Matrix { data }
	}

	#[allow(dead_code)]
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
		let matrix2x2_filled = Matrix::new(Some(MatrixFill::Array(fill_array)));
		assert_eq!(matrix2x2_filled.data, fill_array);

		let fill_array: [[f32; 3]; 3] = [
			[-3., 5., 0.],
			[1., -2., 7.],
			[1., -2., 1.]
		];
		let matrix3x3_filled = Matrix::new(Some(MatrixFill::Array(fill_array)));
		assert_eq!(matrix3x3_filled.data, fill_array);

		let fill_array: [[f32; 4]; 4] = [
			[1., 2., 3., 4.],
			[5.5, 6.5, 7.5, 8.5],
			[9., 10., 11., 12.],
			[13.5, 14.5, 15.5, 16.5]
		];
		let matrix4x4_filled = Matrix::new(Some(MatrixFill::Array(fill_array)));
		assert_eq!(matrix4x4_filled.data, fill_array);
	}

	#[test]
	fn test_set_function() {
		let mut matrix2x2 = Matrix::new(None);

		matrix2x2.set(0, 0, 1.0);
		matrix2x2.set(0, 1, 2.0);
		matrix2x2.set(1, 0, 3.0);
		matrix2x2.set(1, 1, 4.0);

		assert_eq!(matrix2x2.data, [[1.0, 2.0], [3.0, 4.0]]);
	}
}
