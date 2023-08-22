// #![allow(incomplete_features)]
// #![feature(generic_const_exprs, adt_const_params, const_generics, const_evaluatable_checked)]

use std::ops::{Add, Mul};
use crate::tuple::Tuple;

#[allow(dead_code)]
#[repr(usize)]
enum MatrixSize {
	_2X2 = 2,
	_3X3 = 3,
	_4X4 = 4,
}

pub enum MatrixFill<T, const N: usize> {
	Array([[T; N]; N]),
	Single(T),
}

#[derive(Debug, PartialEq)]
struct Matrix<T, const N: usize> {
	data: [[T; N]; N],
}

impl<T: Default + Copy + Add<Output=T> + Mul<Output=T>> Mul<Matrix<T, 4>> for Matrix<T, 4> {
	type Output = Self;
	fn mul(self, rhs: Matrix<T, 4>) -> Self::Output {
		let mut res = Matrix::new(None);
		for row in 0..=3 {
			for col in 0..=3 {
				let val =
					(self.data[row][0] * rhs.data[0][col]) +
						(self.data[row][1] * rhs.data[1][col]) +
						(self.data[row][2] * rhs.data[2][col]) +
						(self.data[row][3] * rhs.data[3][col]);
				res.set(row, col, val);
			}
		}
		res
	}
}

impl<T: Default + Copy + Add<Output=T> + Mul<Output=T>> Mul<Tuple<T>> for Matrix<T, 4> {
	type Output = Tuple<T>;
	fn mul(self, rhs: Tuple<T>) -> Self::Output {
		let mut res = Tuple::new(None);
		for row in 0..=3 {
			let val =
				(self.data[row][0] * rhs.get(0)) +
					(self.data[row][1] * rhs.get(1)) +
					(self.data[row][2] * rhs.get(2)) +
					(self.data[row][3] * rhs.get(3));
			res.set(row, val);
		}
		res
	}
}

impl<T: Default + Copy, const N: usize> Matrix<T, N> {
	#[allow(dead_code)]
	const CHECK: () = assert!(N > 1 && N < 4);

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

	pub fn set(&mut self, row: usize, col: usize, value: T) {
		self.data[row][col] = value;
	}
	pub fn get(&self, row: usize, col: usize) -> T {
		self.data[row][col]
	}
}

#[cfg(test)]
mod tests {
	use crate::tuple::TupleFill;
	use super::*;

	#[test]
	fn test_matrix_equality() {
		let matrix1 = Matrix {
			data: [
				[1, 2, 3],
				[4, 5, 6],
				[7, 8, 9],
			],
		};
		let matrix1f = Matrix {
			data: [
				[1., 2., 3678678678.555555273786767896789676796796796796795555555555555555555555555],
				[4., 5., 6.],
				[7., 8., 9.],
			],
		};

		let matrix2 = Matrix {
			data: [
				[1, 2, 3],
				[4, 5, 6],
				[7, 8, 9],
			],
		};
		let matrix2f = Matrix {
			data: [
				[1., 2., 3678678678.555555273786767896789676796796796796795555555555555555555555555],
				[4., 5., 6.],
				[7., 8., 9.],
			],
		};

		let matrix3 = Matrix {
			data: [
				[1, 2, 3],
				[4, 5, 6],
				[7, 8, 10],
			],
		};
		let matrix3f = Matrix {
			data: [
				[1., 2., 3678678678.555555273786767896789676796796796796795555555555555555555555551],
				[4., 5., 6.],
				[7., 8., 10.],
			],
		};

		assert_eq!(matrix1, matrix2);
		assert_eq!(matrix1f, matrix2f);
		assert_ne!(matrix1, matrix3);
		assert_ne!(matrix1f, matrix3f);
	}

	#[test]
	fn test_multiplying_two_matrices() {
		let fill_array_a: [[i64; 4]; 4] = [
			[1, 2, 3, 4],
			[5, 6, 7, 8],
			[9, 8, 7, 6],
			[5, 4, 3, 2]
		];
		let matrix4x4_filled_a = Matrix::new(Some(MatrixFill::Array(fill_array_a)));

		let fill_array_b: [[i64; 4]; 4] = [
			[-2, 1, 2, 3],
			[3, 2, 1, -1],
			[4, 3, 6, 5],
			[1, 2, 7, 8]
		];
		let matrix4x4_filled_b = Matrix::new(Some(MatrixFill::Array(fill_array_b)));

		let fill_array_c: [[i64; 4]; 4] = [
			[20, 22, 50, 48],
			[44, 54, 114, 108],
			[40, 58, 110, 102],
			[16, 26, 46, 42]
		];
		assert_eq!(matrix4x4_filled_a.mul(matrix4x4_filled_b).data, fill_array_c)
	}

	#[test]
	fn test_multiplying_a_matrix_with_a_tuple() {
		let fill_array_a: [[i64; 4]; 4] = [
			[1, 2, 3, 4],
			[2, 4, 4, 2],
			[8, 6, 4, 1],
			[0, 0, 0, 1]
		];
		let matrix4x4_filled_a = Matrix::new(Some(MatrixFill::Array(fill_array_a)));

		let fill_array_b: [i64; 4] = [1, 2, 3, 1];
		let tuple4x1_filled_b = Tuple::new(Some(TupleFill::Array(fill_array_b)));

		let fill_array_c: [i64; 4] = [18, 24, 33, 1];
		assert_eq!(matrix4x4_filled_a.mul(tuple4x1_filled_b).data, fill_array_c)
	}

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
