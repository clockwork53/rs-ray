// #![allow(incomplete_features)]
// #![feature(generic_const_exprs, adt_const_params, const_generics, const_evaluatable_checked)]

use std::fmt::Debug;
use std::ops::{Add, Mul, Div, Neg, Sub};
use crate::tuple::Tuple;
use crate::vec3::EPSILON;

#[allow(dead_code)]
#[repr(usize)]
enum MatrixSize {
	_2X2 = 2,
	_3X3 = 3,
	_4X4 = 4,
}

#[allow(dead_code)]
pub enum MatrixFill<T, const N: usize> {
	Array([[T; N]; N]),
	Single(T),
}

#[derive(Debug)]
pub struct Matrix<T, const N: usize> {
	data: [[T; N]; N],
}

pub const IDENTITY_MATRIX: Matrix<u8, 4> = Matrix
{
	data: [
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
	]
};

// 4x4 * 4x4
impl<T, F> Mul<Matrix<F, 4>> for Matrix<T, 4>
	where T: Default + Copy + Add<Output=T> + Mul<Output=T>,
	      F: Default + Copy + TryInto<T>,
	      <F as TryInto<T>>::Error: Debug
{
	type Output = Self;
	fn mul(self, rhs: Matrix<F, 4>) -> Self::Output {
		let mut res = Matrix::new(None);
		for row in 0..4 {
			for col in 0..4 {
				let val =
					self.get(row, 0) * rhs.get(0, col).try_into().unwrap() +
						self.get(row, 1) * rhs.get(1, col).try_into().unwrap() +
						self.get(row, 2) * rhs.get(2, col).try_into().unwrap() +
						self.get(row, 3) * rhs.get(3, col).try_into().unwrap();
				res.set(row, col, val);
			}
		}
		res
	}
}

// 4x4 * 4x1
impl<T, F> Mul<Tuple<T>> for Matrix<F, 4>
	where T: Default + Copy + Add<Output=T> + Mul<Output=T>,
	      F: Default + Copy + TryInto<T>,
	      <F as TryInto<T>>::Error: Debug
{
	type Output = Tuple<T>;
	fn mul(self, rhs: Tuple<T>) -> Self::Output {
		let mut res = Tuple::new(None);
		for row in 0..4 {
			let val =
				self.get(row, 0).try_into().unwrap() * rhs.get(0) +
					self.get(row, 1).try_into().unwrap() * rhs.get(1) +
					self.get(row, 2).try_into().unwrap() * rhs.get(2) +
					self.get(row, 3).try_into().unwrap() * rhs.get(3);
			res.set(row, val);
		}
		res
	}
}


macro_rules! impl_matrix_n_x_n {
	(2) => {
		impl<T> Matrix<T, 2>
		where
            T: Default + Copy + Mul<Output = T> + Sub<Output = T> + Add<Output = T> + Neg<Output = T> + PartialEq,
		{
			pub fn determinant(&self) -> T {
				self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0)
			}

			#[allow(dead_code)]
			pub fn invertible(&self) -> bool {
				self.determinant() != T::default()
			}
		}
	};
    ($size:expr) => {
        impl<T> Matrix<T, $size>
        where
            T: Default + Copy + Mul<Output = T> + Div<Output=T> +
                Sub<Output = T> + Add<Output = T> + Neg<Output = T> + PartialEq, f64: From<T>, T: std::fmt::Debug
        {
            pub fn sub_matrix(&self, row: usize, col: usize) -> Matrix<T, {$size - 1}> {
                let mut sub_matrix = Matrix::new(None);
                let mut sub_row = 0;
                for c_row in 0..$size {
                    if c_row == row {
                        continue;
                    }
                    let mut sub_col = 0;
                    for c_col in 0..$size {
                        if c_col == col {
                            continue;
                        }
                        sub_matrix.set(sub_row, sub_col, self.get(c_row, c_col));
                        sub_col += 1;
                    }
                    sub_row += 1;
                }
                sub_matrix
            }

            pub fn minor(&self, row: usize, col: usize) -> T {
                self.sub_matrix(row, col).determinant()
            }

            pub fn cofactor(&self, row: usize, col: usize) -> T {
                if (row + col) % 2 == 0 {
                    self.minor(row, col)
                } else {
                    -self.minor(row, col)
                }
            }

            pub fn determinant(&self) -> T {
                let mut result = T::default();
                for col in 0..$size {
                    result = result + self.get(0, col) * self.cofactor(0, col);
                }
                result
            }

            #[allow(dead_code)]
            pub fn invertible(&self) -> bool {
                self.determinant() != T::default()
            }

            #[allow(dead_code)]
            pub fn inverse(&self) -> Option<Matrix<f64, $size>> {
				if !self.invertible() {
					return None;
				}
				let mut res = Matrix::new(None);
	            for row in 0..$size {
					for col in 0..$size {
						let cof = f64::try_from(self.cofactor(row, col)).unwrap();
						let det = f64::try_from(self.determinant()).unwrap();
						// dbg!(row, col, cof, det, cof / det);
						res.set(col, row, cof / det);
					}
				}
	            Some(res)
			}
        }
    };
}

impl_matrix_n_x_n!(2);
impl_matrix_n_x_n!(3);
impl_matrix_n_x_n!(4);


impl<T, const N: usize> PartialEq for Matrix<T, N>
	where T: Default + Copy, f64: From<T>
{
	fn eq(&self, other: &Self) -> bool {
		for i in 0..N {
			for j in 0..N {
				if (
					f64::try_from(other.get(i, j)).unwrap() -
						f64::try_from(self.get(i, j)).unwrap()
				).abs() > EPSILON as f64 {
					return false;
				}
			}
		}
		true
	}
}

// NxN
impl<T: Default + Copy, const N: usize> Matrix<T, N> {
	#[allow(dead_code)]
	const CHECK: () = assert!(N >= 2 && N <= 4);

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

	pub fn transpose(&mut self) {
		for i in 0..N {
			for j in (i + 1)..N {
				let t = self.get(i, j);
				self.set(i, j, self.get(j, i));
				self.set(j, i, t)
			}
		}
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
		assert_eq!((matrix4x4_filled_a * matrix4x4_filled_b).data, fill_array_c)
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
		assert_eq!((matrix4x4_filled_a * tuple4x1_filled_b).data, fill_array_c)
	}

	#[test]
	fn test_multiplying_by_identity_matrix() {
		let fill_array_a: [[i64; 4]; 4] = [
			[1, 2, 3, 4],
			[2, 4, 4, 2],
			[8, 6, 4, 1],
			[0, 0, 0, 1]
		];
		let fill_array_c: [[f32; 4]; 4] = [
			[1., 2., 3., 4.],
			[2., 4., 4., 2.],
			[8., 6., 4., 1.],
			[0., 0., 0., 1.]
		];
		let matrix4x4_filled_a = Matrix::new(Some(MatrixFill::Array(fill_array_a)));
		let matrix4x4_filled_c = Matrix::new(Some(MatrixFill::Array(fill_array_c)));
		assert_eq!((matrix4x4_filled_a * IDENTITY_MATRIX).data, fill_array_a);
		assert_eq!((matrix4x4_filled_c * IDENTITY_MATRIX).data, fill_array_c);

		let fill_array_b: [i64; 4] = [1, 2, 3, 1];
		let tuple4x1_filled_b = Tuple::new(Some(TupleFill::Array(fill_array_b)));

		assert_eq!((IDENTITY_MATRIX * tuple4x1_filled_b).data, fill_array_b)
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
	fn test_transpose_4x4() {
		let mut matrix = Matrix::new(Some(MatrixFill::Array([
			[1.0, 2.0, 3.0, 4.0],
			[5.0, 6.0, 7.0, 8.0],
			[9.0, 10.0, 11.0, 12.0],
			[13.0, 14.0, 15.0, 16.0],
		])));

		let expected_transposed = [
			[1.0, 5.0, 9.0, 13.0],
			[2.0, 6.0, 10.0, 14.0],
			[3.0, 7.0, 11.0, 15.0],
			[4.0, 8.0, 12.0, 16.0],
		];

		matrix.transpose();

		assert_eq!(matrix.data, expected_transposed);
	}

	#[test]
	fn test_transpose_identity() {
		let mut id = IDENTITY_MATRIX;
		id.transpose();
		assert_eq!(id.data, IDENTITY_MATRIX.data);
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

	#[test]
	fn test_determinant_matrix_2x2() {
		let matrix = Matrix::new(Some(MatrixFill::Array([
			[1., 5.],
			[-3., 2.],
		])));

		assert_eq!(matrix.determinant(), 17.);
	}

	#[test]
	fn test_determinant_matrix_3x3() {
		let matrix = Matrix::new(Some(MatrixFill::Array([
			[1., 2., 6.],
			[-5., 8., -4.],
			[2., 6., 4.],
		])));

		assert_eq!(matrix.cofactor(0, 0), 56.);
		assert_eq!(matrix.cofactor(0, 1), 12.);
		assert_eq!(matrix.cofactor(0, 2), -46.);
		assert_eq!(matrix.determinant(), -196.);
	}

	#[test]
	fn test_determinant_matrix_4x4() {
		let matrix = Matrix::new(Some(MatrixFill::Array([
			[-2., -8., 3., 5.],
			[-3., 1., 7., 3.],
			[1., 2., -9., 6.],
			[-6., 7., 7., -9.]
		])));

		assert_eq!(matrix.cofactor(0, 0), 690.);
		assert_eq!(matrix.cofactor(0, 1), 447.);
		assert_eq!(matrix.cofactor(0, 2), 210.);
		assert_eq!(matrix.cofactor(0, 3), 51.);
		assert_eq!(matrix.determinant(), -4071.);
	}

	#[test]
	fn test_sub_matrix_3x3() {
		let matrix = Matrix::new(Some(MatrixFill::Array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])));

		let expected_transposed = [
			[1.0, 3.0],
			[7.0, 9.0],
		];

		assert_eq!(matrix.sub_matrix(1, 1).data, expected_transposed);
	}

	#[test]
	fn test_sub_matrix_4x4() {
		let matrix = Matrix::new(Some(MatrixFill::Array([
			[1.0, 2.0, 3.0, 4.0],
			[5.0, 6.0, 7.0, 8.0],
			[9.0, 10.0, 11.0, 12.0],
			[13.0, 14.0, 15.0, 16.0],
		])));

		let expected_transposed = [
			[1.0, 3.0, 4.0],
			[5.0, 7.0, 8.0],
			[13.0, 15.0, 16.0],
		];

		assert_eq!(matrix.sub_matrix(2, 1).data, expected_transposed);
	}

	#[test]
	fn test_minor_matrix_3x3() {
		let matrix = Matrix::new(Some(MatrixFill::Array([
			[3, 5, 0],
			[2, -1, -7],
			[6, -1, 5],
		])));

		let sub_matrix = matrix.sub_matrix(1, 0);

		assert_eq!(sub_matrix.determinant(), 25);
		assert_eq!(matrix.minor(1, 0), 25)
	}

	#[test]
	fn test_cofactor_matrix_3x3() {
		let matrix = Matrix::new(Some(MatrixFill::Array([
			[3, 5, 0],
			[2, -1, -7],
			[6, -1, 5],
		])));

		assert_eq!(matrix.minor(0, 0), -12);
		assert_eq!(matrix.cofactor(0, 0), -12);
		assert_eq!(matrix.minor(1, 0), 25);
		assert_eq!(matrix.cofactor(1, 0), -25);
	}

	#[test]
	fn test_invertible_matrix_4x4() {
		let matrix = Matrix::new(Some(MatrixFill::Array([
			[6, 4, 4, 4],
			[5, 5, 7, 6],
			[4, -9, 3, -7],
			[9, 1, 7, -6]
		])));

		assert_eq!(matrix.invertible(), true);
	}

	#[test]
	fn test_non_invertible_matrix_4x4() {
		let matrix = Matrix::new(Some(MatrixFill::Array([
			[-4, 2, -2, -3],
			[9, 6, 2, 6],
			[0, -5, 1, -5],
			[0, 0, 0, 0]
		])));

		assert_eq!(matrix.invertible(), false);
	}

	#[test]
	fn test_inverse_matrix_4x4_a() {
		let matrix = Matrix::new(Some(MatrixFill::Array([
			[-5, 2, 6, -8],
			[1, -5, 1, 8],
			[7, 7, -6, -7],
			[1, -3, 7, 4]
		])));

		let inverse = Matrix::new(Some(MatrixFill::Array([
			[0.21805, 0.4511278, 0.24060150, -0.045112781],
			[-0.808270, -1.45676, -0.443609, 0.5206766917],
			[-0.078947, -0.2236842, -0.0526315, 0.19736842],
			[-0.522556, -0.81390, -0.3007518, 0.30639097]
		])));

		assert_eq!(matrix.invertible(), true);
		let b = matrix.inverse().unwrap();

		assert_eq!(matrix.determinant(), 532);
		//
		assert_eq!(matrix.cofactor(1, 1), -775);
		assert_eq!(b.get(1, 1), -775. / 532.);
		//
		assert_eq!(matrix.cofactor(2, 3), -160);
		assert_eq!(b.get(3, 2), -160. / 532.);
		//
		assert_eq!(matrix.cofactor(3, 2), 105);
		assert_eq!(b.get(2, 3), 105. / 532.);
		assert_eq!(b, inverse);
	}

	#[test]
	fn test_inverse_matrix_4x4_b() {
		let matrix = Matrix::new(Some(MatrixFill::Array([
			[8, -5, 9, 2],
			[7, 5, 6, 1],
			[-6, 0, 9, 6],
			[-3, 0, -9, -4]
		])));

		let inverse = Matrix::new(Some(MatrixFill::Array([
			[-0.15384615384615385, -0.15384615384615385, -0.28205128205128205, -0.5384615384615384],
			[-0.07692307692307693, 0.12307692307692308, 0.02564102564102564, 0.03076923076923077],
			[0.358974358974359, 0.358974358974359, 0.4358974358974359, 0.9230769230769231],
			[-0.6923076923076923, -0.6923076923076923, -0.7692307692307693, -1.9230769230769231]
		])));

		assert_eq!(matrix.invertible(), true);
		let b = matrix.inverse().unwrap();

		assert_eq!(b, inverse);
	}

	#[test]
	fn test_inverse_matrix_4x4_c() {
		let matrix = Matrix::new(Some(MatrixFill::Array([
			[9, 3, 0, 9],
			[-5, -2, -6, -3],
			[-4, 9, 6, 4],
			[-7, 6, 6, 2]
		])));

		let inverse = Matrix::new(Some(MatrixFill::Array([
			[-0.040740740740740744, -0.07777777777777778, 0.14444444444444443, -0.2222222222222222],
			[-0.07777777777777778, 0.03333333333333333, 0.36666666666666664, -0.3333333333333333],
			[-0.029012345679012345, -0.14629629629629629, -0.10925925925925926, 0.12962962962962962],
			[0.17777777777777778, 0.06666666666666667, -0.26666666666666666, 0.3333333333333333]
		])));

		assert_eq!(matrix.invertible(), true);
		let b = matrix.inverse().unwrap();

		assert_eq!(b, inverse);
	}

	#[test]
	fn test_inverse_matrix_4x4_d() {
		let matrix_a = Matrix::new(Some(MatrixFill::Array([
			[3, -9, 7, 3],
			[3, -8, 2, -9],
			[-4, 4, 4, 1],
			[-6, 5, -1, 1]
		])));
		let matrix_b = Matrix::new(Some(MatrixFill::Array([
			[8, 2, 2, 2],
			[3, -1, 7, 0],
			[7, 0, 5, 4],
			[6, -2, 0, 5]
		])));
		let matrix_c = matrix_a * matrix_b;
		assert_eq!(matrix_c * matrix_b.inverse().unwrap(), matrix_a);
	}
}
