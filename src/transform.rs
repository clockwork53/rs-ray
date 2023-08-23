use crate::matrix::{Matrix, MatrixFill};
use crate::misc::{Float};

#[allow(dead_code)]
pub fn translation(x: Float, y: Float, z: Float) -> Matrix<Float, 4> {
	Matrix::new(Some(MatrixFill::Array([
		[1., 0., 0., x],
		[0., 1., 0., y],
		[0., 0., 1., z],
		[0., 0., 0., 1.],
	])))
}

#[allow(dead_code)]
pub fn scaling(x: Float, y: Float, z: Float) -> Matrix<Float, 4> {
	Matrix::new(Some(MatrixFill::Array([
		[x, 0., 0., 0.],
		[0., y, 0., 0.],
		[0., 0., z, 0.],
		[0., 0., 0., 1.],
	])))
}

#[allow(dead_code)]
pub fn rotation_x(radians: Float) -> Matrix<Float, 4> {
	let cos_r = radians.cos();
	let sin_r = radians.sin();
	Matrix::new(Some(MatrixFill::Array([
		[1.,    0.,    0.,     0.],
		[0.,    cos_r, -sin_r, 0.],
		[0.,    sin_r, cos_r,  0.],
		[0.,    0.,    0.,     1.],
	])))
}

#[allow(dead_code)]
pub fn rotation_y(radians: Float) -> Matrix<Float, 4> {
	let cos_r = radians.cos();
	let sin_r = radians.sin();
	Matrix::new(Some(MatrixFill::Array([
		[cos_r,    0.,      sin_r,     0.],
		[0.,       1.,      0.,        0.],
		[-sin_r,   0.,      cos_r,     0.],
		[0.,       0.,      0.,        1.],
	])))
}

#[allow(dead_code)]
pub fn rotation_z(radians: Float) -> Matrix<Float, 4> {
	let cos_r = radians.cos();
	let sin_r = radians.sin();
	Matrix::new(Some(MatrixFill::Array([
		[cos_r,    -sin_r,  0.,        0.],
		[sin_r,    cos_r,   0.,        0.],
		[0.,       0.,      1.,        0.],
		[0.,       0.,      0.,        1.],
	])))
}

#[allow(dead_code)]
pub fn shearing(x_y: Float, x_z: Float, y_x: Float, y_z: Float, z_x: Float, z_y: Float) -> Matrix<Float, 4> {
	Matrix::new(Some(MatrixFill::Array([
		[1.,    x_y,    x_z,    0.],
		[y_x,   1.,     y_z,    0.],
		[z_x,   z_y,    1.,     0.],
		[0.,    0.,     0.,     1.],
	])))
}


#[cfg(test)]
mod tests {
	use crate::matrix::IDENTITY_MATRIX;
	use crate::misc::PI;
	use crate::vec4::{point, vector};
	use super::*;

	#[test]
	fn test_multiplying_by_translation_matrix() {
		let transform = translation(5., -3., 2.);
		let p = point(-3., 4., 5.);
		assert_eq!(transform * p, point(2., 1., 7.))
	}

	#[test]
	fn test_multiplying_by_the_inverse_translation_matrix() {
		let transform = translation(5., -3., 2.);
		let inv = transform.inverse().unwrap();
		let p = point(-3., 4., 5.);
		assert_eq!(inv * p, point(-8., 7., 3.))
	}

	#[test]
	fn test_translation_does_not_affect_vectors() {
		let transform = translation(5., -3., 2.);
		let v = vector(-3., 4., 5.);
		assert_eq!(transform * v, v)
	}

	#[test]
	fn test_scaling_a_point() {
		let transform = scaling(2., 3., 4.);
		let p = point(-4., 6., 8.);
		assert_eq!(transform * p, point(-8., 18., 32.))
	}

	#[test]
	fn test_scaling_a_vector() {
		let transform = scaling(2., 3., 4.);
		let v = vector(-4., 6., 8.);
		assert_eq!(transform * v, vector(-8., 18., 32.))
	}

	#[test]
	fn test_multiplying_by_the_inverse_scaling_matrix() {
		let transform = scaling(2., 3., 4.);
		let inv = transform.inverse().unwrap();
		let v = vector(-4., 6., 8.);
		assert_eq!(inv * v, vector(-2., 2., 2.))
	}

	#[test]
	fn test_reflection_is_scaling_by_negative_value() {
		let transform = scaling(-1., 1., 1.);
		let inv = transform.inverse().unwrap();
		let p = point(2., 3., 4.);
		assert_eq!(inv * p, point(-2., 3., 4.))
	}

	#[test]
	fn test_rotating_around_x_axis() {
		let p = point(0., 1., 0.);
		let half_q = rotation_x(PI / 4.);
		let full_q = rotation_x(PI / 2.);
		assert_eq!(half_q * p, point(0., 2f32.sqrt()/2., 2f32.sqrt()/2.));
		assert_eq!(full_q * p, point(0., 0., 1.));
	}

	#[test]
	fn test_inverse_of_rotating_around_x_axis() {
		let p = point(0., 1., 0.);
		let half_q = rotation_x(PI / 4.);
		let inverse = half_q.inverse().unwrap();
		assert_eq!(inverse * p, point(0., 2f32.sqrt()/2., -2f32.sqrt()/2.));
	}

	#[test]
	fn test_rotating_around_y_axis() {
		let p = point(0., 0., 1.);
		let half_q = rotation_y(PI / 4.);
		let full_q = rotation_y(PI / 2.);
		assert_eq!(half_q * p, point(2f32.sqrt()/2., 0., 2f32.sqrt()/2.));
		assert_eq!(full_q * p, point(1., 0., 0.));
	}

	#[test]
	fn test_rotating_around_z_axis() {
		let p = point(0., 1., 0.);
		let half_q = rotation_z(PI / 4.);
		let full_q = rotation_z(PI / 2.);
		assert_eq!(half_q * p, point(-2f32.sqrt()/2., 2f32.sqrt()/2., 0.));
		assert_eq!(full_q * p, point(-1., 0., 0.));
	}

	#[test]
	fn test_shearing_moves_x_in_proportion_to_y() {
		let transform = shearing(1., 0., 0., 0., 0., 0.);
		let p = point(2., 3., 4.);
		assert_eq!(transform * p, point(5., 3., 4.));
	}

	#[test]
	fn test_shearing_moves_x_in_proportion_to_z() {
		let transform = shearing(0., 1., 0., 0., 0., 0.);
		let p = point(2., 3., 4.);
		assert_eq!(transform * p, point(6., 3., 4.));
	}

	#[test]
	fn test_shearing_moves_y_in_proportion_to_x() {
		let transform = shearing(0., 0., 1., 0., 0., 0.);
		let p = point(2., 3., 4.);
		assert_eq!(transform * p, point(2., 5., 4.));
	}

	#[test]
	fn test_shearing_moves_y_in_proportion_to_z() {
		let transform = shearing(0., 0., 0., 1., 0., 0.);
		let p = point(2., 3., 4.);
		assert_eq!(transform * p, point(2., 7., 4.));
	}

	#[test]
	fn test_shearing_moves_z_in_proportion_to_x() {
		let transform = shearing(0., 0., 0., 0., 1., 0.);
		let p = point(2., 3., 4.);
		assert_eq!(transform * p, point(2., 3., 6.));
	}

	#[test]
	fn test_shearing_moves_z_in_proportion_to_y() {
		let transform = shearing(0., 0., 0., 0., 0., 1.);
		let p = point(2., 3., 4.);
		assert_eq!(transform * p, point(2., 3., 7.));
	}

	#[test]
	fn test_individual_transformations_sequence() {
		let p = point(1., 0., 1.);
		let a = rotation_x(PI / 2.);
		let b = scaling(5., 5., 5.);
		let c = translation(10., 5., 7.);

		let p2 = a * p;
		assert_eq!(p2, point(1., -1., 0.));

		let p3 = b * p2;
		assert_eq!(p3, point(5., -5., 0.));

		let p4 = c * p3;
		assert_eq!(p4, point(15., 0., 7.));
	}

	#[test]
	fn test_chained_transformations_in_reverse_order() {
		let p = point(1., 0., 1.);
		let a = rotation_x(PI / 2.);
		let b = scaling(5., 5., 5.);
		let c = translation(10., 5., 7.);

		let t = c * &(b * &a);
		assert_eq!(t * p, point(15., 0., 7.))
	}

	#[test]
	fn test_chained_fluid_transformations() {
		let p = point(1., 0., 1.);
		let t = IDENTITY_MATRIX
			.rotate_x(PI / 2.)
			.scale(5., 5., 5.)
			.translate(10., 5., 7.);

		assert_eq!(t * p, point(15., 0., 7.))
	}
}