use std::ops::{Add, Div, Mul, Neg, Sub};
use crate::misc::{EPSILON, Float};
use crate::tuple::{Tuple};

#[derive(Debug, Clone, Copy)]
pub struct Vec4 {
	pub x: Float,
	pub y: Float,
	pub z: Float,
	pub w: Float,
}

impl From<Tuple<Float>> for Vec4 {
	fn from(value: Tuple<Float>) -> Self {
		Vec4 {
			x: value.data[0],
			y: value.data[1],
			z: value.data[2],
			w: value.data[3],
		}
	}
}

impl Add for Vec4 {
	type Output = Self;
	fn add(self, rhs: Self) -> Self::Output {
		if self.w == 1. && rhs.w == 1. {
			panic!("Illegal Operation: Adding two points!");
		}
		Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z, w: self.w + rhs.w }
	}
}

impl Sub for Vec4 {
	type Output = Self;
	fn sub(self, rhs: Self) -> Self::Output {
		if self.w == 0. && rhs.w == 1. {
			panic!("Illegal Operation: Subtracting a point from a vector!");
		}
		Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z, w: self.w - rhs.w }
	}
}

impl Neg for Vec4 {
	type Output = Self;
	fn neg(self) -> Self::Output {
		if self.w == 1. {
			panic!("Illegal Operation: Negating a point!");
		}
		Self { x: -self.x, y: -self.y, z: -self.z, w: -self.w }
	}
}

impl Mul<Float> for Vec4 {
	type Output = Self;
	fn mul(self, rhs: Float) -> Self::Output {
		Self { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs, w: self.w * rhs }
	}
}

impl Div<Float> for Vec4 {
	type Output = Self;
	fn div(self, rhs: Float) -> Self::Output {
		if rhs == 0. {
			panic!("Illegal Operation: Division by zero!");
		}
		Self { x: self.x / rhs, y: self.y / rhs, z: self.z / rhs, w: self.w / rhs }
	}
}

impl PartialEq for Vec4 {
	fn eq(&self, other: &Self) -> bool {
		Float::abs(self.x - other.x) < EPSILON &&
		Float::abs(self.y - other.y) < EPSILON &&
		Float::abs(self.z - other.z) < EPSILON &&
		Float::abs(self.w - other.w) < EPSILON
	}
}

impl Vec4 {
	pub fn magnitude(self) -> Float {
		let x2 = self.x * self.x;
		let y2 = self.y * self.y;
		let z2 = self.z * self.z;
		let w2 = self.w * self.w;
		(x2 + y2 + z2 + w2).sqrt()
	}

	pub fn normalize(self) -> Vec4 {
		let magnitude = self.magnitude();
		Vec4 { x: self.x / magnitude, y: self.y / magnitude, z: self.z / magnitude, w: self.w / magnitude }
	}

	#[allow(dead_code)]
	pub fn dot(self, other: Vec4) -> Float {
		(self.x * other.x) +
			(self.y * other.y) +
			(self.z * other.z) +
			(self.w * other.w)
	}

	#[allow(dead_code)]
	pub fn cross(self, other: Vec4) -> Vec4 {
		if self.w != 0. || other.w != 0. {
			panic!("Illegal Operation: Cross product involving a point!");
		}
		vector(
			(self.y * other.z) - (self.z * other.y),
			(self.z * other.x) - (self.x * other.z),
			(self.x * other.y) - (self.y * other.x),
		)
	}
}

pub fn point(x: Float, y: Float, z: Float) -> Vec4 {
	Vec4 { x, y, z, w: 1. }
}

pub fn vector(x: Float, y: Float, z: Float) -> Vec4 {
	Vec4 { x, y, z, w: 0. }
}


#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_tuple_point() {
		let a = Vec4 { x: 4.3, y: -4.2, z: 3.1, w: 1. };
		assert_eq!(a.x, 4.3);
		assert_eq!(a.y, -4.2);
		assert_eq!(a.z, 3.1);
		assert_eq!(a.w, 1.);
		assert_ne!(a.w, 0.);
	}

	#[test]
	fn test_tuple_vector() {
		let a = Vec4 { x: 4.3, y: -4.2, z: 3.1, w: 0. };
		assert_eq!(a.x, 4.3);
		assert_eq!(a.y, -4.2);
		assert_eq!(a.z, 3.1);
		assert_eq!(a.w, 0.);
		assert_ne!(a.w, 1.);
	}

	#[test]
	fn test_new_point() {
		let p = point(4., -4., 3.);
		assert_eq!(p, Vec4 { x: 4., y: -4., z: 3., w: 1. })
	}

	#[test]
	fn test_new_vector() {
		let v = vector(4., -4., 3.);
		assert_eq!(v, Vec4 { x: 4., y: -4., z: 3., w: 0. })
	}

	#[test]
	fn test_adding_tuples() {
		let a1 = Vec4 { x: 3., y: -2., z: 1., w: 1. };
		let a2 = Vec4 { x: -2., y: 3., z: 1., w: 0. };
		assert_eq!(a1 + a2, Vec4 { x: 1., y: 1., z: 2., w: 1. })
	}

	#[test]
	fn test_subtracting_two_points() {
		let p1 = Vec4 { x: 3., y: 2., z: 1., w: 1. };
		let p2 = Vec4 { x: 5., y: 6., z: 7., w: 1. };
		assert_eq!(p1 - p2, Vec4 { x: -2., y: -4., z: -6., w: 0. })
	}

	#[test]
	fn test_subtracting_a_vector_from_a_point() {
		let p = Vec4 { x: 3., y: 2., z: 1., w: 1. };
		let v = Vec4 { x: 5., y: 6., z: 7., w: 0. };
		assert_eq!(p - v, Vec4 { x: -2., y: -4., z: -6., w: 1. })
	}

	#[test]
	fn test_subtracting_two_vectors() {
		let v1 = Vec4 { x: 3., y: 2., z: 1., w: 0. };
		let v2 = Vec4 { x: 5., y: 6., z: 7., w: 0. };
		assert_eq!(v1 - v2, Vec4 { x: -2., y: -4., z: -6., w: 0. })
	}

	#[test]
	fn test_subtracting_a_vector_from_the_zero_vector() {
		let zero = vector(0., 0., 0.);
		let v = vector(1., -2., 3.);
		assert_eq!(zero - v, vector(-1., 2., -3.))
	}

	#[test]
	fn test_negating_a_tuple() {
		let a = Vec4 { x: 1., y: -2., z: 3., w: -4. };
		assert_eq!(-a, Vec4 { x: -1., y: 2., z: -3., w: 4. })
	}

	#[test]
	fn test_multiplying_a_tuple_by_a_scalar() {
		let a = Vec4 { x: 1., y: -2., z: 3., w: -4. };
		assert_eq!(a * 3.5, Vec4 { x: 3.5, y: -7., z: 10.5, w: -14. })
	}

	#[test]
	fn test_multiplying_a_tuple_by_a_fraction() {
		let a = Vec4 { x: 1., y: -2., z: 3., w: -4. };
		assert_eq!(a * 0.5, Vec4 { x: 0.5, y: -1., z: 1.5, w: -2. })
	}

	#[test]
	fn test_dividing_a_tuple_by_a_scalar() {
		let a = Vec4 { x: 1., y: -2., z: 3., w: -4. };
		assert_eq!(a / 2., Vec4 { x: 0.5, y: -1., z: 1.5, w: -2. })
	}

	#[test]
	fn test_magnitude_of_vector_1_0_0() {
		let v = vector(1., 0., 0.);
		assert_eq!(v.magnitude(), 1.)
	}

	#[test]
	fn test_magnitude_of_vector_0_1_0() {
		let v = vector(0., 1., 0.);
		assert_eq!(v.magnitude(), 1.)
	}

	#[test]
	fn test_magnitude_of_vector_0_0_1() {
		let v = vector(0., 0., 1.);
		assert_eq!(v.magnitude(), 1.)
	}

	#[test]
	fn test_magnitude_of_vector_1_2_3() {
		let v = vector(1., 2., 3.);
		assert_eq!(v.magnitude(), 14f32.sqrt())
	}

	#[test]
	fn test_magnitude_of_vector_n1_n2_n3() {
		let v = vector(-1., -2., -3.);
		assert_eq!(v.magnitude(), 14f32.sqrt())
	}

	#[test]
	fn test_normalizing_vector_4_0_0() {
		let v = vector(4., 0., 0.);
		assert_eq!(v.normalize(), vector(1., 0., 0.))
	}

	#[test]
	fn test_normalizing_vector_1_2_3() {
		let v = vector(1., 2., 3.);
		assert_eq!(v.normalize(), vector(0.26726, 0.53452, 0.80178))
	}

	#[test]
	fn test_magnitude_of_normalized_vector_1_2_3() {
		let v = vector(1., 2., 3.);
		assert!((v.normalize().magnitude() - 1.).abs() < EPSILON)
	}

	#[test]
	fn test_dot_product_of_two_tuples() {
		let a = vector(1., 2., 3.);
		let b = vector(2., 3., 4.);
		assert_eq!(a.dot(b), 20.)
	}

	#[test]
	fn test_cross_product_of_two_tuples() {
		let a = vector(1., 2., 3.);
		let b = vector(2., 3., 4.);
		assert_eq!(a.cross(b), vector(-1., 2., -1.));
		assert_eq!(b.cross(a), vector(1., -2., 1.));
	}
}
