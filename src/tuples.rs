use std::ops::{Add, Div, Mul, Neg, Sub};

pub const EPSILON: f32 = 0.00001;

#[derive(Debug, Clone, Copy)]
pub struct Tuple {
	pub x: f32,
	pub y: f32,
	pub z: f32,
	pub w: f32,
}

impl Add for Tuple {
	type Output = Self;
	fn add(self, rhs: Self) -> Self::Output {
		if self.w == 1f32 && rhs.w == 1f32 {
			panic!("Illegal Operation: Adding two points!");
		}
		Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z, w: self.w + rhs.w }
	}
}

impl Sub for Tuple {
	type Output = Self;
	fn sub(self, rhs: Self) -> Self::Output {
		if self.w == 0f32 && rhs.w == 1f32 {
			panic!("Illegal Operation: Subtracting a point from a vector!");
		}
		Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z, w: self.w - rhs.w }
	}
}

impl Neg for Tuple {
	type Output = Self;
	fn neg(self) -> Self::Output {
		if self.w == 1f32 {
			panic!("Illegal Operation: Negating a point!");
		}
		Self { x: -self.x, y: -self.y, z: -self.z, w: -self.w }
	}
}

impl Mul<f32> for Tuple {
	type Output = Self;
	fn mul(self, rhs: f32) -> Self::Output {
		Self { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs, w: self.w * rhs }
	}
}

impl Div<f32> for Tuple {
	type Output = Self;
	fn div(self, rhs: f32) -> Self::Output {
		if rhs == 0f32 {
			panic!("Illegal Operation: Division by zero!");
		}
		Self { x: self.x / rhs, y: self.y / rhs, z: self.z / rhs, w: self.w / rhs }
	}
}

impl PartialEq for Tuple {
	fn eq(&self, other: &Self) -> bool {
		f32::abs(self.x - other.x) < EPSILON &&
			f32::abs(self.y - other.y) < EPSILON &&
			f32::abs(self.z - other.z) < EPSILON &&
			f32::abs(self.w - other.w) < EPSILON
	}
}

impl Tuple {
	pub fn magnitude(self) -> f32 {
		let x2 = self.x * self.x;
		let y2 = self.y * self.y;
		let z2 = self.z * self.z;
		let w2 = self.w * self.w;
		(x2 + y2 + z2 + w2).sqrt()
	}

	pub fn normalize(self) -> Tuple {
		let magnitude = self.magnitude();
		Tuple { x: self.x / magnitude, y: self.y / magnitude, z: self.z / magnitude, w: self.w / magnitude }
	}

	pub fn dot(self, other: Tuple) -> f32 {
		(self.x * other.x) +
			(self.y * other.y) +
			(self.z * other.z) +
			(self.w * other.w)
	}

	pub fn cross(self, other: Tuple) -> Tuple {
		if self.w != 0f32 || other.w != 0f32 {
			panic!("Illegal Operation: Cross product involving a point!");
		}
		vector(
			(self.y * other.z) - (self.z * other.y),
			(self.z * other.x) - (self.x * other.z),
			(self.x * other.y) - (self.y * other.x),
		)
	}
}

pub fn point(x: f32, y: f32, z: f32) -> Tuple {
	Tuple { x, y, z, w: 1f32 }
}

pub fn vector(x: f32, y: f32, z: f32) -> Tuple {
	Tuple { x, y, z, w: 0f32 }
}


#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_tuple_point() {
		let a = Tuple { x: 4.3, y: -4.2, z: 3.1, w: 1f32 };
		assert_eq!(a.x, 4.3);
		assert_eq!(a.y, -4.2);
		assert_eq!(a.z, 3.1);
		assert_eq!(a.w, 1f32);
		assert_ne!(a.w, 0f32);
	}

	#[test]
	fn test_tuple_vector() {
		let a = Tuple { x: 4.3, y: -4.2, z: 3.1, w: 0f32 };
		assert_eq!(a.x, 4.3);
		assert_eq!(a.y, -4.2);
		assert_eq!(a.z, 3.1);
		assert_eq!(a.w, 0f32);
		assert_ne!(a.w, 1f32);
	}

	#[test]
	fn test_new_point() {
		let p = point(4f32, -4f32, 3f32);
		assert_eq!(p, Tuple { x: 4f32, y: -4f32, z: 3f32, w: 1f32 })
	}

	#[test]
	fn test_new_vector() {
		let v = vector(4f32, -4f32, 3f32);
		assert_eq!(v, Tuple { x: 4f32, y: -4f32, z: 3f32, w: 0f32 })
	}

	#[test]
	fn test_adding_tuples() {
		let a1 = Tuple { x: 3f32, y: -2f32, z: 1f32, w: 1f32 };
		let a2 = Tuple { x: -2f32, y: 3f32, z: 1f32, w: 0f32 };
		assert_eq!(a1 + a2, Tuple { x: 1f32, y: 1f32, z: 2f32, w: 1f32 })
	}

	#[test]
	fn test_subtracting_two_points() {
		let p1 = Tuple { x: 3f32, y: 2f32, z: 1f32, w: 1f32 };
		let p2 = Tuple { x: 5f32, y: 6f32, z: 7f32, w: 1f32 };
		assert_eq!(p1 - p2, Tuple { x: -2f32, y: -4f32, z: -6f32, w: 0f32 })
	}

	#[test]
	fn test_subtracting_a_vector_from_a_point() {
		let p = Tuple { x: 3f32, y: 2f32, z: 1f32, w: 1f32 };
		let v = Tuple { x: 5f32, y: 6f32, z: 7f32, w: 0f32 };
		assert_eq!(p - v, Tuple { x: -2f32, y: -4f32, z: -6f32, w: 1f32 })
	}

	#[test]
	fn test_subtracting_two_vectors() {
		let v1 = Tuple { x: 3f32, y: 2f32, z: 1f32, w: 0f32 };
		let v2 = Tuple { x: 5f32, y: 6f32, z: 7f32, w: 0f32 };
		assert_eq!(v1 - v2, Tuple { x: -2f32, y: -4f32, z: -6f32, w: 0f32 })
	}

	#[test]
	fn test_subtracting_a_vector_from_the_zero_vector() {
		let zero = vector(0f32, 0f32, 0f32);
		let v = vector(1f32, -2f32, 3f32);
		assert_eq!(zero - v, vector(-1f32, 2f32, -3f32))
	}

	#[test]
	fn test_negating_a_tuple() {
		let a = Tuple { x: 1f32, y: -2f32, z: 3f32, w: -4f32 };
		assert_eq!(-a, Tuple { x: -1f32, y: 2f32, z: -3f32, w: 4f32 })
	}

	#[test]
	fn test_multiplying_a_tuple_by_a_scalar() {
		let a = Tuple { x: 1f32, y: -2f32, z: 3f32, w: -4f32 };
		assert_eq!(a * 3.5f32, Tuple { x: 3.5f32, y: -7f32, z: 10.5f32, w: -14f32 })
	}

	#[test]
	fn test_multiplying_a_tuple_by_a_fraction() {
		let a = Tuple { x: 1f32, y: -2f32, z: 3f32, w: -4f32 };
		assert_eq!(a * 0.5f32, Tuple { x: 0.5f32, y: -1f32, z: 1.5f32, w: -2f32 })
	}

	#[test]
	fn test_dividing_a_tuple_by_a_scalar() {
		let a = Tuple { x: 1f32, y: -2f32, z: 3f32, w: -4f32 };
		assert_eq!(a / 2f32, Tuple { x: 0.5f32, y: -1f32, z: 1.5f32, w: -2f32 })
	}

	#[test]
	fn test_magnitude_of_vector_1_0_0() {
		let v = vector(1f32, 0f32, 0f32);
		assert_eq!(v.magnitude(), 1f32)
	}

	#[test]
	fn test_magnitude_of_vector_0_1_0() {
		let v = vector(0f32, 1f32, 0f32);
		assert_eq!(v.magnitude(), 1f32)
	}

	#[test]
	fn test_magnitude_of_vector_0_0_1() {
		let v = vector(0f32, 0f32, 1f32);
		assert_eq!(v.magnitude(), 1f32)
	}

	#[test]
	fn test_magnitude_of_vector_1_2_3() {
		let v = vector(1f32, 2f32, 3f32);
		assert_eq!(v.magnitude(), 14f32.sqrt())
	}

	#[test]
	fn test_magnitude_of_vector_n1_n2_n3() {
		let v = vector(-1f32, -2f32, -3f32);
		assert_eq!(v.magnitude(), 14f32.sqrt())
	}

	#[test]
	fn test_normalizing_vector_4_0_0() {
		let v = vector(4f32, 0f32, 0f32);
		assert_eq!(v.normalize(), vector(1f32, 0f32, 0f32))
	}

	#[test]
	fn test_normalizing_vector_1_2_3() {
		let v = vector(1f32, 2f32, 3f32);
		assert_eq!(v.normalize(), vector(0.26726, 0.53452, 0.80178))
	}

	#[test]
	fn test_magnitude_of_normalized_vector_1_2_3() {
		let v = vector(1f32, 2f32, 3f32);
		assert!((v.normalize().magnitude() - 1f32).abs() < EPSILON)
	}

	#[test]
	fn test_dot_product_of_two_tuples() {
		let a = vector(1f32, 2f32, 3f32);
		let b = vector(2f32, 3f32, 4f32);
		assert_eq!(a.dot(b), 20f32)
	}

	#[test]
	fn test_cross_product_of_two_tuples() {
		let a = vector(1f32, 2f32, 3f32);
		let b = vector(2f32, 3f32, 4f32);
		assert_eq!(a.cross(b), vector(-1f32, 2f32, -1f32));
		assert_eq!(b.cross(a), vector(1f32, -2f32, 1f32));
	}
}
