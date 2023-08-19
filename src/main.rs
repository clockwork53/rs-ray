use std::ops::{Add, Neg, Sub};

const EPSILON: f32 = 0.00001;

#[derive(Debug, Clone, Copy)]
struct Tuple {
	x: f32,
	y: f32,
	z: f32,
	w: f32,
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

impl PartialEq for Tuple {
	fn eq(&self, other: &Self) -> bool {
		f32::abs(self.x - other.x) < EPSILON &&
			f32::abs(self.y - other.y) < EPSILON &&
			f32::abs(self.z - other.z) < EPSILON &&
			f32::abs(self.w - other.w) < EPSILON
	}
}

fn point(x: f32, y: f32, z: f32) -> Tuple {
	Tuple { x, y, z, w: 1f32 }
}

fn vector(x: f32, y: f32, z: f32) -> Tuple {
	Tuple { x, y, z, w: 0f32 }
}


fn main() {
	println!("Hello, world!");
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
}
