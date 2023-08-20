use std::ops::{Add, Div, Mul, Neg, Sub};

pub const EPSILON: f32 = 0.00001;

#[derive(Debug, Clone, Copy)]
pub struct Vec3 {
	pub r: f32,
	pub g: f32,
	pub b: f32,
}

impl Add for Vec3 {
	type Output = Self;
	fn add(self, rhs: Self) -> Self::Output {
		Self { r: self.r + rhs.r, g: self.g + rhs.g, b: self.b + rhs.b }
	}
}

impl Sub for Vec3 {
	type Output = Self;
	fn sub(self, rhs: Self) -> Self::Output {
		Self { r: self.r - rhs.r, g: self.g - rhs.g, b: self.b - rhs.b }
	}
}

impl Mul<f32> for Vec3 {
	type Output = Self;
	fn mul(self, rhs: f32) -> Self::Output {
		Self { r: self.r * rhs, g: self.g * rhs, b: self.b * rhs }
	}
}

impl Mul<Vec3> for Vec3 {
	type Output = Self;
	fn mul(self, rhs: Vec3) -> Self::Output {
		Self { r: self.r * rhs.r, g: self.g * rhs.g, b: self.b * rhs.b }
	}
}


impl PartialEq for Vec3 {
	fn eq(&self, other: &Self) -> bool {
		f32::abs(self.r - other.r) < EPSILON &&
			f32::abs(self.g - other.g) < EPSILON &&
			f32::abs(self.b - other.b) < EPSILON
	}
}


pub fn color(r: f32, g: f32, b: f32) -> Vec3 {
	Vec3 { r, g, b }
}


#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_tuple_color() {
		let a = Vec3 { r: -0.5, g: 0.4, b: 1.7 };
		assert_eq!(a.r, -0.5);
		assert_eq!(a.g, 0.4);
		assert_eq!(a.b, 1.7);
	}

	#[test]
	fn test_new_color() {
		let c = color(4., -4., 3.);
		assert_eq!(c, Vec3 { r: 4., g: -4., b: 3. })
	}

	#[test]
	fn test_adding_colors() {
		let c1 = Vec3 { r: 0.9, g: 0.6, b: 0.75 };
		let c2 = Vec3 { r: 0.7, g: 0.1, b: 0.25 };
		assert_eq!(c1 + c2, Vec3 { r: 1.6, g: 0.7, b: 1. })
	}

	#[test]
	fn test_subtracting_two_colors() {
		let c1 = Vec3 { r: 0.9, g: 0.6, b: 0.75 };
		let c2 = Vec3 { r: 0.7, g: 0.1, b: 0.25 };
		assert_eq!(c1 - c2, Vec3 { r: 0.2, g: 0.5, b: 0.5 })
	}

	#[test]
	fn test_multiplying_a_color_by_a_scalar() {
		let c = color(0.2, 0.3, 0.4);
		assert_eq!(c * 2., color(0.4, 0.6, 0.8))
	}

	#[test]
	fn test_multiplying_a_color_by_a_color() {
		let c1 = color(1., 0.2, 0.4);
		let c2 = color(0.9, 1., 0.1);
		assert_eq!(c1 * c2, color(0.9, 0.2, 0.04))
	}
}
