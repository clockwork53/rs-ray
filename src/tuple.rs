#[allow(dead_code)]
pub enum TupleFill<T> {
	Array([T; 4]),
	Single(T),
}

#[derive(Debug, PartialEq)]
pub struct Tuple<T> {
	pub(crate) data: [T; 4],
}

impl<T: Default + Copy> Tuple<T> {
	pub fn new(values: Option<TupleFill<T>>) -> Tuple<T> {
		let mut data = [T::default(); 4];

		if let Some(fill) = values {
			match fill {
				TupleFill::Array(values) => {
					for (i, row) in values.iter().enumerate() {
						data[i] = *row;
					}
				}
				TupleFill::Single(value) => {
					for elem in data.iter_mut() {
						*elem = value;
					}
				}
			}
		}

		Tuple { data }
	}

	pub fn set(&mut self, row: usize, value: T) {
		self.data[row] = value;
	}
	pub fn get(&self, row: usize) -> T {
		self.data[row]
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_tuple_equality() {
		let tuple1 = Tuple {
			data: [1, 2, 3, 4],
		};
		let tuple1f = Tuple {
			data: [1., 2., 3., 3678678678.555555273786767896789676796796796796795555555555555555555555555],
		};

		let tuple2 = Tuple {
			data: [1, 2, 3, 4],
		};
		let tuple2f = Tuple {
			data: [1., 2., 3., 3678678678.555555273786767896789676796796796796795555555555555555555555555],
		};

		let tuple3 = Tuple {
			data: [1, 2, 3, 5],
		};
		let tuple3f = Tuple {
			data: [1., 2., 3., 3678678678.555555273786767896789676796796796796795555555555555555555555551],
		};

		assert_eq!(tuple1, tuple2);
		assert_eq!(tuple1f, tuple2f);
		assert_ne!(tuple1, tuple3);
		assert_ne!(tuple1f, tuple3f);
	}

	#[test]
	fn test_new_tuple() {
		let fill_array: [f32; 4] = [1., 2., 3., 4.];

		let matrix4x1_filled = Tuple::new(Some(TupleFill::Array(fill_array)));
		assert_eq!(matrix4x1_filled.data, fill_array);
	}

	#[test]
	fn test_set_function() {
		let mut matrix4x1 = Tuple::new(None);

		matrix4x1.set(0, 1.0);
		matrix4x1.set(1, 2.0);
		matrix4x1.set(2, 3.0);
		matrix4x1.set(3, 4.0);

		assert_eq!(matrix4x1.data, [0., 1., 2., 3.]);
	}
}
