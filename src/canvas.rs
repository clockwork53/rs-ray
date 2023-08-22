use std::fs::OpenOptions;
use std::io::Write;
use crate::vec3::{color, Vec3};

pub struct Canvas {
	width: u64,
	height: u64,
	grid: Vec<Vec3>,
}

impl Canvas {
	pub fn new(width: u64, height: u64) -> Self {
		Canvas {
			width,
			height,
			grid: vec![color(0., 0., 0.); (width * height) as usize],
		}
	}

	#[allow(dead_code)]
	pub fn get_width(&self) -> u64 {
		self.width
	}

	pub fn get_height(&self) -> u64 {
		self.height
	}

	pub fn write_pixel(&mut self, x: u64, y: u64, color: Vec3) {
		if x > self.width - 1 || y > self.height - 1 {
			return;
		}
		let position = x + (y * self.width);
		self.grid[position as usize] = color;
	}

	#[allow(dead_code)]
	pub fn pixel_at(self, x: u64, y: u64) -> Option<Vec3> {
		if x > self.width - 1 || y > self.height - 1 {
			return None;
		}
		let position = x + (y * self.width);
		Some(self.grid[position as usize])
	}

	#[allow(dead_code)]
	pub fn canvas_to_ppm_new(self) -> String {
		const MAX_COLOR_VALUE: f32 = 255.;
		const LINE_LENGTH_LIMIT: usize = 69;
		const NEWLINE: &str = "\n";

		let mut ppm = format!("P3\n{} {}\n255\n", self.width, self.height);
		let mut line_length = 0;

		let pixel_data = self.grid.iter().flat_map(|pixel| vec![pixel.r, pixel.g, pixel.b]);

		for (i, color) in pixel_data.enumerate() {
			let component_value = (color * MAX_COLOR_VALUE).ceil().clamp(0.0, MAX_COLOR_VALUE);
			let component_string = format!("{}", component_value);

			if line_length + component_string.len() > LINE_LENGTH_LIMIT {
				ppm.push_str(NEWLINE);
				line_length = 0;
			} else if i > 0 && !((i + 1) % 3 == 1 && (i + 1) % (self.width as usize * 3) == 1) {
				ppm.push_str(" ");
				line_length += 1;
			}

			ppm.push_str(&component_string);
			line_length += component_string.len();

			if (i + 1) % 3 == 0 && (i + 1) % (self.width as usize * 3) == 0 {
				ppm.push_str(NEWLINE);
				line_length = 0;
			}
		}
		ppm
	}

	pub fn canvas_to_ppm(self) -> String {
		let mut header = format!("P3\n{} {}\n255\n", self.width, self.height);
		let mut data = String::new();
		let mut current_line_length = 0;
		for (pos, pixel) in self.grid.iter().enumerate() {

			// 1
			let pxr = (pixel.r * 255.).ceil().clamp(0., 255.).to_string();
			if current_line_length + pxr.chars().count() > 69 {
				data.push_str("\n");
				current_line_length = 0;
			}
			data.push_str(format!("{}", pxr).as_str());
			current_line_length += pxr.chars().count() + 1;

			// 2
			let pxg = (pixel.g * 255.).ceil().clamp(0., 255.).to_string();
			if current_line_length + pxg.chars().count() > 69 {
				data.push_str("\n");
				current_line_length = 0;
			} else {
				data.push_str(" ");
			}
			data.push_str(format!("{}", pxg).as_str());
			current_line_length += pxg.chars().count() + 1;

			// 3
			let pxb = (pixel.b * 255.).ceil().clamp(0., 255.).to_string();
			if current_line_length + pxb.chars().count() > 69 {
				data.push_str("\n");
				current_line_length = 0;
			} else {
				data.push_str(" ");
			}
			data.push_str(format!("{}", pxb).as_str());
			current_line_length += pxb.chars().count() + 1;

			if (pos + 1) as u64 % self.width == 0 || pos == self.grid.len() - 1 {
				data.push_str("\n");
				current_line_length = 0;
			} else {
				data.push_str(" ");
			}
		}
		header.push_str(data.as_str());
		header
	}

	pub fn save_to_file(self, address: String) {
		let data = self.canvas_to_ppm();
		let mut f = OpenOptions::new()
			.write(true)
			.truncate(true)
			.create(true)
			.open(address)
			.expect("Unable to open file");
		f.write_all(data.as_bytes()).expect("Unable to write data");
	}
}


#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn creating_a_canvas() {
		let c = Canvas::new(10, 20);
		assert_eq!(c.width, 10);
		assert_eq!(c.height, 20);
		for pixel in c.grid {
			assert_eq!(pixel, color(0., 0., 0.));
		}
	}

	#[test]
	fn writing_a_pixel_to_canvas() {
		let mut c = Canvas::new(10, 20);
		let red = color(1., 0., 0.);
		c.write_pixel(2, 3, red);
		assert_eq!(c.pixel_at(2, 3).unwrap(), red);
	}

	#[test]
	fn constructing_ppm_file_from_canvas() {
		let mut c = Canvas::new(5, 3);
		let c1 = color(1.5, 0., 0.);
		let c2 = color(0., 0.5, 0.);
		let c3 = color(-0.5, 0., 1.);
		c.write_pixel(0, 0, c1);
		c.write_pixel(2, 1, c2);
		c.write_pixel(4, 2, c3);
		assert_eq!(c.canvas_to_ppm(),
		           "P3\n5 3\n255\n255 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 128 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0 0 0 0 0 255\n"
		)
	}

	#[test]
	fn constructing_ppm_file_from_canvas_2() {
		let mut c = Canvas::new(10, 2);
		for i in 0..10 {
			for j in 0..2 {
				c.write_pixel(i, j, color(1., 0.8, 0.6));
			}
		}
		assert_eq!(c.canvas_to_ppm(),
		           "P3\n10 2\n255\n255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204\n153 255 204 153 255 204 153 255 204 153 255 204 153\n255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204\n153 255 204 153 255 204 153 255 204 153 255 204 153\n"
		)
	}
}