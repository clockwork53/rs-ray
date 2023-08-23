use crate::canvas::Canvas;
use crate::matrix::IDENTITY_MATRIX;
use crate::misc::{EPSILON, Float, PI};
use crate::vec3::color;
use crate::vec4::{point, vector};
use self::vec4::Vec4;

mod vec4;
mod vec3;
mod canvas;
mod matrix;
mod tuple;
mod transform;
mod misc;


fn main() {
	draw_projectile();
	draw_clock();
}

fn draw_projectile() {
	#[derive(Debug)]
	struct Projectile {
		position: Vec4,
		velocity: Vec4,
	}

	struct Environment {
		gravity: Vec4,
		wind: Vec4,
	}

	fn tick(env: &Environment, proj: &Projectile) -> Projectile {
		Projectile { position: proj.position + proj.velocity, velocity: proj.velocity + env.gravity + env.wind }
	}

	let mut p = Projectile {
		position: point(0., 1., 0.),
		velocity: vector(1., 1.8, 0.).normalize() * 11.25,
	};
	let e = Environment {
		gravity: vector(0., -0.1, 0.),
		wind: vector(-0.01, 0., 0.),
	};

	let mut canvas = Canvas::new(900, 550);

	let red_color = color(1., 0., 0.);
	canvas.write_pixel(
		p.position.x as u64,
		(canvas.get_height() as Float - p.position.y) as i64 as u64,
		red_color,
	);

	while &p.position.y - 0. > EPSILON {
		p = tick(&e, &p);
		canvas.write_pixel(
			p.position.x as u64,
			(canvas.get_height() as Float - p.position.y) as i64 as u64,
			red_color,
		);
	}
	canvas.save_to_file("/home/clockwork/Projects/Practice/rs-ray/projectile.ppm".to_string());
}

fn draw_clock() {
	let mut canvas = Canvas::new(500, 500);
	let white_color = color(1., 0., 0.);

	let twelve = point(0., 1., 0.);
	for hour in 1..12 {
		let pos = IDENTITY_MATRIX.rotate_z(-hour as Float * PI / 6.) * twelve;
		let x = (pos.x * 200.) + 250.;
		let y = (-pos.y * 200.) + 250.;
		canvas.write_pixel(
			x as i64 as u64,
			y as i64 as u64,
			white_color,
		);
	}

	canvas.save_to_file("/home/clockwork/Projects/Practice/rs-ray/clock.ppm".to_string());
}