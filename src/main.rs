use crate::canvas::Canvas;
use crate::vec3::color;
use crate::vec4::{EPSILON, point, vector};
use self::vec4::Vec4;

mod vec4;
mod vec3;
mod canvas;
mod matrix;
mod tuple;

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

fn main() {
	let mut p = Projectile {
		position: point(0., 1., 0.),
		velocity: vector(1., 1.8, 0.).normalize() * 11.25,
	};
	let e = Environment {
		gravity: vector(0f32, -0.1, 0f32),
		wind: vector(-0.01, 0f32, 0f32),
	};

	let mut canvas = Canvas::new(900, 550);

	let red_color = color(1., 0., 0.);
	canvas.write_pixel(
		p.position.x as u64,
		(canvas.get_height() as f32 - p.position.y) as i64 as u64,
		red_color,
	);

	let mut ticks = 0;
	while &p.position.y - 0. > EPSILON {
		ticks += 1;
		p = tick(&e, &p);
		canvas.write_pixel(
			p.position.x as u64,
			(canvas.get_height() as f32 - p.position.y) as i64 as u64,
			red_color,
		);
		// dbg!(&p);
	}
	// dbg!(&p);
	dbg!(ticks);
	canvas.save_to_file("/home/clockwork/Projects/Practice/rs-ray/pic.ppm".to_string());
}