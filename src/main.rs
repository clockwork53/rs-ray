use crate::canvas::Canvas;
use crate::vec3::color;
use crate::vec4::{EPSILON, point, vector};
use self::vec4::Vec4;

mod vec4;
mod vec3;
mod canvas;

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
	let mut p = Projectile { position: point(0f32, 1f32, 0f32), velocity: vector(1f32, 1f32, 0f32).normalize() };
	let e = Environment { gravity: vector(0f32, -0.1, 0f32), wind: vector(-0.01, 0f32, 0f32) };

	let mut canvas = Canvas::new(900, 500);

	let red_color = color(1., 0., 0.);
	canvas.write_pixel(
		(p.position.x as u64).clamp(0, 900),
		(canvas.get_height() - (p.position.y as u64)).clamp(0, 500),
		red_color
	);

	let mut ticks = 0;
	while &p.position.y - 0f32 > EPSILON {
		ticks += 1;
		p = tick(&e, &p);
		canvas.write_pixel(
			(p.position.x as u64).clamp(0, 899),
			(canvas.get_height() - (p.position.y as u64)).clamp(0, 499),
			red_color
		);
		// dbg!(&p);
	}
	dbg!(&p);
	dbg!(ticks);
	canvas.save_to_file("/home/clockwork/Projects/Practice/rs-ray/pic.ppm".to_string());
}