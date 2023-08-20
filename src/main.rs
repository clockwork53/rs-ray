use crate::vec4::{EPSILON, point, vector};
use crate::vec3::{color};
use self::vec4::Vec4;
use self::vec3::Vec3;

mod vec4;
mod vec3;

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

	let mut ticks = 0;
	while &p.position.y - 0f32 > EPSILON {
		ticks += 1;
		p = tick(&e, &p);
	}
	dbg!(&p);
	dbg!(ticks);
}