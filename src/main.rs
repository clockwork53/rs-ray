use crate::tuples::{EPSILON, point, vector};
use self::tuples::Tuple;

mod tuples;

#[derive(Debug)]
struct Projectile {
	position: Tuple,
	velocity: Tuple,
}

struct Environment {
	gravity: Tuple,
	wind: Tuple,
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
		dbg!(&p);
	}
	dbg!(ticks);
}