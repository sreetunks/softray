use softbuffer::GraphicsContext;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
	dpi::LogicalSize};
use std::{
	ops,
	collections::HashMap,
	vec::Vec
};

#[derive(Copy, Clone)]
struct Vec3 {
	x : f32,
	y : f32,
	z : f32}

impl Vec3 {
	fn create(x : f32, y : f32, z : f32) -> Self { Vec3 {x, y, z} }
	fn dot(self, vec : Self) -> f32 { (self.x * vec.x) + (self.y * vec.y) + (self.z * vec.z) }
	fn cross(self, vec : Self) -> Self {
		Vec3 {
			x : (self.y * vec.z - self.z * vec.y),
			y : (self.z * vec.x - self.x * vec.z),
			z : (self.x * vec.y - self.y * vec.x)
		}
	}
	fn square_magnitude(self) -> f32 { (self.x * self.x) + (self.y * self.y) + (self.z * self.z) }
	fn magnitude(self) -> f32 { self.square_magnitude().sqrt() }
	fn normalize(self) -> Self {
		let inv_len = self.magnitude().recip();
		self * inv_len
	}}

impl ops::Add<Vec3> for Vec3 {
	type Output = Vec3;

	fn add(self, _rhs : Vec3) -> Vec3 { Vec3 {x : self.x + _rhs.x, y : self.y + _rhs.y, z : self.z + _rhs.z} }}

impl ops::Sub<Vec3> for Vec3 {
	type Output = Self;

	fn sub(self, _rhs : Self) -> Self { Self {x : self.x - _rhs.x, y : self.y - _rhs.y, z : self.z - _rhs.z} }}

impl ops::Mul<f32> for Vec3 {
	type Output = Self;

	fn mul(self, _rhs : f32) -> Self { Self {x : self.x * _rhs, y : self.y * _rhs, z : self.z * _rhs}}}

impl ops::Mul<Vec3> for Vec3 {
	type Output = Self;

	fn mul(self, _rhs : Vec3) -> Self { Self {x : self.x * _rhs.x, y : self.y * _rhs.y, z : self.z * _rhs.z } }}

struct Camera {
	pos : Vec3,
	lens_plane : f32,
	size : f32,
	x : Vec3,
	y : Vec3,
	z : Vec3
}
impl Camera {
	fn create(pos : Vec3, size : f32, lens_plane : f32) -> Camera {
		let z = (pos * -1.0).normalize();
		let x = Vec3::create(0.0, -1.0, 0.0).cross(z).normalize();
		let y = z.cross(x).normalize();
		Camera { pos, lens_plane, size, x, y, z }
	}

	fn look_at(&mut self, target : Vec3) {
		self.z = (target - self.pos).normalize();
		self.x = Vec3::create(0.0, -1.0, 0.0).cross(self.z).normalize();
		self.y = self.z.cross(self.x).normalize();
	}

	fn get_world_pos(&self, screen_x : f32, screen_y : f32) -> Vec3 {
		(self.x * screen_x * self.size) +
		(self.y * screen_y * self.size) +
		self.pos + (self.z * self.lens_plane)
	}
}

struct Material {
	specular : f32,
	reflect_color : Vec3,
	emit_color : Vec3
}

#[derive(Copy, Clone)]
struct RayCastHit {
	position	: Vec3,
	normal		: Vec3,
	distance	: f32
}

struct Plane {
	normal : Vec3,
	d : f32,
	material_id : u64
}

impl Plane {
	fn create(normal : Vec3, d : f32, material_id : u64) -> Plane { Plane { normal, d, material_id } }
	fn distance(&self, pos : Vec3) -> f32 { self.normal.dot(pos) + self.d }
	fn has_contact(&self, dir : Vec3) -> bool { self.normal.dot(dir * -1.0) > 0.0 }
	fn get_contact(&self, pos : Vec3, dir : Vec3) -> RayCastHit {
		let distance = self.distance(pos) / self.normal.dot(dir * -1.0);
		let position = pos + (dir * distance);
		let normal = self.normal;
		RayCastHit {position, normal, distance }
	}
}

struct Sphere {
	pos : Vec3,
	radius : f32,
	material_id : u64
}

impl Sphere {
	fn create(pos : Vec3, radius : f32, material_id : u64) -> Sphere { Sphere { pos, radius, material_id } }
	fn has_contact(&self, pos : Vec3, dir : Vec3) -> bool {
		let c2o = self.pos - pos;
		let r_2 = self.radius * self.radius;
		let c2o_sqlen = c2o.square_magnitude();
		if c2o_sqlen < r_2 { return true }
		let c2o_proj = c2o.dot(dir);
		if c2o_proj < 0.0 { return false }
		!(c2o_sqlen - (c2o_proj * c2o_proj) > r_2)
	}
	fn get_contact(&self, pos : Vec3, dir : Vec3) -> RayCastHit {
		let r_2 = self.radius * self.radius;
		let c2o = self.pos - pos;
		let c2o_proj = c2o.dot(dir);
		let c2o_sqlen = c2o.square_magnitude();
		let q  = (r_2 + (c2o_proj * c2o_proj) - c2o_sqlen).sqrt();
		let distance : f32;
		if c2o_sqlen < r_2 { distance = c2o_proj + q }
		else { distance = c2o_proj - q }
		let position = pos + (dir * distance);
		let normal = (position - self.pos).normalize();
		RayCastHit {position, normal, distance }
	}
}

struct World {
	material_map : HashMap<u64, Material>,
	planes : Vec<Plane>,
	spheres : Vec<Sphere>}

impl World {
	fn new() -> World {
		World { material_map: HashMap::new(), planes : Vec::new(), spheres : Vec::new() }
	}

	fn add_material(&mut self, reflect_color : Vec3, emit_color : Vec3, specular : f32) -> u64 {
		let id = self.material_map.len().try_into().unwrap();
		self.material_map.insert(id, Material { specular, reflect_color, emit_color});
		id
	}

	fn add_plane(&mut self, normal : Vec3, d : f32, material_id : u64) {
		assert!(self.material_map.contains_key(&material_id));
		self.planes.push(Plane::create(normal, d, material_id));
	}

	fn add_sphere(&mut self, pos : Vec3, radius : f32, material_id : u64) {
		assert!(self.material_map.contains_key(&material_id));
		self.spheres.push(Sphere::create(pos, radius, material_id));
	}}

fn cast_ray(world : &World, orig: Vec3, dir : Vec3) -> Vec3 {
	let mut orig = orig;
	let mut dir = dir;

	let mut num_bounces = 12;
	let mut color = Vec3::create(0.0, 0.0, 0.0);
	let mut atten = Vec3::create(1.0, 1.0, 1.0);
	while num_bounces > 0 { 
		let mut hit = RayCastHit {
			position: Vec3::create(0.0, 0.0, 0.0),
			normal	: Vec3::create(0.0, 0.0, 0.0),
			distance: f32::INFINITY};
		let mut hit_material = world.material_map.get(&0).unwrap();

		for plane in world.planes.iter() {
			if plane.has_contact(dir) {
				let contact = plane.get_contact(orig, dir);
				if contact.distance < hit.distance {
					hit = contact;
					hit_material = world.material_map.get(&plane.material_id).unwrap();
				}
			}
		}

		for sphere in &world.spheres {
			if sphere.has_contact(orig, dir) {
				let contact = sphere.get_contact(orig, dir);
				if contact.distance < hit.distance {
					hit = contact;
					hit_material = world.material_map.get(&sphere.material_id).unwrap();
				}
			}
		}

		color = color + (atten * hit_material.emit_color);
		if hit.distance == f32::INFINITY { break; }

		atten = atten * hit_material.reflect_color;
		atten = atten * ((dir * -1.0).dot(hit.normal)).max(0.0);
		orig = hit.position;
		dir = (dir - (hit.normal * 2.0 * dir.dot(hit.normal))).normalize();
		let rand_dir = hit.normal + Vec3::create(-1.0 + 2.0 * rand::random::<f32>(), -1.0 + 2.0 * rand::random::<f32>(), -1.0 + 2.0 * rand::random::<f32>());
		dir = dir * hit_material.specular + rand_dir.normalize() * (1.0 - hit_material.specular);
		num_bounces = num_bounces - 1;
	}

	color
}

fn main() {
    let event_loop = EventLoop::new();
	let builder = WindowBuilder::new();
    let window = builder.build(&event_loop).unwrap();
	window.set_inner_size(LogicalSize::new(1280.0f32, 720.0f32));
    let mut graphics_context = unsafe { GraphicsContext::new(window) }.unwrap();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::RedrawRequested(window_id) if window_id == graphics_context.window().id() => {
                let (width, height) = {
                    let size = graphics_context.window().inner_size();
                    (size.width, size.height)
                };

				let aspect_ratio = width as f32 / height as f32;
				let mut camera = Camera::create(Vec3::create(0.0, 6.0, 6.0), 5.0, 1.0);
				camera.look_at(Vec3::create(0.0, 0.0, 0.0));
				let mut world = World::new();

				world.add_material(Vec3::create(0.0, 0.0, 0.0), Vec3::create(0.392, 0.584, 0.929), 0.0);

				let sun		= world.add_material(Vec3::create(0.0, 0.0, 0.0), Vec3::create(1.0, 1.0, 1.0), 0.0);
				let mat1	= world.add_material(Vec3::create(0.5, 0.5, 0.5), Vec3::create(0.0, 0.0, 0.0), 0.2);
				let mat2	= world.add_material(Vec3::create(0.5, 0.7, 0.3), Vec3::create(0.0, 0.0, 0.0), 0.95);
				let mat3	= world.add_material(Vec3::create(0.95, 0.95, 0.95), Vec3::create(0.0, 0.0, 0.0), 1.0);
				let mat4	= world.add_material(Vec3::create(0.8, 0.8, 0.8), Vec3::create(0.6, 0.0, 0.0), 0.6);

				world.add_plane(Vec3::create(0.0, 1.0, 0.0), 0.0, mat1);

				world.add_sphere(Vec3::create(0.0, 10.0, 0.0), 1.0, sun);

				world.add_sphere(Vec3::create(0.0, 1.0, 0.0), 1.0, mat2);
				world.add_sphere(Vec3::create(2.0, 0.0, 0.0), 1.0, mat3);
				world.add_sphere(Vec3::create(-2.0, 1.0, 2.0), 1.0, mat4);

				let buffer = (0..((width * height) as usize))
					.map(|index| {
						let pixel_y = index as u32 / width;
						let pixel_x = index as u32 % width;

						let mut color = Vec3::create(0.0, 0.0, 0.0);

						//if pixel_x != 640 || pixel_y != 360 { return 0 }

						const RAYS_PER_PIXEL : u32 = 64;
						for _ in 0..RAYS_PER_PIXEL {
							let y = -1.0 + 2.0 *((pixel_y as f32) / (height as f32));
							let x = -1.0 + 2.0 * ((pixel_x as f32) / (width as f32));

							let offset = (-1.0 + 2.0 * rand::random::<f32>(), -1.0 + 2.0 * rand::random::<f32>());
							let offset = (offset.0 * 0.5 / width as f32, offset.1 * 0.5 / height as f32);

							let film_point = camera.get_world_pos((x * 0.5 * aspect_ratio) + offset.0, (y * 0.5) + offset.1);

							//color = color + (cast_ray(&world, camera.pos, (film_point - camera.pos).normalize()) * (1.0 / RAYS_PER_PIXEL as f32));
							color = color + (cast_ray(&world, film_point, camera.z) * (1.0 / RAYS_PER_PIXEL as f32));
						}

						if color.x <= 0.0031308 { color.x = color.x * 12.92 }
						else { color.x = (1.055 * color.x.powf(1.0/2.4)) - 0.055}
						if color.y <= 0.0031308 { color.y = color.y * 12.92 }
						else { color.y = (1.055 * color.y.powf(1.0/2.4)) - 0.055}
						if color.z <= 0.0031308 { color.z = color.z * 12.92 }
						else { color.z = (1.055 * color.z.powf(1.0/2.4)) - 0.055}

						(color.z * 255.0) as u32 |
						((color.y * 255.0) as u32) << 8 |
						((color.x * 255.0) as u32) << 16
                    })
                    .collect::<Vec<_>>();

                graphics_context.set_buffer(&buffer, width as u16, height as u16);
            }

            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == graphics_context.window().id() => *control_flow = ControlFlow::Exit,
            _ => (),
        }
    });
}
