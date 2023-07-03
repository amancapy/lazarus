use dashmap::{DashMap, DashSet};
use minifb::{Key, Window, WindowOptions};
use rand::prelude::*;
use rayon::prelude::*;
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

const W_SIZE: usize = 720;

#[derive(Debug)]
struct Being {
    id: u32,

    pos: (f32, f32),
    rotation: f32,

    health: f32,
    hunger: f32,
}

#[derive(Debug)]
struct Food {
    id: u32,
    pos: (f32, f32),
    val: f32,
}

#[derive(Debug)]
struct Chunk {
    pos: (u32, u32),
    being_keys: Vec<u32>,
    food_keys: Vec<u32>,
}

#[derive(Debug)]
struct World {
    chunk_size: f32,
    n_chunks: u32,
    worldsize: f32,

    chunks: Vec<Vec<Arc<Mutex<Chunk>>>>,

    beings: DashMap<u32, Being>,
    foods: DashMap<u32, Food>,

    being_speed: f32,
    being_radius: f32,

    beingkey: u32,
    foodkey: u32,

    repr: Vec<u32>,
}

fn normalize_2d((i, j): (f32, f32)) -> (f32, f32) {
    let norm = (i.powi(2) + j.powi(2)).sqrt();

    (i / norm, j / norm)
}

fn add_2d((i, j): (f32, f32), (k, l): (f32, f32)) -> (f32, f32) {
    (i + k, j + l)
}

fn scale_2d((i, j): (f32, f32), c: f32) -> (f32, f32) {
    (i * c, j * c)
}

fn dist_2d((i1, j1): (f32, f32), (i2, j2): (f32, f32)) -> f32 {
    ((i1 - i2).powi(2) + (j1 - j2).powi(2)).sqrt()
}

fn one_to_two(ij: usize) -> (usize, usize) {
    ((ij - ij % W_SIZE) / W_SIZE, ij % W_SIZE)
}

fn two_to_one((i, j): (usize, usize)) -> usize {
    i * W_SIZE + j
}

fn dir_from_theta(theta: f32) -> (f32, f32) {
    (theta.cos(), theta.sin())
}

impl World {
    pub fn new(chunk_size: f32, n_chunks: u32) -> Self {
        World {
            chunk_size: chunk_size,
            n_chunks: n_chunks,
            worldsize: chunk_size * (n_chunks as f32),

            chunks: (0..n_chunks)
                .into_par_iter()
                .map(|i| {
                    (0..n_chunks)
                        .into_iter()
                        .map(|j| {
                            Arc::new(Mutex::new(Chunk {
                                pos: (i, j),
                                being_keys: vec![],
                                food_keys: vec![],
                            }))
                        })
                        .collect()
                })
                .collect(),
            beings: DashMap::new(),
            foods: DashMap::new(),

            being_speed: 1.,
            being_radius: chunk_size / 3.,

            beingkey: 0,
            foodkey: 0,

            repr: vec![],
        }
    }

    fn pos_to_chunk(&self, pos: (f32, f32)) -> (usize, usize) {
        let i = ((pos.0 - (pos.0 % self.chunk_size)) / self.chunk_size) as usize;
        let j = ((pos.1 - (pos.1 % self.chunk_size)) / self.chunk_size) as usize;

        (i, j)
    }

    pub fn add_food(&mut self, pos: (f32, f32), val: f32, age: f32) {
        self.foods.insert(
            self.foodkey,
            Food {
                id: self.foodkey,
                pos: pos,
                val: val,
            },
        );

        let (i, j) = self.pos_to_chunk(pos);
        self.chunks[i][j]
            .lock()
            .unwrap()
            .food_keys
            .push(self.foodkey);

        self.foodkey += 1;
    }

    pub fn add_being(&mut self, pos: (f32, f32), rotation: f32, health: f32) {
        self.beings.insert(
            self.beingkey,
            Being {
                id: self.beingkey,
                pos: pos,
                rotation: rotation,
                health: 10.,
                hunger: 0.,
            },
        );

        let (i, j) = self.pos_to_chunk(pos);
        self.chunks[i][j]
            .lock()
            .unwrap()
            .being_keys
            .push(self.beingkey);

        self.beingkey += 1;
    }

    pub fn decay_food(mut self) {
        self.foods.par_iter_mut().for_each(|mut entry| {
            entry.value_mut().val *= 0.9;
        });

        self.foods.retain(|_, food| food.val > 0.05);
    }

    pub fn move_beings(&mut self) {
        self.beings.par_iter_mut().for_each(|mut entry| {
            let being = entry.value_mut();
            let direction = (being.rotation.cos(), being.rotation.sin());
            let fatigue_speed = (10. - being.hunger) / 10. * self.being_speed;

            let curr_pos = being.pos.clone();
            let new_pos = add_2d(curr_pos, scale_2d(direction, fatigue_speed));

            if !(new_pos.0 - self.being_radius < 1.
                || new_pos.0 + self.being_radius >= self.worldsize - 1.
                || new_pos.1 - self.being_radius < 1.
                || new_pos.1 + self.being_radius >= self.worldsize - 1.)
            {
                being.pos = new_pos;
                let curr_chunk = self.pos_to_chunk(curr_pos);
                let new_chunk = self.pos_to_chunk(new_pos);

                if !(curr_chunk == new_chunk) {
                    self.chunks[curr_chunk.0][curr_chunk.1]
                        .lock()
                        .unwrap()
                        .being_keys
                        .retain(|x| x != &being.id);
                    self.chunks[new_chunk.0][new_chunk.1]
                        .lock()
                        .unwrap()
                        .being_keys
                        .push(being.id);
                }
            } else {
                let new_pos = add_2d(new_pos, scale_2d(direction, -fatigue_speed));
                being.pos = new_pos;
                being.rotation = being.rotation * -1. + 0.1;
            }
        });
    }

    pub fn check_being_collision(&mut self) {
        self.beings.iter_mut().for_each(|being| {
            let (key, being) = (being.key(), being.value());

            let (bci, bcj) = self.pos_to_chunk(being.pos);
            for (di, dj) in [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ] {
                let (a, b) = (bci as i32 + di, bcj as i32 + dj);

                if !(a < 0 || b < 0 || a >= self.n_chunks as i32 || b >= self.n_chunks as i32) {
                    let (a, b) = (a as usize, b as usize);
                    self.chunks[a][b]
                        .lock()
                        .unwrap()
                        .being_keys
                        .iter()
                        .for_each(|other_key| {
                            if !(*other_key == *key) {
                                let mut other = self.beings.get_mut(&other_key).unwrap();
                                let self_pos = being.pos;
                                let other_pos = other.value().pos;
                                if dist_2d(self_pos, other_pos) < 2. * self.being_radius {
                                    other.pos = add_2d(
                                        other_pos,
                                        scale_2d(dir_from_theta(other.rotation), 2.),
                                    );
                                }
                            }
                        });
                }
            }
        })
    }

    pub fn decide_being_pixels(&mut self) {
        let mut shared_buffer: Vec<Arc<Mutex<u32>>> = (0..W_SIZE.pow(2))
            .into_par_iter()
            .map(|ij| Arc::new(Mutex::new(0)))
            .collect();

        self.beings.par_iter_mut().for_each(|being| {
            let (bi, bj) = being.value().pos;

            (0.max((bi - self.being_radius.ceil()) as usize)
                ..self.worldsize.min(bi + self.being_radius.ceil()) as usize)
                .into_par_iter()
                .for_each(|i| {
                    (0.max((bj - self.being_radius.ceil()) as usize)
                        ..self.worldsize.min(bj + self.being_radius.ceil()) as usize)
                        .into_par_iter()
                        .for_each(|j| {
                            let (fi, fj) = (i as f32, j as f32);

                            if dist_2d((fi, fj), (bi, bj)) <= self.being_radius {
                                *shared_buffer[two_to_one((i, j))].lock().unwrap() = 100000;
                            }
                        })
                });
        });

        self.repr = shared_buffer
            .into_par_iter()
            .map(|i| *i.lock().unwrap())
            .collect();
    }

    // fn pacwatch(&self, (pi, pj): (f32, f32), rad: f32) -> Vec<Vec<u32>> {
    //     let (pi, pj) = (pi as u32, pj as u32);

    // }
}

fn main() {
    let mut world = World::new(11.25, 64);
    world.add_being((50., 50.), 1.57, 10.);
    world.add_being((100., 50.), 1.57, 10.);
    world.add_being((150., 50.), 1.57, 10.);
    world.add_being((200., 50.), 1.57, 10.);
    world.add_being((250., 50.), 1.57, 10.);
    world.add_being((300., 50.), 1.57, 10.);
    world.add_being((350., 50.), 1.57, 10.);
    world.add_being((400., 50.), 1.57, 10.);
    world.add_being((450., 50.), 1.57, 10.);
    world.add_being((50., 670.), -1.57, 10.);
    world.add_being((100., 670.), -1.57, 10.);
    world.add_being((150., 670.), -1.57, 10.);
    world.add_being((200., 670.), -1.57, 10.);
    world.add_being((250., 670.), -1.57, 10.);
    world.add_being((300., 670.), -1.57, 10.);
    world.add_being((350., 670.), -1.57, 10.);
    world.add_being((400., 670.), -1.57, 10.);
    world.add_being((450., 670.), -1.57, 10.);

    let mut window = Window::new(
        "samsarsa",
        W_SIZE,
        W_SIZE,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let mut i = 0;
    window.limit_update_rate(Some(std::time::Duration::from_micros(0)));
    while window.is_open() && !window.is_key_down(Key::Escape) {

        world.move_beings();
        world.check_being_collision();
        
        if i % 1000 == 0 {
            println!("{}", i);
            world.decide_being_pixels();
            std::thread::sleep(Duration::new(0, 1));
            window
                .update_with_buffer(&world.repr, W_SIZE, W_SIZE)
                .unwrap();
        }
        i += 1;

    }
}
