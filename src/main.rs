use dashmap::{DashMap, DashSet};
use piston_window::*;
use rand::{prelude::*, distributions::Uniform};
use rayon::prelude::*;
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

const W_SIZE: usize = 720;

#[derive(Debug)]
struct Being {
    id: u32,

    pos: (f64, f64),
    rotation: f64,

    health: f64,
    hunger: f64,
}

#[derive(Debug)]
struct Food {
    id: u32,
    pos: (f64, f64),
    val: f64,
}

#[derive(Debug)]
struct Chunk {
    pos: (u32, u32),
    being_keys: Vec<u32>,
    food_keys: Vec<u32>,
}

#[derive(Debug)]
struct World {
    chunk_size: f64,
    n_chunks: u32,
    worldsize: f64,

    chunks: Vec<Vec<Arc<Mutex<Chunk>>>>,

    beings: DashMap<u32, Being>,
    foods: DashMap<u32, Food>,

    being_speed: f64,
    being_radius: f64,

    beingkey: u32,
    foodkey: u32,

    repr: Vec<u32>,
}

fn normalize_2d((i, j): (f64, f64)) -> (f64, f64) {
    let norm = (i.powi(2) + j.powi(2)).sqrt();

    (i / norm, j / norm)
}

fn add_2d((i, j): (f64, f64), (k, l): (f64, f64)) -> (f64, f64) {
    (i + k, j + l)
}

fn scale_2d((i, j): (f64, f64), c: f64) -> (f64, f64) {
    (i * c, j * c)
}

fn dist_2d((i1, j1): (f64, f64), (i2, j2): (f64, f64)) -> f64 {
    ((i1 - i2).powi(2) + (j1 - j2).powi(2)).sqrt()
}

fn one_to_two(ij: usize) -> (usize, usize) {
    ((ij - ij % W_SIZE) / W_SIZE, ij % W_SIZE)
}

fn two_to_one((i, j): (usize, usize)) -> usize {
    i * W_SIZE + j
}

fn dir_from_theta(theta: f64) -> (f64, f64) {
    (theta.cos(), theta.sin())
}

impl World {
    pub fn new(chunk_size: f64, n_chunks: u32) -> Self {
        World {
            chunk_size: chunk_size,
            n_chunks: n_chunks,
            worldsize: chunk_size * (n_chunks as f64),

            chunks: (0..n_chunks)
                .into_par_iter()
                .map(|i| {
                    (0..n_chunks)
                        .into_iter()
                        .map(|j| {
                            Arc::new(Mutex::new(Chunk {
                                pos: (i * (chunk_size as u32), j * (chunk_size as u32)),
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

    fn pos_to_chunk(&self, pos: (f64, f64)) -> (usize, usize) {
        let i = ((pos.0 - (pos.0 % self.chunk_size)) / self.chunk_size) as usize;
        let j = ((pos.1 - (pos.1 % self.chunk_size)) / self.chunk_size) as usize;

        (i, j)
    }

    pub fn add_food(&mut self, pos: (f64, f64), val: f64, age: f64) {
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

    pub fn add_being(&mut self, pos: (f64, f64), rotation: f64, health: f64) {
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
        let mut rdist = Uniform::new(-1., 1.);

        self.beings.par_iter_mut().for_each_init(thread_rng,|rng, mut entry| {
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
                being.rotation = being.rotation * -1. + rng.sample(rdist);
            }
        });
    }

    pub fn check_being_collision(&mut self) {
        self.beings.iter_mut().for_each(|being| {
            let (key, being) = (being.key(), being.value());

            let (bci, bcj) = self.pos_to_chunk(being.pos);
            // println!("{:?} {:?}, {}", being.pos, (bci, bcj), self.chunk_size);
            [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)].into_par_iter().for_each(|(di, dj)|{
                let (ni, nj) = (bci as isize + di, bcj as isize + dj);
                if ni.min(nj) >= 0 && ni.max(nj) < self.n_chunks as isize{

                }
            });
        })
    }

    // fn pacwatch(&self, (pi, pj): (f64, f64), rad: f64) -> Vec<Vec<u32>> {
    //     let (pi, pj) = (pi as u32, pj as u32);

    // }
}

fn main() {
    let mut world = World::new(45., 16);
    world.add_being((50., 50.), 1.57, 10.);
    // world.add_being((100., 50.), 1.57, 10.);
    // world.add_being((150., 50.), 1.57, 10.);
    // world.add_being((200., 50.), 1.57, 10.);
    // world.add_being((250., 50.), 1.57, 10.);
    // world.add_being((300., 50.), 1.57, 10.);
    // world.add_being((350., 50.), 1.57, 10.);
    // world.add_being((400., 50.), 1.57, 10.);
    // world.add_being((450., 50.), 1.57, 10.);
    // world.add_being((50., 670.), -1.57, 10.);
    // world.add_being((100., 670.), -1.57, 10.);
    // world.add_being((150., 670.), -1.57, 10.);
    // world.add_being((200., 670.), -1.57, 10.);
    // world.add_being((250., 670.), -1.57, 10.);
    // world.add_being((300., 670.), -1.57, 10.);
    // world.add_being((350., 670.), -1.57, 10.);
    // world.add_being((400., 670.), -1.57, 10.);
    // world.add_being((450., 670.), -1.57, 10.);

    let mut window: PistonWindow =
        WindowSettings::new("Hello Piston!", [W_SIZE as u32, W_SIZE as u32])
            .exit_on_esc(true)
            .build()
            .unwrap();
    let mut i = 0;
    loop {
        if i % 100 == 0 {
            {
                if let Some(e) = window.next() {
                    world.beings.iter().for_each(|b| {
                        window.draw_2d(&e, |c, g, device| {
                            // clear([1.0; 4], g);
                            rectangle([1., 0., 0., 1.], [b.pos.1, b.pos.0, 5., 5.], c.transform, g);
                        });
                    });
                }
            }
        }
        i += 1;
        println!("{}", i);
        world.move_beings();
        world.check_being_collision();
    }
}
