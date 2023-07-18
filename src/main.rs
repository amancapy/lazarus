use rand::{distributions::Uniform, prelude::*};
use rayon::{prelude::*, iter::plumbing};
use std::{
    thread,
    borrow::BorrowMut,
    sync::{Arc, Mutex, RwLock},
    time::*, collections::HashSet,
};


const W_SIZE: usize = 1000;
const N_CHUNKS: usize = 25;
const CHUNK_SIZE: usize = W_SIZE / N_CHUNKS;


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

fn equal_idx((a, b): (usize, usize), (c, d): (usize, usize)) -> bool {
    a == c && b == d
}

pub fn pos_to_chunk(pos: (f64, f64)) -> (usize, usize) {
    let c = CHUNK_SIZE as f64;
    let i = ((pos.0 - (pos.0 % c)) / c) as usize;
    let j = ((pos.1 - (pos.1 % c)) / c) as usize;

    (i, j)
}

struct Ball {
    radius: f64,
    pos: (f64, f64),
    rotation: f64,
    speed: f64,
    chunk: (usize, usize)
}


struct Chunk {
    ball_indexes: HashSet<usize>
}

impl Chunk {
    pub fn new() -> Self {
        Chunk { ball_indexes: HashSet::new() }
    }
}


struct World {
    balls: Vec<Ball>,
    chunks: Vec<Vec<Chunk>>,
    chunk_size: usize,
    ball_id: usize
}

impl World {
    pub fn new(n_chunks: usize) -> Self {
        assert!(W_SIZE % n_chunks == 0);
        World { balls: 
            vec![],
            chunks: (0..n_chunks).into_iter().map(|_| {
                (0..n_chunks).into_iter().map(|_| {
                    Chunk::new()
                }).collect()
            }).collect(),
            chunk_size: W_SIZE / n_chunks,
            ball_id: 0
         }
    }

    pub fn add_ball(&mut self, radius: f64, pos: (f64, f64), rotation: f64, speed: f64) {
        let (i, j) = pos_to_chunk(pos);
        self.balls.push(Ball {radius: radius, pos: pos, rotation: rotation, speed: speed, chunk: (i, j)});
        self.chunks[i][j].ball_indexes.insert(self.ball_id);
        self.ball_id += 1;
    }

    

    pub fn move_balls(&mut self) {
        self.balls.iter_mut().for_each(|ball| {
            let move_vec = scale_2d(dir_from_theta(ball.rotation), ball.speed);
            ball.pos = add_2d(ball.pos, move_vec);

            let new_chunk = pos_to_chunk(ball.pos);
            if !equal_idx(new_chunk, ball.chunk) {
                ball.chunk = new_chunk;
            }
        })
    }
}


fn main() {
    let n_chunks = 25;
    let mut world = World::new(n_chunks);

    world.add_ball(5., (3., 3.), 0.7853981633974483, 5.);


    for i in 1..10000000_u64 {
        world.move_balls();
    }

    println!("{:?}, {:?}", world.balls[0].pos, world.balls[0].chunk);
}