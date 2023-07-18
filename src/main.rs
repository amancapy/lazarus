use piston_window::*;
use rand::{distributions::Uniform, prelude::*};
use rayon::prelude::*;
use std::{
    thread,
    borrow::BorrowMut,
    sync::{Arc, Mutex, RwLock},
    time::*, collections::HashSet,
};


const W_SIZE: usize = 1000;


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


struct Ball {
    radius: f64,
    pos: (f64, f64),
    rotation: f64,
    speed: f64
}


struct Chunk {
    ball_indexes: HashSet<u32>
}

impl Chunk {
    pub fn new() -> Self {
        Chunk { ball_indexes: HashSet::new() }
    }
}


struct World {
    balls: Vec<Ball>,
    chunks: Vec<Vec<Chunk>>,
    chunk_size: usize
}

impl World {
    pub fn new(n_chunks: usize) -> Self {
        assert!(W_SIZE % n_chunks == 0);
        World { balls: 
            vec![],
            chunks: (0..n_chunks).into_par_iter().map(|_| {
                (0..n_chunks).into_par_iter().map(|_| {
                    Chunk::new()
                }).collect()
            }).collect(),
            chunk_size: W_SIZE / n_chunks
         }
    }

    pub fn add_ball(&mut self, radius: f64, pos: (f64, f64), rotation: f64, speed: f64) {
        self.balls.push(Ball {radius: radius, pos: pos, rotation: rotation, speed: speed })
    }

    pub fn pos_to_chunk(&self, pos: (f64, f64)) -> (usize, usize) {
        let c = self.chunk_size as f64;
        let i = ((pos.0 - (pos.0 % c)) / c) as usize;
        let j = ((pos.1 - (pos.1 % c)) / c) as usize;

        (i, j)
    }
}


fn main() {
    let n_chunks = 25;
    let a = World::new(n_chunks);
}