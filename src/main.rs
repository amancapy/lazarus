use rand::{distributions::Uniform, prelude::*};
use rayon::{prelude::*, iter::plumbing};
use std::{
    thread,
    borrow::BorrowMut,
    sync::{Arc, Mutex, RwLock},
    time::*, collections::HashSet,
};


const W_SIZE: usize = 1000;
const N_CELLS: usize = 25;
const CHUNK_SIZE: usize = W_SIZE / N_CELLS;


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
    balls: Vec<Ball>,
    cells: Vec<Vec<HashSet<usize>>>,
    cell_size: usize,
    ball_id: usize
}

impl Chunk {
    pub fn new(n_cells: usize) -> Self {
        assert!(W_SIZE % n_cells == 0);
        Chunk { balls: 
            vec![],
            cells: (0..n_cells).into_iter().map(|_| {
                (0..n_cells).into_iter().map(|_| {
                    HashSet::new()
                }).collect()
            }).collect(),
            cell_size: W_SIZE / n_cells,
            ball_id: 0
         }
    }

    pub fn add_ball(&mut self, radius: f64, pos: (f64, f64), rotation: f64, speed: f64) {
        let (i, j) = pos_to_chunk(pos);
        self.balls.push(Ball {radius: radius, pos: pos, rotation: rotation, speed: speed, chunk: (i, j)});
        self.cells[i][j].insert(self.ball_id);
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

    pub fn check_collisions(&mut self) {
        for i in 1..N_CELLS - 1 {
            for j in 1..N_CELLS - 1 {
                for id1 in &self.cells[i][j] {
                    let b1 = &self.balls[*id1];
                    
                    for (di, dj) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)] {
                        let (ni, nj) = ((i as isize + di) as usize, (j as isize + dj) as usize);
                        for id2 in &self.cells[ni][nj] {
                            if *id1 != *id2 {
                                let b2 = &self.balls[*id2];
                            }
                        }
                    }
                }
            }
        }
    }
}


fn main() {
    let n_chunks = 25;
    let mut chunk = Chunk::new(n_chunks);

    for i in 1..25000{chunk.add_ball(5., (3., 3.), 0.7853981633974483, 5.);}


    for i in 1..10000000_u64 {
        if i % 1000 == 0 {println!("{}", i)}
        chunk.move_balls();
        chunk.check_collisions();
    }

    println!("{:?}, {:?}", chunk.balls[0].pos, chunk.balls[0].chunk);
}