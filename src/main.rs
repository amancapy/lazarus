use rand::{distributions::Uniform, prelude::*};
use rayon::{iter::plumbing, prelude::*};
use splitmut::{SplitMut, SplitMutError};
use std::{
    borrow::BorrowMut,
    collections::HashSet,
    sync::{Arc, Mutex, RwLock},
    thread::{self, JoinHandle},
    time::*,
};

const W_SIZE: usize = 1000;
const N_CELLS: usize = 50;
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

fn same_index((a, b): (usize, usize), (c, d): (usize, usize)) -> bool {
    a == c && b == d
}

pub fn pos_to_chunk(pos: (f64, f64)) -> (usize, usize) {
    let c = CHUNK_SIZE as f64;
    let i = ((pos.0 - (pos.0 % c)) / c) as usize;
    let j = ((pos.1 - (pos.1 % c)) / c) as usize;

    (i, j)
}

pub fn oob((i, j): (f64, f64)) -> bool {
    let w = W_SIZE as f64;
    i < 0. || j < 0. || i > w || j > w
}

pub fn balls_collide(b1: &Being, b2: &Being) -> bool {
    let centre_dist = dist_2d(b1.pos, b2.pos);
    let (r1, r2) = (b1.radius, b2.radius);

    centre_dist < r1 + r2
}

pub fn resolve_balls(b1: &mut Being, b2: &Being) {
    let (i1, j1) = b1.pos;
    let (i2, j2) = b2.pos;
    let c1c2 = (i2 - i1, j2 - j1);
    let half_dist = scale_2d(c1c2, 0.5);

    b1.pos = add_2d((i1, j1), scale_2d(half_dist, -1.));
}

pub struct Being {
    radius: f64,
    pos: (f64, f64),
    rotation: f64,
    speed: f64,
    chunk: (usize, usize),
    id: usize,
}

struct Chunk {
    balls: Vec<Being>,
    cells: Vec<Vec<HashSet<usize>>>,
    cell_size: usize,
    ball_id: usize,
}

impl Chunk {
    pub fn new(n_cells: usize) -> Self {
        assert!(W_SIZE % n_cells == 0);
        Chunk {
            balls: vec![],
            cells: (0..n_cells)
                .into_iter()
                .map(|_| (0..n_cells).into_iter().map(|_| HashSet::new()).collect())
                .collect(),
            cell_size: W_SIZE / n_cells,
            ball_id: 0,
        }
    }

    pub fn add_ball(&mut self, radius: f64, pos: (f64, f64), rotation: f64, speed: f64) {
        let (i, j) = pos_to_chunk(pos);
        self.balls.push(Being {
            radius: radius,
            pos: pos,
            rotation: rotation,
            speed: speed,
            chunk: (i, j),
            id: self.ball_id,
        });
        self.cells[i][j].insert(self.ball_id);
        self.ball_id += 1;
    }

    pub fn move_balls(&mut self, substeps: usize) {
        let s = substeps as f64;
       for _ in 0..substeps{let w = W_SIZE as f64;
        self.balls.iter_mut().for_each(|ball| {
            let move_vec = scale_2d(dir_from_theta(ball.rotation), ball.speed / s);
            let (newi, newj) = add_2d(ball.pos, move_vec);

            let r = ball.radius;
            if !oob((newi, newj)) {
                ball.pos = add_2d(ball.pos, move_vec);
            }

            let (ni, nj) = pos_to_chunk(ball.pos);
            if !same_index((ni, nj), ball.chunk) {
                let (oi, oj) = ball.chunk;
                ball.chunk = (ni, nj);

                self.cells[oi][oj].remove(&ball.id);
                self.cells[ni][nj].insert(ball.id);
            }
        })}
    }

    pub fn check_collisions(&mut self) {
        for i in 0..N_CELLS {
            for j in 0..N_CELLS {
                for id1 in &self.cells[i][j] {
                    for (di, dj) in [
                        (-1, -1),
                        (-1, 0),
                        (-1, 1),
                        (0, -1),
                        (0, 0),
                        (0, 1),
                        (1, -1),
                        (1, 0),
                        (1, 1),
                    ] {
                        let (ni, nj) = ((i as isize + di), (j as isize + dj));
                        let w = N_CELLS as isize;
                        if !(ni < 0 || ni >= w || nj < 0 || nj >= w) {
                            let (ni, nj) = (ni as usize, nj as usize);
                            for id2 in &self.cells[ni][nj] {
                                if *id1 != *id2 {
                                    let (mut b1, mut b2) = self.balls.get2_mut(*id1, *id2);
                                    if balls_collide(&b1.as_ref().unwrap(), &b2.as_ref().unwrap()) {
                                        let (i1, j1) = &b1.as_ref().unwrap().pos;
                                        let (i2, j2) = &b2.unwrap().pos;
                                        let c1c2 = (i2 - i1, j2 - j1);
                                        let half_dist = scale_2d(c1c2, 0.5);

                                        let new_pos = add_2d((*i1, *j1), scale_2d(half_dist, -1.));
                                        if !oob(new_pos) {
                                            b1.unwrap().pos.0 = new_pos.0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn step(&mut self, substeps: usize) {
        for _ in 0..substeps{self.move_balls(substeps);
        self.check_collisions();}
    }
}

pub struct World {}

fn main() {
    let threads: Vec<JoinHandle<_>> = (0..1)
        .into_par_iter()
        .map(|i| {
            thread::spawn(|| {
                assert!(W_SIZE % N_CELLS == 0);
                let mut chunk = Chunk::new(N_CELLS);
                let rdist = Uniform::new(100., (W_SIZE as f64) - 100.);
                let mut rng = thread_rng();

                for i in 1..2000 {
                    chunk.add_ball(
                        5.,
                        (rng.sample(rdist), rng.sample(rdist)),
                        rng.sample(rdist),
                        50.,
                    );
                }

                for i in 1..10000000_u64 {
                    if i % 60 == 0 {
                        println!("{}", i)
                    }
                    chunk.step(1);
                }
            })
        })
        .collect();

    for t in threads {
        t.join();
    }
}
