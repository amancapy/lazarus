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

pub fn oob((i, j): (f64, f64), r: f64) -> bool {
    let w = W_SIZE as f64;
    i - r < 0. || j - r < 0. || i + r > w || j + r > w
}

pub fn balls_collide(b1: &Being, b2: &Being) -> bool {
    let centre_dist = dist_2d(b1.pos, b2.pos);
    let (r1, r2) = (b1.radius, b2.radius);

    centre_dist < r1 + r2
}

pub fn obstruct_collide(b: &Being, o: &Obstruct) -> bool {
    let centre_dist = dist_2d(b.pos, o.pos);
    let (r1, r2) = (b.radius, o.radius);
    centre_dist < r1 + r2
}

pub struct Obstruct {
    radius: f64,
    pos: (f64, f64),
    age: f64,
    id: usize,
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
    obstructs: Vec<Obstruct>,
    cells: Vec<Vec<(HashSet<usize>, HashSet<usize>)>>,
    ball_id: usize,
    ob_id: usize,
}

impl Chunk {
    pub fn new(n_cells: usize) -> Self {
        assert!(W_SIZE % n_cells == 0);
        Chunk {
            balls: vec![],
            obstructs: vec![],
            cells: (0..n_cells)
                .into_iter()
                .map(|_| {
                    (0..n_cells)
                        .into_iter()
                        .map(|_| (HashSet::new(), HashSet::new()))
                        .collect()
                })
                .collect(),
            ball_id: 0,
            ob_id: 0,
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
        self.cells[i][j].0.insert(self.ball_id);
        self.ball_id += 1;
    }

    pub fn add_obstruct(&mut self, pos: (f64, f64)) {
        let (i, j) = pos_to_chunk(pos);
        self.obstructs.push(Obstruct {
            radius: 4.,
            pos: pos,
            age: 5.,
            id: self.ob_id,
        });
        self.cells[i][j].1.insert(self.ob_id);
        self.ob_id += 1;
    }

    pub fn move_balls(&mut self, substeps: usize) {
        let s = substeps as f64;
        for _ in 0..substeps {
            let w = W_SIZE as f64;
            self.balls.iter_mut().for_each(|ball| {
                let move_vec = scale_2d(dir_from_theta(ball.rotation), ball.speed / s);
                let (newi, newj) = add_2d(ball.pos, move_vec);

                let r = ball.radius;


                if !oob((newi, newj), r) {
                    ball.pos = add_2d(ball.pos, move_vec);
                }

                let (ni, nj) = pos_to_chunk(ball.pos);
            })
        }
    }

    pub fn check_collisions(&mut self) {
        for i in 0..N_CELLS {
            for j in 0..N_CELLS {
                for id1 in &self.cells[i][j].0 {
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
                            for id2 in &self.cells[ni][nj].0 {
                                if *id1 != *id2 {
                                    let (b1, b2) = self.balls.get2_mut(*id1, *id2);
                                    if balls_collide(&b1.as_ref().unwrap(), &b2.as_ref().unwrap()) {
                                        let (i1, j1) = &b1.as_ref().unwrap().pos;
                                        let (i2, j2) = &b2.unwrap().pos;
                                        let c1c2 = (i2 - i1, j2 - j1);
                                        let half_dist = scale_2d(c1c2, -0.5);

                                        let new_pos = add_2d((*i1, *j1), half_dist);
                                        if !oob(new_pos, b1.as_ref().unwrap().radius) {
                                            b1.unwrap().pos.0 = new_pos.0;
                                        }
                                    }
                                }
                            }

                            for ob_id in &self.cells[ni][nj].1 {
                                let b = self.balls.get_mut(*id1);
                                let o = self.obstructs.get_mut(*ob_id);
                                if obstruct_collide(&b.as_ref().unwrap(), &o.as_ref().unwrap()) {
                                    let (i1, j1) = &b.as_ref().unwrap().pos;
                                    let (i2, j2) = &o.unwrap().pos;
                                    let c1c2 = (i2 - i1, j2 - j1);
                                    let half_dist = scale_2d(c1c2, 0.5);

                                    let new_pos = add_2d((*i1, *j1), scale_2d(half_dist, -1.));
                                        if !oob(new_pos, b.as_ref().unwrap().radius) {
                                            b.unwrap().pos.0 = new_pos.0;
                                        }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn update_cells(&mut self) {
        for b in &mut self.balls {
            let (bi, bj) = b.pos;
            let (oi, oj) = b.chunk;
            let (i, j) = pos_to_chunk((bi, bj));

            if !same_index((oi, oj), (i, j)) {
                b.chunk = (i, j);

                self.cells[oi][oj].0.remove(&b.id);
                self.cells[i][j].0.insert(b.id);
            }
        }
    }

    pub fn step(&mut self, substeps: usize) {
        for _ in 0..substeps {
            self.move_balls(substeps);
            self.check_collisions();
            self.update_cells();
        }
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

                for i in 1..2500 {
                    chunk.add_ball(
                        5.,
                        (rng.sample(rdist), rng.sample(rdist)),
                        rng.sample(rdist),
                        50.,
                    );
                }

                for i in 1..2500 {
                    chunk.add_obstruct(
                        (rng.sample(rdist), rng.sample(rdist)));
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
