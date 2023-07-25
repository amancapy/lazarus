use rand::{distributions::Uniform, prelude::*};
use rayon::prelude::*;
use splitmut::SplitMut;

const W_SIZE: usize = 1000;
const N_CELLS: usize = 25;
const CELL_SIZE: usize = W_SIZE / N_CELLS;
const W_FLOAT: f64 = W_SIZE as f64;
const HZ: usize = 60;

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
    i * N_CELLS + j
}

fn dir_from_theta(theta: f64) -> (f64, f64) {
    (theta.cos(), theta.sin())
}

fn same_index((a, b): (usize, usize), (c, d): (usize, usize)) -> bool {
    a == c && b == d
}

pub fn pos_to_cell(pos: (f64, f64)) -> (usize, usize) {
    let c = CELL_SIZE as f64;
    let i = ((pos.0 - (pos.0 % c)) / c) as usize;
    let j = ((pos.1 - (pos.1 % c)) / c) as usize;

    (i, j)
}

pub fn lef_border_trespass(i: f64, r: f64) -> bool {
    i - r < 1.
}

pub fn rig_border_trespass(i: f64, r: f64) -> bool {
    i + r >= W_FLOAT - 1.
}

pub fn top_border_trespass(j: f64, r: f64) -> bool {
    j - r < 1.
}

pub fn bot_border_trespass(j: f64, r: f64) -> bool {
    j + r >= W_FLOAT - 1.
}

pub fn oob((i, j): (f64, f64), r: f64) -> bool {
    lef_border_trespass(i, r)
        || rig_border_trespass(i, r)
        || top_border_trespass(j, r)
        || bot_border_trespass(j, r)
}

pub fn balls_collide(b1: &Being, b2: &Being) -> bool {
    let centre_dist = dist_2d(b1.pos, b2.pos);
    let (r1, r2) = (b1.radius, b2.radius);

    centre_dist <= r1 + r2
}

pub fn obstruct_collide(b: &Being, o: &Obstruct) -> bool {
    let centre_dist = dist_2d(b.pos, o.pos);
    let (r1, r2) = (b.radius, o.radius);
    centre_dist <= r1 + r2
}

pub fn food_collide(b: &Being, f: &Food) -> bool {
    let centre_dist = dist_2d(b.pos, f.pos);
    let (r1, r2) = (b.radius, 1.);
    centre_dist <= r1 + r2
}

pub struct Obstruct {
    radius: f64,
    pos: (f64, f64),
    age: f64,
    id: usize,
}

#[derive(Debug)]
pub struct Being {
    radius: f64,
    pos: (f64, f64),
    rotation: f64,
    speed: f64,
    cell: (usize, usize),
    id: usize,

    pos_update: (f64, f64),
}

pub struct Food {
    pos: (f64, f64),
    age: f64,
    id: usize,
}

struct World {
    balls: Vec<Being>,
    obstructs: Vec<Obstruct>,
    foods: Vec<Food>,

    being_cells: Vec<Vec<usize>>,
    obstruct_cells: Vec<Vec<usize>>,
    food_cells: Vec<Vec<usize>>,

    ball_id: usize,
    ob_id: usize,
    food_id: usize,

    ball_collision_count: usize,
    obstruct_collision_count: usize,
    food_collision_count: usize,
}

impl World {
    pub fn new(n_cells: usize) -> Self {
        assert!(W_SIZE % n_cells == 0);
        World {
            balls: vec![],
            obstructs: vec![],
            foods: vec![],

            being_cells: (0..(n_cells + 1).pow(2)).map(|_| Vec::new()).collect(),
            obstruct_cells: (0..(n_cells + 1).pow(2)).map(|_| Vec::new()).collect(),
            food_cells: (0..(n_cells + 1).pow(2)).map(|_| Vec::new()).collect(),

            ball_id: 0,
            ob_id: 0,
            food_id: 0,

            ball_collision_count: 0,
            obstruct_collision_count: 0,
            food_collision_count: 0,
        }
    }

    pub fn add_ball(&mut self, radius: f64, pos: (f64, f64), rotation: f64, speed: f64) {
        let (i, j) = pos_to_cell(pos);
        self.balls.push(Being {
            radius: radius,
            pos: pos,
            rotation: rotation,
            speed: speed,
            cell: (i, j),
            id: self.ball_id,

            pos_update: (0., 0.),
        });
        let ij = two_to_one((i, j));
        self.being_cells[ij].push(self.ball_id);
        self.ball_id += 1;
    }

    pub fn add_obstruct(&mut self, pos: (f64, f64)) {
        let (i, j) = pos_to_cell(pos);
        self.obstructs.push(Obstruct {
            radius: 2.,
            pos: pos,
            age: 5.,
            id: self.ob_id,
        });

        let ij = two_to_one((i, j));
        self.obstruct_cells[ij].push(self.ob_id);
        self.ob_id += 1;
    }

    pub fn add_food(&mut self, pos: (f64, f64)) {
        let (i, j) = pos_to_cell(pos);
        self.foods.push(Food {
            pos: pos,
            age: 5.,
            id: self.food_id,
        });

        let ij = two_to_one((i, j));
        self.food_cells[ij].push(self.food_id);
        self.food_id += 1;
    }
    pub fn move_balls(&mut self, substeps: usize) {
        let s = substeps as f64;

        let rdist = Uniform::new(1., (W_SIZE as f64) - 1.);
        let mut rng = thread_rng();

        for _ in 0..substeps {
            let w = W_SIZE as f64;
            self.balls.iter_mut().for_each(|ball| {
                let move_vec = scale_2d(dir_from_theta(ball.rotation), ball.speed / s);
                let (newi, newj) = add_2d(ball.pos, move_vec);

                let r = ball.radius;

                // TEMP TEMP TEMP TEMP NOTICE TEMP
                let (newi, newj) = (rng.sample(rdist), rng.sample(rdist));

                if !oob((newi, newj), r) {
                    ball.pos = (newi, newj);
                }
            });
        }
    }

    pub fn check_collisions(&mut self, timestep: usize) {
        for i in 0..N_CELLS {
            for j in 0..N_CELLS {
                let ij = two_to_one((i, j));
                for id1 in &self.being_cells[ij] {
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
                        let (ni, nj) = ((i as isize) + di, (j as isize) + dj);
                        let w = N_CELLS as isize;
                        if !(ni < 0 || ni >= w || nj < 0 || nj >= w) {
                            let (ni, nj) = (ni as usize, nj as usize);
                            let nij = two_to_one((ni, nj));

                            for id2 in &self.being_cells[nij] {
                                if !(*id1 == *id2) {
                                    let (b1, b2) = self.balls.get2_mut(*id1, *id2);

                                    let b1_ref = b1.as_ref().unwrap();
                                    let b2_ref = b2.as_ref().unwrap();

                                    if balls_collide(b1_ref, b2_ref) {
                                        // println!("{:?}    {:?}", b1_ref.pos, b2_ref.pos);
                                        self.ball_collision_count += 1;

                                        let (i1, j1) = b1_ref.pos;
                                        let (i2, j2) = b2.unwrap().pos;
                                        let c1c2 = (i2 - i1, j2 - j1);
                                        let half_dist = scale_2d(c1c2, -0.5);

                                        let new_pos = add_2d((i1, j1), half_dist);
                                        if !oob(new_pos, b1_ref.radius) {
                                            b1.unwrap().pos_update = half_dist;
                                        }
                                    }
                                }
                            }

                            for ob_id in &self.obstruct_cells[nij] {
                                let b = self.balls.get_mut(*id1); // see this line here happens 3 times for some reason because the bc won't allow non overlapping borrows from the same vec even if i use a splitmut method the second time but i was assured the compiler compiles this away so who knows. for later.
                                let o = self.obstructs.get_mut(*ob_id);

                                let b_ref = b.as_ref().unwrap();

                                if obstruct_collide(b_ref, o.as_ref().unwrap()) {
                                    self.obstruct_collision_count += 1;
                                    let (i1, j1) = b_ref.pos;
                                    let (i2, j2) = o.unwrap().pos;

                                    let c1c2 = (i2 - i1, j2 - j1);
                                    let half_dist = scale_2d(c1c2, -0.5);

                                    b.unwrap().pos_update = half_dist;
                                }
                            }

                            for f_id in &self.food_cells[nij] {
                                let b = self.balls.get_mut(*id1);
                                let f = self.foods.get_mut(*f_id);

                                let b_ref = b.as_ref().unwrap();

                                if food_collide(b_ref, f.as_ref().unwrap()) {
                                    // food.do_something(), this one along with beings dying completely ruins the sequential id scheme, sol here.
                                    self.food_collision_count += 1;
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
            let new_pos = add_2d(b.pos, b.pos_update);

            if !oob(new_pos, b.radius) {
                let (oi, oj) = b.cell;
                let (i, j) = pos_to_cell(new_pos);

                if !same_index((oi, oj), (i, j)) {
                    b.cell = (i, j);

                    let oij = two_to_one((oi, oj));
                    let ij = two_to_one((i, j));

                    self.being_cells[oij].retain(|x| x != &b.id);
                    self.being_cells[ij].push(b.id);
                }
            }
        }
    }

    pub fn step(&mut self, substeps: usize, timestep: usize) {
        for _ in 0..substeps {
            self.move_balls(substeps);
            self.check_collisions(timestep);
            self.update_cells();
        }
    }
}

fn main() {
    assert!(W_SIZE % N_CELLS == 0);
    let mut world = World::new(N_CELLS);
    let rdist = Uniform::new(1., (W_SIZE as f64) - 1.);
    let mut rng = thread_rng();

    for i in 1..10000 {
        world.add_ball(
            3.,
            (rng.sample(rdist), rng.sample(rdist)),
            rng.sample(rdist),
            1.,
        );
    }

    for i in 1..0 {
        world.add_obstruct((rng.sample(rdist), rng.sample(rdist)));
    }

    for i in 1..0 {
        world.add_food((rng.sample(rdist), rng.sample(rdist)))
    }

    for i in 1..10000000_usize {
        if i % HZ == 0 {
            // println!("{}", i / HZ)
            println!(
                "{} {} {}",
                world.ball_collision_count,
                world.obstruct_collision_count,
                world.food_collision_count
            );
            world.ball_collision_count = 0;
            world.obstruct_collision_count = 0;
            world.food_collision_count = 0;
        }
        world.step(1, i);
    }
}
