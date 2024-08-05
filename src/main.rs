use being_nn::{tensorize_2dvec, Activation, Sigmoid, SumFxModel, Tanh};
use ggez::{
    conf::{Backend, NumSamples, WindowMode, WindowSetup},
    event,
    glam::*,
    graphics::{Canvas, Color, DrawParam, Image, InstanceArray},
    Context, GameResult,
};
use nn::Relu;
use rand::{seq::SliceRandom, thread_rng, Rng};
use slotmap::{DefaultKey, SlotMap};
use std::{
    borrow::{Borrow, BorrowMut},
    env,
    f32::consts::PI,
    path::PathBuf,
    process::{exit, id},
    thread::sleep,
    time::{Duration, SystemTime},
};

use burn::prelude::*;

mod being_nn;

#[rustfmt::skip]
pub mod consts {
    use burn::backend;

    pub const W_SIZE:                                 usize = 625;
    pub const N_CELLS:                                usize = 125;
    pub const CELL_SIZE:                              usize = W_SIZE / N_CELLS;
    pub const CELL_SIZE_FLOAT:                          f32 = CELL_SIZE as f32;
    pub const W_FLOAT:                                  f32 = W_SIZE as f32;
    pub const W_USIZE:                                  u32 = W_SIZE as u32;

    pub const B_START_COUNT:                          usize = 100;
    pub const REWORLDING_THRESHOLD:                   usize = 40;

    pub const B_FOV:                                  isize = 10;
    pub const B_FOV_PX:                                 f32 = (B_FOV as usize * CELL_SIZE) as f32;
    pub const B_SPEED:                                  f32 = 0.5;
    pub const B_RADIUS:                                 f32 = 3.5;
    pub const O_RADIUS:                                 f32 = 3.5;
    pub const F_RADIUS:                                 f32 = 3.5;
    pub const S_RADIUS:                                 f32 = 1.5;

    pub const GENOME_LEN:                             usize = 10;                  // future prospect
    pub const S_GROW_RATE:                              f32 = 1.;

    pub const B_DEATH_ENERGY:                           f32 = 0.5;
    pub const B_SCATTER_RADIUS:                         f32 = 10.;
    pub const B_SCATTER_COUNT:                        usize = 100;

    pub const BASE_ANG_SPEED_DEGREES:                   f32 = 10.;

    pub const B_START_ENERGY:                           f32 = 10.;
    pub const O_START_HEALTH:                           f32 = 25.;
    pub const S_START_AGE:                              f32 = 5.;
    pub const F_VAL:                                    f32 = 2.;
    
    pub const B_TIRE_RATE:                              f32 = 0.01;
    pub const B_MOVE_TIRE_RATE:                         f32 = 0.01;
    pub const B_ROT_TIRE_RATE:                          f32 = 0.01;
    pub const O_AGE_RATE:                               f32 = 0.001;
    pub const F_ROT_RATE:                               f32 = F_VAL / 1000.;
    pub const S_SOFTEN_RATE:                            f32 = 0.1;

    pub const B_HEADON_DAMAGE:                          f32 = 0.25;
    pub const B_REAR_DAMAGE:                            f32 = 1.;
    pub const HEADON_B_HITS_O_DAMAGE:                   f32 = 0.1;
    pub const SPAWN_O_RATIO:                            f32 = 0.1;                 // fraction of start_energy spent to spawn obstruct
    pub const SPAWN_S_RATIO:                            f32 = 0.05;                // fraction of start_energy spent to speak
    pub const OOB_PENALTY:                              f32 = 0.25;

    pub const LOW_ENERGY_SPEED_DAMP_RATE:               f32 = 0.001;                 // beings slow down when their energy runs low
    pub const OFF_DIR_MOVEMENT_SPEED_DAMP_RATE:         f32 = 0.001;                 // beings slow down when not moving face-forward

    pub const N_FOOD_SPAWN_PER_STEP:                  usize = 1;
    
    pub static mut MAX_FOOD:                          usize = 750;
    pub const MIN_FOOD:                               usize = 25;
    pub const MAX_FOOD_REDUCTION:                     usize = 5;

    pub const SPEECHLET_LEN:                          usize = 8;                   // length of the sound vector a being can emit
    pub const B_OUTPUT_LEN:                           usize = 4 + SPEECHLET_LEN;   // (f-b, rotate, spawn obstruct, spawn_speechlet, *speechlet)
    
    pub type BACKEND                                        = backend::NdArray;
    pub const DEVICE:       backend::ndarray::NdArrayDevice = backend::ndarray::NdArrayDevice::Cpu;
}

use consts::*;

// maps 2D space-partition index to 1D Vec index
fn two_to_one((i, j): (usize, usize)) -> usize {
    i * N_CELLS + j
}

fn dir_from_theta(theta: f32) -> Vec2 {
    Vec2::from_angle(theta)
}

fn same_partition_index((a, b): (usize, usize), (c, d): (usize, usize)) -> bool {
    a == c && b == d
}

// maps an entity's position to the cell that contains its centre
pub fn pos_to_cell(pos: Vec2) -> (usize, usize) {
    let c = CELL_SIZE as f32;
    let i = ((pos[0] - (pos[0] % c)) / c) as usize;
    let j = ((pos[1] - (pos[1] % c)) / c) as usize;

    (i, j)
}

pub fn lef_border_trespass(i: f32, r: f32) -> bool {
    i - r <= 1.
}

pub fn rig_border_trespass(i: f32, r: f32) -> bool {
    i + r >= W_FLOAT - 1.
}

pub fn top_border_trespass(j: f32, r: f32) -> bool {
    j - r <= 1.
}

pub fn bot_border_trespass(j: f32, r: f32) -> bool {
    j + r >= W_FLOAT - 1.
}

// out of bounds
pub fn oob(xy: Vec2, r: f32) -> bool {
    let (x, y) = (xy[0], xy[1]);
    lef_border_trespass(x, r)
        || rig_border_trespass(x, r)
        || top_border_trespass(y, r)
        || bot_border_trespass(y, r)
}

pub fn b_collides_b(b1: &Being, b2: &Being) -> (f32, f32, Vec2, [f32; 3 + GENOME_LEN]) {
    let c1c2 = b2.pos - b1.pos;
    let centre_dist = c1c2.length();
    let (r1, r2) = (b1.radius, b2.radius);

    let other_genome = b2.genome.clone();
    let rel_vec = [
        b1.pos.angle_between(b2.pos) / PI,
        centre_dist / B_FOV_PX,
        b2.energy / B_START_ENERGY,
    ];

    let mut full_vec = [0.; 3 + GENOME_LEN];
    (0..3).for_each(|i| {
        full_vec[i] = rel_vec[i];
    });
    (0..GENOME_LEN).for_each(|i| {
        full_vec[i + 3] = other_genome[i];
    });

    (r1 + r2 - centre_dist, centre_dist, c1c2, full_vec)
}

pub fn b_collides_o(b: &Being, o: &Obstruct) -> (f32, f32, Vec2, [f32; 4]) {
    let c1c2 = o.pos - b.pos;
    let centre_dist = c1c2.length();
    let (r1, r2) = (b.radius, O_RADIUS);

    (
        r1 + r2 - centre_dist,
        centre_dist,
        c1c2,
        [
            0.,
            centre_dist / B_FOV_PX,
            b.pos.angle_between(o.pos) / PI,
            o.age / O_START_HEALTH,
        ],
    )
}

pub fn b_collides_f(b: &Being, f: &Food) -> (f32, [f32; 4]) {
    let centre_dist = b.pos.distance(f.pos);
    let (r1, r2) = (b.radius, F_RADIUS);
    (
        r1 + r2 - centre_dist,
        [
            1.,
            centre_dist / B_FOV_PX,
            b.pos.angle_between(f.pos) / PI,
            f.val / F_VAL,
        ],
    )
}

pub fn b_collides_s(b: &Being, s: &Speechlet) -> f32 {
    let c1c2 = s.pos - b.pos;
    let centre_dist = c1c2.length();
    let (r1, r2) = (b.radius, S_RADIUS);

    r1 + r2 - centre_dist
}

pub fn is_border_in_sight(pos: Vec2, rot: f32) -> [f32; 4] {
    let (x, y) = (pos.x, pos.y);
    let mut rel_vec: [f32; 4] = [1., 0., 1., 0.];
    let w = W_SIZE as f32;
    if x + B_FOV_PX > w {
        rel_vec[0] = (w - x) / B_FOV_PX;
        rel_vec[1] = rot + 0.5;
    } else if x - B_FOV_PX < 0. {
        rel_vec[0] = x / B_FOV_PX;
        rel_vec[1] = rot - 0.5;
    }
    if y + B_FOV_PX > w {
        rel_vec[2] = (w - y) / B_FOV_PX;
        rel_vec[3] = rot + 1.;
    } else if y - B_FOV_PX < 0. {
        rel_vec[2] = y / B_FOV_PX;
        rel_vec[3] = rot;
    }

    rel_vec
}

#[derive(Debug)]
pub struct Being {
    pos: Vec2,
    radius: f32,
    rotation: f32,
    energy: f32,
    genome: [f32; GENOME_LEN],

    cell: (usize, usize),
    id: usize,

    pos_update: Vec2,
    energy_update: f32,
    rotation_update: f32,

    being_inputs: Vec<Vec<f32>>,
    food_obstruct_inputs: Vec<Vec<f32>>,
    speechlet_inputs: Vec<Vec<f32>>,

    output: [f32; B_OUTPUT_LEN],
}

pub struct Obstruct {
    pos: Vec2,
    age: f32,
    id: usize,
}

pub struct Food {
    pos: Vec2,
    val: f32,
    eaten: bool,

    is_flesh: bool,
    id: usize,
}

#[derive(Debug)]
pub struct Speechlet {
    speechlet: [f32; SPEECHLET_LEN],
    pos: Vec2,
    radius: f32,
    age: f32,

    recepient_being_ids: Vec<usize>,
}

pub struct World<const D: usize> {
    beings_and_models: SlotMap<DefaultKey, (Being, SumFxModel<BACKEND>)>,
    obstructs: SlotMap<DefaultKey, Obstruct>,
    foods: SlotMap<DefaultKey, Food>,
    speechlets: SlotMap<DefaultKey, Speechlet>,

    being_cells: Vec<Vec<DefaultKey>>,
    obstruct_cells: Vec<Vec<DefaultKey>>,
    food_cells: Vec<Vec<DefaultKey>>,
    speechlet_cells: Vec<Vec<DefaultKey>>,

    being_id: usize,
    ob_id: usize,
    food_id: usize,

    being_deaths: Vec<(DefaultKey, Vec2)>,
    obstruct_deaths: Vec<(DefaultKey, Vec2)>,
    food_deaths: Vec<(DefaultKey, Vec2)>,
    speechlet_deaths: Vec<(DefaultKey, Vec2)>,

    fov_indices: Vec<(isize, isize)>,

    age: usize,
    generation: usize,
    last_survivors: Vec<SumFxModel<BACKEND>>,
}

impl<const D: usize> World<D> {
    pub fn new() -> Self {
        World::<D> {
            beings_and_models: SlotMap::new(),
            obstructs: SlotMap::new(),
            foods: SlotMap::new(),
            speechlets: SlotMap::new(),

            being_cells: (0..(N_CELLS + 1).pow(2)).map(|_| Vec::new()).collect(),
            obstruct_cells: (0..(N_CELLS + 1).pow(2)).map(|_| Vec::new()).collect(),
            food_cells: (0..(N_CELLS + 1).pow(2)).map(|_| Vec::new()).collect(),
            speechlet_cells: (0..(N_CELLS + 1).pow(2)).map(|_| Vec::new()).collect(),

            being_id: 0,
            ob_id: 0,
            food_id: 0,

            being_deaths: vec![],
            food_deaths: vec![],
            obstruct_deaths: vec![],
            speechlet_deaths: vec![],

            fov_indices: (-B_FOV..=B_FOV)
                .flat_map(|i| (-B_FOV..=B_FOV).map(move |j| (i, j)))
                .filter(|(i, j)| i.pow(2) + j.pow(2) <= B_FOV.pow(2))
                .collect(),

            age: 0,
            generation: 0,
            last_survivors: vec![],
        }
    }

    // a world populated as intended, this fn mainly to relieve World::new() of some clutter
    pub fn standard_world() -> Self {
        let mut world = World::new();
        let mut rng = thread_rng();

        for _ in 0..B_START_COUNT {
            world.add_being(
                B_RADIUS,
                Vec2::new(
                    rng.gen_range(B_RADIUS..W_FLOAT - B_RADIUS),
                    rng.gen_range(B_RADIUS..W_FLOAT - B_RADIUS),
                ),
                rng.gen_range(-PI..PI),
                B_START_ENERGY,
                [0.; GENOME_LEN],
                SumFxModel::standard_model(&DEVICE),
            );
        }

        unsafe {
            for _ in 0..MAX_FOOD {
                world.add_food(
                    Vec2::new(
                        rng.gen_range(1.0..W_FLOAT - 1.),
                        rng.gen_range(1.0..W_FLOAT - 1.),
                    ),
                    F_VAL,
                    false,
                );
            }
        }

        world
    }

    pub fn add_being(
        &mut self,
        radius: f32,
        pos: Vec2,
        rotation: f32,
        health: f32,
        genome: [f32; GENOME_LEN],

        model: SumFxModel<BACKEND>,
    ) {
        let (i, j) = pos_to_cell(pos);

        let being = Being {
            radius: radius,
            pos: pos,
            rotation: rotation,
            energy: health,
            genome,

            cell: (i, j),
            id: self.being_id,

            pos_update: Vec2::new(0., 0.),
            energy_update: 0.,
            rotation_update: 0.,

            being_inputs: vec![],
            food_obstruct_inputs: vec![],
            speechlet_inputs: vec![],

            output: [0.; B_OUTPUT_LEN],
        };

        let k = self.beings_and_models.insert((being, model));
        let ij = two_to_one((i, j));
        self.being_cells[ij].push(k);

        self.being_id += 1;
    }

    pub fn add_obstruct(&mut self, pos: Vec2) {
        let (i, j) = pos_to_cell(pos);

        let obstruct = Obstruct {
            pos: pos,
            age: O_START_HEALTH,
            id: self.ob_id,
        };

        let k = self.obstructs.insert(obstruct);

        let ij = two_to_one((i, j));
        self.obstruct_cells[ij].push(k);
        self.ob_id += 1;
    }

    pub fn add_food(&mut self, pos: Vec2, val: f32, is_flesh: bool) {
        let (i, j) = pos_to_cell(pos);

        let food = Food {
            pos: pos,
            val: val,
            eaten: false,
            is_flesh: is_flesh,

            id: self.food_id,
        };

        let k = self.foods.insert(food);

        let ij = two_to_one((i, j));
        self.food_cells[ij].push(k);
        self.food_id += 1;
    }

    pub fn add_speechlet(&mut self, speechlet: [f32; SPEECHLET_LEN], pos: Vec2) {
        let (i, j) = pos_to_cell(pos);

        let speechlet = Speechlet {
            speechlet: speechlet,
            pos: pos,
            radius: S_RADIUS,
            age: S_START_AGE,

            recepient_being_ids: vec![],
        };

        let k = self.speechlets.insert(speechlet);
        let ij = two_to_one((i, j));
        self.speechlet_cells[ij].push(k);
    }

    pub fn move_beings(&mut self, substeps: usize) {
        let s = substeps as f32;

        for _ in 0..substeps {
            self.beings_and_models
                .iter_mut()
                .for_each(|(_, (being, _))| {
                    let being_rotation = dir_from_theta(being.rotation);
                    let move_vec = being.output[0] * being_rotation;
                    let newxy = being.pos + (move_vec * (1. - LOW_ENERGY_SPEED_DAMP_RATE) * (being.energy / B_START_ENERGY) * B_SPEED);

                    if !oob(newxy, being.radius) {
                        let pos_update = move_vec / s;
                        let rot_update = (being.output[1] * PI) / s;

                        being.pos_update += pos_update;
                        being.rotation_update += (being.output[1] * PI) / s;

                        being.energy_update -= (pos_update.length() / B_SPEED) * B_MOVE_TIRE_RATE;
                        being.energy_update -= (rot_update.abs() / PI) * B_ROT_TIRE_RATE;
                    } else {
                        let move_vec = -dir_from_theta(being.rotation) * 1.5; // hacky
                        being.pos_update += move_vec / s;

                        being.energy_update -= OOB_PENALTY;
                    }
                });
        }
    }

    pub fn grow_speechlets(&mut self) {
        self.speechlets.iter_mut().for_each(|(_, s)| {
            s.radius += S_RADIUS;
        });
    }

    pub fn check_collisions(&mut self, substeps: usize) {
        let w = N_CELLS as isize;
        let s = substeps as f32;

        for i in 0..N_CELLS {
            for j in 0..N_CELLS {
                // for each partition
                let ij = two_to_one((i, j));

                for id1 in &self.being_cells[ij] {
                    for (di, dj) in &self.fov_indices {
                        let (ni, nj) = ((i as isize) + di, (j as isize) + dj);

                        if !(ni < 0 || ni >= w || nj < 0 || nj >= w) {
                            // if valid partition
                            let (ni, nj) = (ni as usize, nj as usize);
                            let nij = two_to_one((ni, nj));

                            for id2 in &self.being_cells[nij] {
                                // for another being in the same or one of the 8 neighbouring cells
                                if !(id1 == id2) {
                                    let (overlap, centre_dist, c1c2, rel_vec) = b_collides_b(
                                        &self.beings_and_models.get(*id1).unwrap().0,
                                        &self.beings_and_models.get(*id2).unwrap().0,
                                    );
                                    let (b1, _) = self.beings_and_models.get_mut(*id1).unwrap();
                                    b1.being_inputs.push(Vec::from(rel_vec));

                                    if overlap > 0. {
                                        let d_p = overlap / centre_dist * c1c2;
                                        let half_dist = d_p / 1.5;

                                        let new_pos = b1.pos - half_dist;
                                        if !oob(new_pos, b1.radius) {
                                            b1.pos_update -= half_dist;
                                        }

                                        let b1_dir = dir_from_theta(b1.rotation);
                                        let axis_alignment = b1_dir.dot(c1c2.normalize());

                                        if axis_alignment > 0. {
                                            b1.energy_update -=
                                                B_HEADON_DAMAGE * axis_alignment / s;
                                        } else {
                                            b1.energy_update -=
                                                B_REAR_DAMAGE * axis_alignment.abs() / s;
                                        }
                                    }
                                }
                            }

                            for f_id in &self.food_cells[nij] {
                                // for a food similarly
                                let (b, __) = self.beings_and_models.get_mut(*id1).unwrap();
                                let f = self.foods.get_mut(*f_id);

                                let f_ref = f.as_ref().unwrap();

                                let (overlap, rel_vec) = b_collides_f(&b, f_ref);
                                b.food_obstruct_inputs.push(Vec::from(rel_vec));

                                if overlap > 0. && !f_ref.eaten && b.energy <= B_START_ENERGY {
                                    b.energy_update += f_ref.val;
                                    self.food_deaths.push((*f_id, f_ref.pos));
                                    f.unwrap().eaten = true;
                                }
                            }

                            for ob_id in &self.obstruct_cells[nij] {
                                // for an obstruct similarly
                                let (b, _) = self.beings_and_models.get_mut(*id1).unwrap();
                                let o = self.obstructs.get_mut(*ob_id).unwrap();

                                let (overlap, centre_dist, c1c2, rel_vec) = b_collides_o(b, o);
                                b.food_obstruct_inputs.push(Vec::from(rel_vec));

                                if overlap > 0. {
                                    let d_p = overlap / centre_dist * c1c2;
                                    let half_dist = d_p / 1.5;
                                    b.pos_update -= half_dist;

                                    let b_dir = dir_from_theta(b.rotation);
                                    let axis_alignment = b_dir.dot(c1c2.normalize());

                                    if axis_alignment > 0. {
                                        b.energy_update -=
                                            HEADON_B_HITS_O_DAMAGE * axis_alignment / s;
                                    }
                                }
                            }

                            for s_id in &self.speechlet_cells[nij] {
                                let (b, _) = self.beings_and_models.get_mut(*id1).unwrap();
                                let s = self.speechlets.get_mut(*s_id).unwrap();

                                let overlap = b_collides_s(&b, &s);

                                if overlap > 0. && !s.recepient_being_ids.contains(&b.id) {
                                    b.speechlet_inputs.push(Vec::from(s.speechlet));
                                    s.recepient_being_ids.push(b.id);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // reflect changes in rotation, translation, collision resolution, fatigue, aging, death
    pub fn update_cells(&mut self) {
        for (k, (b, _)) in &mut self.beings_and_models {
            let new_pos = b.pos + b.pos_update;
            // println!("{}", b.pos_update.length());
            b.energy += b.energy_update;
            b.rotation += b.rotation_update;

            b.energy_update = 0.;
            b.rotation_update = 0.;

            if !oob(new_pos, b.radius) {
                b.pos = new_pos;
                b.pos_update = Vec2::ZERO;

                let (oi, oj) = b.cell;
                let (i, j) = pos_to_cell(new_pos);

                if !same_partition_index((oi, oj), (i, j)) {
                    b.cell = (i, j);

                    let oij = two_to_one((oi, oj));
                    let ij = two_to_one((i, j));

                    self.being_cells[oij].retain(|x| *x != k);
                    self.being_cells[ij].push(k);
                }
            }
        }
    }

    // beings tire and/or die
    pub fn tire_beings(&mut self) {
        for (k, (b, _)) in &mut self.beings_and_models {
            b.energy -= B_TIRE_RATE;

            if b.energy <= 0. {
                self.being_deaths.push((k, b.pos));
            }
        }

        let mut rng = thread_rng();
        for (k, pos) in &self.being_deaths.clone() {
            self.beings_and_models.remove(*k);
            self.being_cells[two_to_one(pos_to_cell(*pos))].retain(|x| x != k);

            for _ in 0..B_SCATTER_COUNT {
                let (theta, dist) = (rng.gen_range(-PI..PI), rng.gen_range(0.0..B_SCATTER_RADIUS));
                let dvec = Vec2::new(theta.cos() * dist, theta.sin() * dist);

                let food_pos = *pos + dvec;
                if !oob(food_pos, F_RADIUS) {
                    self.add_food(food_pos, B_DEATH_ENERGY / B_SCATTER_RADIUS as f32, true);
                };
            }
        }

        self.being_deaths.clear();
    }

    // walls crack and/or crumble
    pub fn age_obstructs(&mut self) {
        for (k, o) in &mut self.obstructs {
            o.age -= O_AGE_RATE;

            if o.age < 0.05 {
                self.obstruct_deaths.push((k, o.pos));
            }
        }

        for (k, pos) in &self.obstruct_deaths {
            self.obstructs.remove(*k);
            self.obstruct_cells[two_to_one(pos_to_cell(*pos))].retain(|x| x != k);
        }

        self.obstruct_deaths.clear();
    }

    // food grows stale and/or disappears
    pub fn age_foods(&mut self) {
        for (k, f) in &mut self.foods {
            f.val -= F_ROT_RATE;
            if f.val < 0. {
                self.food_deaths.push((k, f.pos));
            }
        }

        for (k, pos) in &self.food_deaths {
            self.foods.remove(*k);

            self.food_cells[two_to_one(pos_to_cell(*pos))].retain(|x| x != k);
        }

        self.food_deaths.clear();
    }

    pub fn soften_speechlets(&mut self) {
        for (k, s) in &mut self.speechlets {
            s.age -= S_SOFTEN_RATE;

            if s.age <= 0. {
                self.speechlet_deaths.push((k, s.pos));
            }
        }

        for (k, pos) in &self.speechlet_deaths {
            self.speechlets.remove(*k);
            self.speechlet_cells[two_to_one(pos_to_cell(*pos))].retain(|x| x != k);
        }

        self.speechlet_deaths.clear();
    }

    pub fn repop_foods(&mut self) {
        let mut rng = thread_rng();
        unsafe {
            for _ in 0..N_FOOD_SPAWN_PER_STEP {
                if self
                    .foods
                    .iter()
                    .filter(|(_, f)| !f.is_flesh)
                    .collect::<Vec<(DefaultKey, &Food)>>()
                    .len()
                    < MAX_FOOD
                {
                    let ij = Vec2::new(rng.gen_range(1.0..W_FLOAT), rng.gen_range(1.0..W_FLOAT));
                    self.add_food(ij, F_VAL, false);
                }
            }
        }
    }

    // has side-effects; probably not worth the effort to separate updates and effects
    pub fn perform_being_outputs(&mut self) {
        let mut obstruct_queue: Vec<Vec2> = Vec::new();
        let mut speechlet_queue: Vec<(Vec2, [f32; SPEECHLET_LEN])> = Vec::new();

        self.beings_and_models
            .iter_mut()
            .for_each(|(_, (b, model))| {
                b.being_inputs.push(vec![-1.; 3 + GENOME_LEN]);
                b.food_obstruct_inputs.push(vec![-1.; 4]);
                b.speechlet_inputs.push(vec![-1.; SPEECHLET_LEN]);

                let being_tensor = tensorize_2dvec(
                    &b.being_inputs,
                    [b.being_inputs.len(), GENOME_LEN + 3],
                    &DEVICE,
                );
                let fo_tensor = tensorize_2dvec(
                    &b.food_obstruct_inputs,
                    [b.food_obstruct_inputs.len(), 4],
                    &DEVICE,
                );
                let speechlet_tensor = tensorize_2dvec(
                    &b.speechlet_inputs,
                    [b.speechlet_inputs.len(), SPEECHLET_LEN],
                    &DEVICE,
                );

                let mut self_vec = is_border_in_sight(b.pos, b.rotation).to_vec();
                self_vec.extend([b.energy / B_START_ENERGY]);

                let self_tensor =
                    Tensor::<BACKEND, 1>::from_floats(self_vec.as_slice(), &DEVICE).reshape([1, 5]);

                b.being_inputs.clear();
                b.food_obstruct_inputs.clear();
                b.speechlet_inputs.clear();

                let model_output = model
                    .forward(being_tensor, fo_tensor, speechlet_tensor, self_tensor)
                    .into_data()
                    .value;

                let mut output = [0.; B_OUTPUT_LEN];
                (0..B_OUTPUT_LEN).into_iter().for_each(|i| {
                    output[i] = model_output[i];
                });

                b.output = output;

                if b.output[2] > 0. {
                    b.energy_update -= SPAWN_O_RATIO * B_START_ENERGY;
                    obstruct_queue.push(b.pos + dir_from_theta(b.rotation) * 2.);
                }

                let mut speechlet = [0.; SPEECHLET_LEN];
                (0..SPEECHLET_LEN).for_each(|i| {
                    speechlet[i] = b.output[i + 3];
                });

                if b.output[3] > 0. {
                    b.energy_update -= SPAWN_S_RATIO * B_START_ENERGY;
                    speechlet_queue.push((b.pos, speechlet));
                }
            });

        for pos in obstruct_queue {
            self.add_obstruct(pos);
        }
        for (pos, speechlet) in speechlet_queue {
            self.add_speechlet(speechlet, pos);
        }
    }

    pub fn reworld(&mut self) {
        if self.beings_and_models.len() < REWORLDING_THRESHOLD {
            unsafe {
                if MAX_FOOD > MIN_FOOD {
                    MAX_FOOD -= MAX_FOOD_REDUCTION;
                }
            }
            println!("generation: {}, world age: {}", self.generation, self.age);

            let mut surviving_models: Vec<SumFxModel<BACKEND>> = self
                .beings_and_models
                .iter_mut()
                .map(|(_, (_, m))| m.clone())
                .collect();

            let mut new_models: Vec<SumFxModel<BACKEND>> = vec![];

            let mut rng = thread_rng();
            if surviving_models.len() == 0 {
                println!("extinction");
                new_models = self.last_survivors.clone();
            } else {
                while new_models.len() + surviving_models.len() < B_START_COUNT {
                    let m1 = surviving_models.choose(&mut thread_rng()).unwrap();
                    let m2 = surviving_models.choose(&mut thread_rng()).unwrap();

                    let new_model = m1
                        .clone()
                        .crossover(m2.clone(), 0.05, &DEVICE)
                        .mutate(0.05, &DEVICE);
                    new_models.push(new_model);
                }
                self.last_survivors = surviving_models.clone();
            }

            self.beings_and_models.clear();
            self.foods.clear();
            self.obstructs.clear();
            self.speechlets.clear();

            unsafe {
                for _ in 0..MAX_FOOD {
                    self.add_food(
                        Vec2::new(
                            rng.gen_range(1.0..W_FLOAT - 1.),
                            rng.gen_range(1.0..W_FLOAT - 1.),
                        ),
                        F_VAL,
                        false,
                    );
                }
            }

            self.being_cells = (0..(N_CELLS + 1).pow(2)).map(|_| Vec::new()).collect();
            self.obstruct_cells = (0..(N_CELLS + 1).pow(2)).map(|_| Vec::new()).collect();
            self.food_cells = (0..(N_CELLS + 1).pow(2)).map(|_| Vec::new()).collect();
            self.speechlet_cells = (0..(N_CELLS + 1).pow(2)).map(|_| Vec::new()).collect();

            self.being_id = 0;
            self.ob_id = 0;
            self.food_id = 0;

            self.being_deaths.clear();
            self.food_deaths.clear();
            self.obstruct_deaths.clear();
            self.speechlet_deaths.clear();

            self.age = 0;
            self.generation += 1;

            surviving_models.extend(new_models);
            for m in surviving_models {
                self.add_being(
                    B_RADIUS,
                    Vec2::new(
                        rng.gen_range(B_RADIUS..W_FLOAT - B_RADIUS),
                        rng.gen_range(B_RADIUS..W_FLOAT - B_RADIUS),
                    ),
                    rng.gen_range(-PI..PI),
                    B_START_ENERGY,
                    [0.; GENOME_LEN],
                    m,
                );
            }
        }
    }

    pub fn step(&mut self, substeps: usize) {
        for _ in 0..substeps {
            self.move_beings(substeps);
            self.check_collisions(substeps);
            self.update_cells();
        }
        self.perform_being_outputs();
        self.grow_speechlets();
        self.tire_beings();
        self.age_foods();
        self.age_obstructs();
        self.soften_speechlets();
        self.repop_foods();

        self.reworld();

        self.age += 1;
    }
}

struct MainState<const D: usize> {
    being_instances: InstanceArray,
    obstruct_instances: InstanceArray,
    food_instances: InstanceArray,
    speechlet_instances: InstanceArray,
    world: World<D>,
}

impl<const D: usize> MainState<D> {
    fn new(ctx: &mut Context, w: World<D>) -> GameResult<MainState<D>> {
        let being = Image::from_path(ctx, "/red_circle.png")?;
        let obstruct = Image::from_path(ctx, "/white_circle.png")?;
        let food = Image::from_path(ctx, "/green_circle.png")?;
        let speechlet = Image::from_path(ctx, "/blue_circle.png")?;

        let being_instances = InstanceArray::new(ctx, being);
        let obstruct_instances = InstanceArray::new(ctx, obstruct);
        let food_instances = InstanceArray::new(ctx, food);
        let speechlet_instances = InstanceArray::new(ctx, speechlet);

        Ok(MainState {
            being_instances: being_instances,
            obstruct_instances: obstruct_instances,
            food_instances: food_instances,
            speechlet_instances: speechlet_instances,
            world: w,
        })
    }
}

impl<const D: usize> event::EventHandler<ggez::GameError> for MainState<D> {
    fn update(&mut self, ctx: &mut Context) -> Result<(), ggez::GameError> {
        // if self.world.age % 60 == 0 {
        //     println!(
        //         "generation: {}, timestep: {}, fps: {}, beings: {}, foods: {}",
        //         self.world.generation,
        //         self.world.age,
        //         ctx.time.fps(),
        //         self.world.beings_and_models.len(),
        //         self.world.foods.len()
        //     );
        // }

        self.world.step(1);
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> Result<(), ggez::GameError> {
        let mut canvas = Canvas::from_frame(ctx, Color::BLACK);

        self.speechlet_instances
            .set(self.world.speechlets.iter().map(|(_, s)| {
                let xy = s.pos;
                DrawParam::new()
                    .scale(Vec2::new(1., 1.) / 512. * s.radius)
                    .dest(xy)
                    .offset(Vec2::new(256., 256.))
                    .color(Color::new(1., 1., 1., s.age / S_START_AGE))
            }));

        self.food_instances
            .set(self.world.foods.iter().map(|(_, f)| {
                let xy = f.pos - Vec2::new(F_RADIUS, F_RADIUS);
                DrawParam::new()
                    .dest(xy.clone())
                    .scale(Vec2::new(1., 1.) / 2048. * 2. * F_RADIUS)
                    .color(Color::new(1., 1., 1., f.val / F_VAL))
            }));

        self.obstruct_instances
            .set(self.world.obstructs.iter().map(|(_, o)| {
                let xy = o.pos;
                DrawParam::new()
                    .dest(xy.clone())
                    .scale(Vec2::new(1., 1.) / 800. * 2. * O_RADIUS)
                    .color(Color::new(1., 1., 1., o.age / O_START_HEALTH))
            }));

        self.being_instances
            .set(self.world.beings_and_models.iter().map(|(_, (b, _))| {
                let xy = b.pos;
                DrawParam::new()
                    .scale(Vec2::new(1., 1.) / 400. * 2. * B_RADIUS)
                    .dest(xy)
                    .offset(Vec2::new(200., 200.))
                    .rotation(b.rotation)
                    .color(Color::new(1., 1., 1., b.energy / B_START_ENERGY))
            }));

        let param = DrawParam::new();
        canvas.draw(&self.speechlet_instances, param);
        canvas.draw(&self.food_instances, param);
        canvas.draw(&self.obstruct_instances, param);
        canvas.draw(&self.being_instances, param);

        let a = canvas.finish(ctx);

        a
    }
}

pub fn run() -> GameResult {
    let world = World::<2>::standard_world();

    let resource_dir = if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let mut path = PathBuf::from(manifest_dir);
        path.push("resources");
        path
    } else {
        PathBuf::from("./resources")
    };

    let cb = ggez::ContextBuilder::new("spritebatch", "ggez")
        .add_resource_path(resource_dir)
        .window_mode(WindowMode {
            width: W_FLOAT,
            height: W_FLOAT,

            ..Default::default()
        })
        .window_setup(WindowSetup {
            title: String::from("neuralang"),
            vsync: false,
            samples: NumSamples::One,
            srgb: false,
            ..Default::default()
        });

    let (mut ctx, event_loop) = cb.build()?;

    let state = MainState::new(&mut ctx, world)?;
    event::run(ctx, event_loop, state)
}

// to let it rip without rendering, mainly to gauge overhead of rendering on top of step()
pub fn gauge() {
    let mut w = World::<2>::standard_world();
    let now = SystemTime::now();
    loop {
        w.step(1);
        if w.age % 60 == 0 {
            let duration = match now.elapsed() {
                Ok(now) => now.as_millis(),
                _ => 5 as u128,
            };
            println!(
                "{} {}, fps: {}",
                w.age / 60,
                w.beings_and_models.len(),
                w.age as f32 / ((duration as f32) / 1000.)
            );
        }
    }
}

pub fn main() {
    assert!(W_SIZE % N_CELLS == 0);
    assert!(B_RADIUS < CELL_SIZE as f32);

    // gauge();
    _ = run();
}
