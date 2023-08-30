use ggez::{
    conf::{NumSamples, WindowMode, WindowSetup},
    event,
    glam::*,
    graphics::{Canvas, Color, DrawParam, Image, InstanceArray, Mesh, MeshBuilder, Sampler},
    Context, GameResult,
};
use image::{
    imageops::{resize, FilterType::*},
    GenericImageView, ImageBuffer, Rgba,
};
use rand::{thread_rng, Rng};
use slotmap::{DefaultKey, SlotMap};
use std::{env, f32::consts::PI, path::PathBuf, process::id};

// use anyhow::Result;
// use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Tensor, Kind};

#[rustfmt::skip]
mod consts {

    pub const W_SIZE:                                 usize = 1000;
    pub const N_CELLS:                                usize = 250;
    pub const CELL_SIZE:                              usize = W_SIZE / N_CELLS;
    pub const CELL_SIZE_FLOAT:                          f32 = CELL_SIZE as f32;
    pub const W_FLOAT:                                  f32 = W_SIZE as f32;
    pub const W_USIZE:                                  u32 = W_SIZE as u32;

    pub const N_CLANS:                                usize = 4;

    pub const B_FOV:                                  isize = 5;

    pub const B_SPEED:                                  f32 = 0.1;

    pub const B_RADIUS:                                 f32 = 3.5;
    pub const O_RADIUS:                                 f32 = 3.;
    pub const F_RADIUS:                                 f32 = 2.5;
    pub const S_RADIUS:                                 f32 = 1.5;

    pub const S_GROW_RATE:                              f32 = 1.;

    pub const B_DEATH_ENERGY:                           f32 = 0.5;
    pub const B_SCATTER_RADIUS:                         f32 = 10.;
    pub const B_SCATTER_COUNT:                        usize = 100;

    pub const BASE_ANG_SPEED_DEGREES:                   f32 = 10.;

    pub const B_START_ENERGY:                           f32 = 10.;
    pub const O_START_HEALTH:                           f32 = 25.;
    pub const F_START_AGE:                              f32 = 5.;
    pub const S_START_AGE:                              f32 = 5.;
    pub const F_VAL:                                    f32 = 1.;

    pub const B_TIRE_RATE:                              f32 = 0.001;
    pub const O_AGE_RATE:                               f32 = 0.001;
    pub const F_AGE_RATE:                               f32 = 0.001;
    pub const S_SOFTEN_RATE:                            f32 = 0.1;

    pub const B_HEADON_DAMAGE:                          f32 = 0.25;
    pub const B_REAR_DAMAGE:                            f32 = 1.;
    pub const HEADON_B_HITS_O_DAMAGE:                   f32 = 0.1;
    pub const SPAWN_O_COST:                             f32 = 1.;                  // cost for a being to spawn an obstruct at their mouth

    pub const LOW_ENERGY_SPEED_DAMP_RATE:               f32 = 0.5;                 // beings slow down when their energy runs low
    pub const OFF_DIR_MOVEMENT_SPEED_DAMP_RATE:         f32 = 0.5;                 // beings slow down when not moving face-forward

    pub const N_FOOD_SPAWN_PER_STEP:                  usize = 2; 

    pub const SPEECHLET_LEN:                          usize = 8;                   // length of the sound vector a being can emit
    pub const B_OUTPUT_LEN:                           usize = 5 + SPEECHLET_LEN;   // f-b, l-r, rotate, spawn obstruct, spawn_speechlet
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
pub fn oob(ij: Vec2, r: f32) -> bool {
    let (i, j) = (ij[0], ij[1]);
    lef_border_trespass(i, r)
        || rig_border_trespass(i, r)
        || top_border_trespass(j, r)
        || bot_border_trespass(j, r)
}

pub fn b_collides_b(b1: &Being, b2: &Being) -> (f32, f32, Vec2, Vec<f32>) {
    let c1c2 = b2.pos - b1.pos;
    let centre_dist = c1c2.length();
    let (r1, r2) = (b1.radius, b2.radius);

    let mut rel_vec = b2.clan.clone();
    rel_vec.append(&mut vec![
        b1.pos.angle_between(b2.pos) / PI,
        centre_dist,
        b2.energy / B_START_ENERGY,
    ]);

    (r1 + r2 - centre_dist, centre_dist, c1c2, rel_vec)
}

pub fn b_collides_o(b: &Being, o: &Obstruct) -> (f32, f32, Vec2, Vec<f32>) {
    let c1c2 = o.pos - b.pos;
    let centre_dist = c1c2.length();
    let (r1, r2) = (b.radius, O_RADIUS);

    (
        r1 + r2 - centre_dist,
        centre_dist,
        c1c2,
        vec![0., b.pos.angle_between(o.pos) / PI, o.age / O_START_HEALTH],
    )
}

pub fn b_collides_f(b: &Being, f: &Food) -> (f32, Vec<f32>) {
    let centre_dist = b.pos.distance(f.pos);
    let (r1, r2) = (b.radius, 1.);
    (
        r1 + r2 - centre_dist,
        vec![
            1.,
            centre_dist,
            b.pos.angle_between(f.pos) / PI,
            f.age / F_START_AGE,
        ],
    )
}

pub fn b_collides_s(b: &Being, s: &Speechlet) -> f32 {
    let c1c2 = s.pos - b.pos;
    let centre_dist = c1c2.length();
    let (r1, r2) = (b.radius, S_RADIUS);

    r1 + r2 - centre_dist
}

#[derive(Debug)]
pub struct Being {
    pos: Vec2,
    radius: f32, // to be deprecated
    rotation: f32,
    energy: f32,
    clan: Vec<f32>,

    cell: (usize, usize),
    id: usize,

    pos_update: Vec2,
    energy_update: f32,
    rotation_update: f32,

    being_inputs: Box<Vec<Vec<f32>>>,
    food_obstruct_inputs: Box<Vec<Vec<f32>>>,
    heard_speechlet_inputs: Box<Vec<Vec<f32>>>,

    output: Vec<f32>,
}

pub struct Obstruct {
    pos: Vec2,
    age: f32,
    id: usize,
}

pub struct Food {
    pos: Vec2,
    age: f32,
    val: f32,
    eaten: bool,

    id: usize,
}

#[derive(Debug)]
pub struct Speechlet {
    speechlet: Vec<f32>,
    pos: Vec2,
    radius: f32,
    age: f32,

    recepient_being_ids: Vec<usize>,
}

pub struct World {
    beings: SlotMap<DefaultKey, Being>,
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
}

impl World {
    pub fn new() -> Self {
        World {
            beings: SlotMap::new(),
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
        }
    }

    // a world populated as intended, this fn mainly to relieve World::new() of some clutter
    pub fn standard_world() -> Self {
        let mut world = World::new();
        let mut rng = thread_rng();

        for i in 0..500 {
            world.add_being(
                B_RADIUS,
                Vec2::new(
                    rng.gen_range(B_RADIUS..W_FLOAT - B_RADIUS),
                    rng.gen_range(B_RADIUS..W_FLOAT - B_RADIUS),
                ),
                rng.gen_range(-PI..PI),
                B_START_ENERGY,
                vec![0., 0., 0., 1.],
            );
        }

        for i in 0..1000 {
            world.add_obstruct(Vec2::new(
                rng.gen_range(1.0..W_FLOAT - 1.),
                rng.gen_range(1.0..W_FLOAT - 1.),
            ));
        }

        for i in 0..2000 {
            world.add_food(
                Vec2::new(
                    rng.gen_range(1.0..W_FLOAT - 1.),
                    rng.gen_range(1.0..W_FLOAT - 1.),
                ),
                F_VAL,
            );
        }

        world
    }

    pub fn add_being(
        &mut self,
        radius: f32,
        pos: Vec2,
        rotation: f32,
        health: f32,
        clan: Vec<f32>,
    ) {
        let (i, j) = pos_to_cell(pos);

        let being = Being {
            radius: radius,
            pos: pos,
            rotation: rotation,
            energy: health,
            clan: clan,

            cell: (i, j),
            id: self.being_id,

            pos_update: Vec2::new(0., 0.),
            energy_update: 0.,
            rotation_update: 0.,

            being_inputs: Box::new(vec![]),
            food_obstruct_inputs: Box::new(vec![]),
            heard_speechlet_inputs: Box::new(vec![]),

            output: vec![],
        };

        let k = self.beings.insert(being);
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

    pub fn add_food(&mut self, pos: Vec2, val: f32) {
        let (i, j) = pos_to_cell(pos);

        let food = Food {
            pos: pos,
            age: F_START_AGE,
            val: val,
            eaten: false,

            id: self.food_id,
        };

        let k = self.foods.insert(food);

        let ij = two_to_one((i, j));
        self.food_cells[ij].push(k);
        self.food_id += 1;
    }

    pub fn add_speechlet(&mut self, speechlet: Vec<f32>, pos: Vec2) {
        let (i, j) = pos_to_cell(pos);

        let speechlet = Speechlet {
            speechlet: speechlet,
            pos: pos,
            radius: S_RADIUS,
            age: S_START_AGE,

            recepient_being_ids: vec![]
        };

        let k = self.speechlets.insert(speechlet);
        let ij = two_to_one((i, j));
        self.speechlet_cells[ij].push(k);
    }

    pub fn move_beings(&mut self, substeps: usize) {
        let s = substeps as f32;

        for _ in 0..substeps {
            self.beings.iter_mut().for_each(|(k, being)| {
                let move_vec = dir_from_theta(being.rotation) * (B_SPEED / s); // this part to be redone based on being outputs
                let newij = being.pos + move_vec;

                if !oob(newij, being.radius) {
                    being.pos_update = move_vec;
                } else {
                    being.pos = Vec2::new(
                        thread_rng().gen_range(1.0..W_FLOAT - 1.),
                        thread_rng().gen_range(1.0..W_FLOAT - 1.),
                    );
                    being.energy_update -= HEADON_B_HITS_O_DAMAGE / s / 10.;
                }
            });
        }
    }

    pub fn grow_speechlets(&mut self) {
        self.speechlets.iter_mut().for_each(|(k, s)| {
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
                                        &self.beings.get(*id1).unwrap(),
                                        &self.beings.get(*id2).unwrap(),
                                    );
                                    let b1 = self.beings.get_mut(*id1).unwrap();
                                    b1.being_inputs.push(rel_vec);

                                    if overlap > 0. {
                                        let d_p = overlap / centre_dist * c1c2;
                                        let half_dist = d_p / 1.9;

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

                            for ob_id in &self.obstruct_cells[nij] {
                                // for an obstruct similarly
                                let b = self.beings.get_mut(*id1).unwrap();
                                let o = self.obstructs.get_mut(*ob_id).unwrap();

                                let (overlap, centre_dist, c1c2, rel_vec) = b_collides_o(b, o);
                                b.food_obstruct_inputs.push(rel_vec);

                                if overlap > 0. {
                                    let d_p = overlap / centre_dist * c1c2;
                                    let half_dist = d_p / 1.9;
                                    b.pos_update -= half_dist;

                                    let b_dir = dir_from_theta(b.rotation);
                                    let axis_alignment = b_dir.dot(c1c2.normalize());

                                    if axis_alignment > 0. {
                                        b.energy_update -=
                                            HEADON_B_HITS_O_DAMAGE * axis_alignment / s;
                                    }
                                }
                            }

                            for f_id in &self.food_cells[nij] {
                                // for a food similarly

                                let b = self.beings.get_mut(*id1).unwrap();
                                let f = self.foods.get_mut(*f_id);

                                let f_ref = f.as_ref().unwrap();

                                let (overlap, rel_vec) = b_collides_f(&b, f_ref);
                                b.food_obstruct_inputs.push(rel_vec);

                                if overlap > 0. && !f_ref.eaten && b.energy <= B_START_ENERGY {
                                    b.energy_update += f_ref.val;
                                    self.food_deaths.push((*f_id, f_ref.pos));
                                    f.unwrap().eaten = true;
                                }
                            }

                            for s_id in &self.speechlet_cells[nij] {
                                let b = self.beings.get_mut(*id1).unwrap();
                                let s = self.speechlets.get_mut(*s_id).unwrap();

                                let overlap = b_collides_s(&b, &s);

                                if overlap > 0. && !s.recepient_being_ids.contains(&b.id) {
                                    b.heard_speechlet_inputs.push(s.speechlet.clone());
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
        for (k, b) in &mut self.beings {
            let new_pos = b.pos + b.pos_update;

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
        for (k, b) in &mut self.beings {
            b.energy -= B_TIRE_RATE;

            if b.energy <= 0. {
                self.being_deaths.push((k, b.pos));
            }
        }

        let mut rng = thread_rng();
        for (k, pos) in &self.being_deaths.clone() {
            self.beings.remove(*k);
            self.being_cells[two_to_one(pos_to_cell(*pos))].retain(|x| x != k);

            for _ in 0..B_SCATTER_COUNT {
                let (theta, dist) = (rng.gen_range(-PI..PI), rng.gen_range(0.0..B_SCATTER_RADIUS));
                let dvec = Vec2::new(theta.cos() * dist, theta.sin() * dist);

                let food_pos = *pos + dvec;
                if !oob(food_pos, F_RADIUS) {
                    self.add_food(food_pos, B_DEATH_ENERGY / B_SCATTER_RADIUS as f32);
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
            f.age -= F_AGE_RATE;
            if f.age < 0.05 {
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
        for _ in 0..N_FOOD_SPAWN_PER_STEP {
            let ij = Vec2::new(rng.gen_range(1.0..W_FLOAT), rng.gen_range(1.0..W_FLOAT));
            self.add_food(ij, F_VAL);
        }
    }

    pub fn perform_being_outputs(&mut self) {
        self.beings.iter().for_each(|(k, b)| {
            let output = b.output.clone();
        });
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

        self.age += 1;
    }
}

struct MainState {
    being_instances: InstanceArray,
    obstruct_instances: InstanceArray,
    food_instances: InstanceArray,
    speechlet_instances: InstanceArray,
    world: World,
}

impl MainState {
    fn new(ctx: &mut Context, w: World) -> GameResult<MainState> {
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

// pub fn get_fovs(
//     frame: Vec<u8>,
//     beings: &SlotMap<DefaultKey, Being>,
// ) -> Vec<ImageBuffer<Rgba<u8>, Vec<u8>>> {
//     let frame = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(W_USIZE, W_USIZE, frame).expect("");

//     beings
//         .iter()
//         .map(|(_, b)| {
//             let xy = b.pos;
//             let (x, y) = (xy[0] as u32, xy[1] as u32);

//             let a = frame
//                 .view(x - B_FOV as u32, y - B_FOV, 2 * B_FOV - 1, 2 * B_FOV + 1)
//                 .to_image()
//                 .clone();

//             resize(&a, 13, 13, Gaussian)
//         })
//         .collect()
// }

impl event::EventHandler<ggez::GameError> for MainState {
    fn update(&mut self, ctx: &mut Context) -> Result<(), ggez::GameError> {

        if self.world.age % 60 == 0 {
            // let frame = ctx.gfx.frame().to_pixels(&ctx.gfx).unwrap();
            // let fovs = get_fovs(frame, &self.world.beings);

            // forward pass on each being
            // update being actions
            println!("timestep: {}, fps: {}", self.world.age, ctx.time.fps(),);
            self.world.add_speechlet(vec![], Vec2 { x: thread_rng().gen_range(0.0..W_FLOAT), y: thread_rng().gen_range(0.0..W_FLOAT) });
        }

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
            .set(self.world.beings.iter().map(|(_, b)| {
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
    let world = World::standard_world();

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

// to let it rip without rendering, mainly to gauge overhead of rendering over step() itself

pub fn gauge() {
    let mut w = World::standard_world();
    // println!("{:?}", w.fov_indices.len());
    loop {
        w.step(1);

        if w.age % 60 == 0 {
            println!("{} {}", w.age / 60, w.beings.len());
            w.add_speechlet(vec![], Vec2 { x: thread_rng().gen_range(0.0..W_FLOAT), y: thread_rng().gen_range(0.0..W_FLOAT) });
        }
    }
}

pub fn main() {
    assert!(W_SIZE % N_CELLS == 0);
    assert!(B_RADIUS < CELL_SIZE as f32);

    // gauge();
    run();
}
