use ggez::conf::WindowMode;
use ggez::event;
use ggez::glam::*;
use ggez::graphics::{self, Color};
use ggez::{Context, GameResult};
use rand::{distributions::Uniform, prelude::*};
use slotmap::DefaultKey;
use slotmap::SlotMap;
use std::env;
use std::f32::consts::PI;
use std::path;

#[rustfmt::skip]
mod consts {
    pub const W_SIZE: usize = 1000;
    pub const N_CELLS: usize = 250;
    pub const CELL_SIZE: usize = W_SIZE / N_CELLS;
    pub const W_FLOAT: f32 = W_SIZE as f32;
    pub const HZ: usize = 60;

    pub const B_RADIUS:                                 f32 = 3.;
    pub const O_RADIUS:                                 f32 = 2.;
    pub const F_RADIUS:                                 f32 = 2.;

    pub const BASE_MOV_SPEED:                           f32 = 1.;
    pub const BASE_ANG_SPEED_DEGREES:                   f32 = 10.;

    pub const B_START_ENERGY:                           f32 = 10.;
    pub const O_START_HEALTH:                           f32 = 5.;
    pub const F_START_VAL:                              f32 = 2.;

    pub const B_ENERGY_LEAK:                            f32 = 0.005;
    pub const O_HEALTH_LEAK:                            f32 = 0.002;
    pub const F_VAL_LEAK:                               f32 = 0.002;

    pub const B_HEADON_DAMAGE:                          f32 = 0.25;
    pub const B_REAR_DAMAGE:                            f32 = 1.;
    pub const TAN_B_HITS_O_DAMAGE:                      f32 = 1.;
    pub const HEADON_B_HITS_O_DAMAGE:                   f32 = 0.1;
    pub const SPAWN_O_COST:                             f32 = 1.;                           // cost for a being to spawn an obstruct at their rear
    pub const LOW_ENERGY_SPEED_DAMP_RATE:               f32 = 0.5;                          // beings slow down when their energy runs low
    pub const OFF_DIR_MOVEMENT_SPEED_DAMP_RATE:         f32 = 0.5;                          // beings slow down when not moving face-forward

    pub const N_FOOD_SPAWN_PER_STEP:                    usize = 1; 

    pub const LANG_VEC_LEN:                           usize = 32;                           // length of the sound vector a being can emit
    pub const B_OUTPUT_LEN:                           usize = LANG_VEC_LEN + 5;             // move_forward, rotate_left, rotate_right, spawn_obstruct, speak
}

use consts::*;

// maps 2D space-partition index to 1D Vec index
fn two_to_one((i, j): (usize, usize)) -> usize {
    i * N_CELLS + j
}

fn dir_from_theta(theta: f32) -> Vec2 {
    Vec2::new(theta.cos(), theta.sin())
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

pub fn b_collides_b(b1: &Being, b2: &Being) -> (f32, f32, Vec2) {
    let c1c2 = b2.pos - b1.pos;
    let centre_dist = c1c2.length();
    let (r1, r2) = (b1.radius, b2.radius);

    (r1 + r2 - centre_dist, centre_dist, c1c2)
}

pub fn b_collides_o(b: &Being, o: &Obstruct) -> (f32, f32, Vec2) {
    let c1c2 = o.pos - b.pos;
    let centre_dist = c1c2.length();
    let (r1, r2) = (b.radius, o.radius);
    (r1 + r2 - centre_dist, centre_dist, c1c2)
}

pub fn b_collides_f(b: &Being, f: &Food) -> bool {
    let centre_dist = b.pos.distance(f.pos);
    let (r1, r2) = (b.radius, 1.);
    r1 + r2 - centre_dist > 0.
}

#[derive(Debug, Clone)]
pub struct Being {
    radius: f32,
    pos: Vec2,
    rotation: f32,
    energy: f32,

    speed: f32,
    cell: (usize, usize),
    id: usize,

    pos_update: Vec2,
    energy_update: f32,
    rotation_update: f32,
}

pub struct Obstruct {
    radius: f32,
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

pub struct World {
    beings: SlotMap<DefaultKey, Being>,
    obstructs: SlotMap<DefaultKey, Obstruct>,
    foods: SlotMap<DefaultKey, Food>,

    being_cells: Vec<Vec<DefaultKey>>,
    obstruct_cells: Vec<Vec<DefaultKey>>,
    food_cells: Vec<Vec<DefaultKey>>,

    being_id: usize,
    ob_id: usize,
    food_id: usize,

    being_deaths: Vec<(DefaultKey, Vec2)>,
    obstruct_deaths: Vec<(DefaultKey, Vec2)>,
    food_deaths: Vec<(DefaultKey, Vec2)>,

    age: usize,

    being_collision_count: usize,
    obstruct_collision_count: usize,
    food_collision_count: usize,
}

impl World {
    pub fn new(n_cells: usize) -> Self {
        assert!(W_SIZE % n_cells == 0);
        World {
            beings: SlotMap::new(),
            obstructs: SlotMap::new(),
            foods: SlotMap::new(),

            being_cells: (0..(n_cells + 1).pow(2)).map(|_| Vec::new()).collect(),
            obstruct_cells: (0..(n_cells + 1).pow(2)).map(|_| Vec::new()).collect(),
            food_cells: (0..(n_cells + 1).pow(2)).map(|_| Vec::new()).collect(),

            being_id: 0,
            ob_id: 0,
            food_id: 0,

            being_collision_count: 0,
            obstruct_collision_count: 0,
            food_collision_count: 0,

            being_deaths: vec![],
            food_deaths: vec![],
            obstruct_deaths: vec![],

            age: 0,
        }
    }

    pub fn add_being(&mut self, radius: f32, pos: Vec2, rotation: f32, speed: f32, health: f32) {
        let (i, j) = pos_to_cell(pos);

        let being = Being {
            radius: radius,
            pos: pos,
            rotation: rotation,

            energy: health,
            speed: speed,
            cell: (i, j),
            id: self.being_id,

            pos_update: Vec2::new(0., 0.),
            energy_update: 0.,
            rotation_update: 0.,
        };

        let k = self.beings.insert(being);
        let ij = two_to_one((i, j));
        self.being_cells[ij].push(k);
        self.being_id += 1;
    }

    pub fn add_obstruct(&mut self, pos: Vec2) {
        let (i, j) = pos_to_cell(pos);

        let obstruct = Obstruct {
            radius: 2.,
            pos: pos,
            age: 5.,
            id: self.ob_id,
        };

        let k = self.obstructs.insert(obstruct);

        let ij = two_to_one((i, j));
        self.obstruct_cells[ij].push(k);
        self.ob_id += 1;
    }

    pub fn add_food(&mut self, pos: Vec2) {
        let (i, j) = pos_to_cell(pos);

        let food = Food {
            pos: pos,
            age: 5.,
            val: 1.,
            eaten: false,

            id: self.food_id,
        };

        let k = self.foods.insert(food);

        let ij = two_to_one((i, j));
        self.food_cells[ij].push(k);
        self.food_id += 1;
    }

    pub fn move_beings(&mut self, substeps: usize) {
        let s = substeps as f32;

        let rdist = Uniform::new(1., (W_SIZE as f32) - 1.);
        let mut rng = thread_rng();

        for _ in 0..substeps {
            let w = W_SIZE as f32;
            self.beings.iter_mut().for_each(|(k, being)| {
                let move_vec = dir_from_theta(being.rotation) * (being.speed / s); // this part to be redone based on being outputs
                let newij = being.pos + move_vec;

                if !oob(newij, being.radius) {
                    being.pos_update = move_vec;
                } else {
                    // TEMP TEMP TEMP TEMP NOTICE TEMP TO BE FIXED
                    let newij = Vec2::new(rng.sample(rdist), rng.sample(rdist));
                    being.pos = newij;
                }
            });
        }
    }

    pub fn check_collisions(&mut self, timestep: usize) {
        let w = N_CELLS as isize;

        for i in 0..N_CELLS {
            for j in 0..N_CELLS {
                // for each partition
                let ij = two_to_one((i, j));

                for id1 in &self.being_cells[ij] {
                    // for each being
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

                        if !(ni < 0 || ni >= w || nj < 0 || nj >= w) {
                            // if valid partition
                            let (ni, nj) = (ni as usize, nj as usize);
                            let nij = two_to_one((ni, nj));

                            for id2 in &self.being_cells[nij] {
                                // for another being in the same or one of the 8 neighbouring cells
                                if !(id1 == id2) {
                                    let (overlap, centre_dist, c1c2) = b_collides_b(
                                        &self.beings.get(*id1).unwrap(),
                                        self.beings.get(*id2).unwrap(),
                                    );

                                    if overlap > 0. {
                                        let b1 = self.beings.get_mut(*id1).unwrap();

                                        let d_p = overlap / centre_dist * c1c2;
                                        let half_dist = 0.5 * d_p;

                                        let new_pos = b1.pos - half_dist;
                                        if !oob(new_pos, b1.radius) {
                                            b1.pos_update -= half_dist;
                                        }

                                        self.being_collision_count += 1;
                                    }
                                }
                            }

                            for ob_id in &self.obstruct_cells[nij] {
                                // for an obstruct similarly
                                let b = self.beings.get_mut(*id1);
                                let o = self.obstructs.get_mut(*ob_id);

                                let b_ref = b.as_ref().unwrap();
                                let o_ref = o.as_ref().unwrap();

                                let (overlap, centre_dist, c1c2) = b_collides_o(b_ref, o_ref);
                                if overlap > 0. {
                                    let d_p = overlap / centre_dist * c1c2;
                                    let half_dist = d_p / 2.;

                                    b.unwrap().pos_update -= half_dist;
                                    self.obstruct_collision_count += 1;
                                }
                            }

                            for f_id in &self.food_cells[nij] {
                                // for a food similarly

                                let b = self.beings.get_mut(*id1);
                                let f = self.foods.get_mut(*f_id);

                                let b_ref = b.as_ref().unwrap();
                                let f_ref = f.as_ref().unwrap();

                                let overlap = b_collides_f(b_ref, f_ref);

                                if overlap && !f_ref.eaten {
                                    b.unwrap().energy_update += f_ref.val;
                                    self.food_deaths.push((*f_id, f_ref.pos));
                                    f.unwrap().eaten = true;

                                    self.food_collision_count += 1;
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
                b.pos_update = Vec2::new(0., 0.);

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
    pub fn tire_beings(&mut self, tire_rate: f32) {
        for (k, b) in &mut self.beings {
            b.energy -= tire_rate;

            if b.energy <= 0. {
                self.being_deaths.push((k, b.pos));
            }
        }

        for b in &self.being_deaths {
            self.beings.remove(b.0);
            self.being_cells[two_to_one(pos_to_cell(b.1))].retain(|x| *x != b.0);
        }

        self.being_deaths.clear();
    }

    // food grows stale and/or disappears
    pub fn age_foods(&mut self, age_rate: f32) {
        for (k, f) in &mut self.foods {
            f.age -= age_rate;
            if f.age < 0.05 {
                self.food_deaths.push((k, f.pos));
            }
        }

        for f in &self.food_deaths {
            self.foods.remove(f.0);

            self.food_cells[two_to_one(pos_to_cell(f.1))].retain(|x| *x != f.0);
        }

        self.food_deaths.clear();
    }

    // walls crack and/or crumble
    pub fn age_obstructs(&mut self, age_rate: f32) {
        for (k, o) in &mut self.obstructs {
            o.age -= age_rate;

            if o.age < 0.05 {
                self.obstruct_deaths.push((k, o.pos));
            }
        }

        for o in &self.obstruct_deaths {
            self.obstructs.remove(o.0);
            self.obstruct_cells[two_to_one(pos_to_cell(o.1))].retain(|x| *x != o.0);
        }

        self.obstruct_deaths.clear();
    }

    // self-explanatory
    pub fn step(&mut self, substeps: usize) {
        for _ in 0..substeps {
            self.move_beings(substeps);
            self.check_collisions(self.age);
            self.update_cells();
        }

        self.tire_beings(0.);
        self.age_foods(0.);
        self.age_obstructs(0.);

        self.age += 1;
    }
}

// a BUNCH of rendering boilerplate, will switch to Bevy rendering soon. wip.
struct MainState {
    being_instances: graphics::InstanceArray,
    world: World,
}

impl MainState {
    fn new(ctx: &mut Context, w: World) -> GameResult<MainState> {
        let image = graphics::Image::from_path(ctx, "/circle.png")?;
        let mut instances = graphics::InstanceArray::new(ctx, image);

        Ok(MainState {
            being_instances: instances,
            world: w,
        })
    }
}

impl event::EventHandler<ggez::GameError> for MainState {
    fn update(&mut self, ctx: &mut Context) -> Result<(), ggez::GameError> {
        self.world.step(8);
        if self.world.age % HZ == 0 {
            println!("timestep: {}, fps: {}", self.world.age, ctx.time.fps());
        }

        Ok(())
    }

    fn draw(&mut self, _ctx: &mut Context) -> Result<(), ggez::GameError> {
        let mut canvas = graphics::Canvas::from_frame(_ctx, Color::BLACK);

        self.being_instances
            .set(self.world.beings.iter().map(|(k, b)| {
                let xy = b.pos;
                graphics::DrawParam::new()
                    .dest(xy.clone())
                    .scale(Vec2::new(1., 1.) / 400. * B_RADIUS)
                    .rotation(b.rotation)
            }));

        let param = graphics::DrawParam::new();
        canvas.draw(&self.being_instances, param);
        canvas.finish(_ctx)
    }
}

// a world populated as intended, this fn mainly to relieve World::new() of some clutter
pub fn get_world() -> World {
    let mut world = World::new(N_CELLS);
    let rdist = Uniform::new(1., (W_SIZE as f32) - 1.);
    let mut rng = thread_rng();

    for i in 1..5000 {
        world.add_being(
            B_RADIUS,
            Vec2::new(rng.sample(rdist), rng.sample(rdist)),
            rng.gen_range(-PI..PI),
            2.,
            1.,
        );
    }

    // for i in 1..5000 {
    //     world.add_obstruct(Vec2::new(rng.sample(rdist), rng.sample(rdist)));
    // }

    // for i in 1..2000 {
    //     world.add_food(Vec2::new(rng.sample(rdist), rng.sample(rdist)))
    // }

    world
}

pub fn run() -> GameResult {
    assert!(W_SIZE % N_CELLS == 0);
    assert!(B_RADIUS < CELL_SIZE as f32);

    let world = get_world();

    // if cfg!(debug_assertions) && env::var("yes_i_really_want_debug_mode").is_err() {
    //     eprintln!(
    //         "Note: Release mode will improve performance greatly.\n    \
    //          e.g. use `cargo run --example spritebatch --release`"
    //     );
    // }

    let resource_dir = if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let mut path = path::PathBuf::from(manifest_dir);
        path.push("resources");
        path
    } else {
        path::PathBuf::from("./resources")
    };

    let cb = ggez::ContextBuilder::new("spritebatch", "ggez")
        .add_resource_path(resource_dir)
        .window_mode(WindowMode {
            width: W_FLOAT,
            height: W_FLOAT,

            ..Default::default()
        });

    let (mut ctx, event_loop) = cb.build()?;

    let state = MainState::new(&mut ctx, world)?;
    event::run(ctx, event_loop, state)
}

// to let it rip without rendering, mainly to gauge overhead of rendering over step() itself
pub fn gauge(n: usize) {
    let mut w = get_world();
    for i in 0..n {
        w.step(1);
        if i % HZ == 0 {
            println!("{}", i / HZ);
        }
    }
}

pub fn main() {
    // gauge(1000000);
    run();
}
