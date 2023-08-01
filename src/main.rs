use ggez::conf::WindowMode;
use ggez::event;
use ggez::glam::*;
use ggez::graphics::{self, Color};
use ggez::winit::window::WindowBuilder;
use ggez::{Context, GameResult};
use rand::{distributions::Uniform, prelude::*};
use rayon::prelude::*;
use splitmut::SplitMut;
use std::env;
use std::f32::consts::TAU;
use std::path;

const W_SIZE: usize = 1000;
const N_CELLS: usize = 250;
const CELL_SIZE: usize = W_SIZE / N_CELLS;
const W_FLOAT: f32 = W_SIZE as f32;
const HZ: usize = 60;

fn add_2d((i, j): (f32, f32), (k, l): (f32, f32)) -> (f32, f32) {
    (i + k, j + l)
}

fn scale_2d((i, j): (f32, f32), c: f32) -> (f32, f32) {
    (i * c, j * c)
}

fn dist_2d((i1, j1): (f32, f32), (i2, j2): (f32, f32)) -> f32 {
    ((i1 - i2).powi(2) + (j1 - j2).powi(2)).sqrt()
}

fn one_to_two(ij: usize) -> (usize, usize) {
    ((ij - ij % W_SIZE) / W_SIZE, ij % W_SIZE)
}

fn two_to_one((i, j): (usize, usize)) -> usize {
    i * N_CELLS + j
}

fn dir_from_theta(theta: f32) -> (f32, f32) {
    (theta.cos(), theta.sin())
}

fn same_index((a, b): (usize, usize), (c, d): (usize, usize)) -> bool {
    a == c && b == d
}

pub fn pos_to_cell(pos: (f32, f32)) -> (usize, usize) {
    let c = CELL_SIZE as f32;
    let i = ((pos.0 - (pos.0 % c)) / c) as usize;
    let j = ((pos.1 - (pos.1 % c)) / c) as usize;

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

pub fn oob((i, j): (f32, f32), r: f32) -> bool {
    lef_border_trespass(i, r)
        || rig_border_trespass(i, r)
        || top_border_trespass(j, r)
        || bot_border_trespass(j, r)
}

pub fn beings_collide(b1: &Being, b2: &Being) -> bool {
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

#[derive(Debug)]
pub struct Being {
    radius: f32,
    pos: (f32, f32),
    rotation: f32,
    energy: f32,

    speed: f32,
    cell: (usize, usize),
    id: usize,

    pos_update: (f32, f32),
}

pub struct Obstruct {
    radius: f32,
    pos: (f32, f32),
    age: f32,
    id: usize,
}

pub struct Food {
    pos: (f32, f32),
    age: f32,
    val: f32,
    eaten: bool,

    id: usize,
}

struct World {
    beings: Vec<Being>,
    obstructs: Vec<Obstruct>,
    foods: Vec<Food>,

    being_cells: Vec<Vec<usize>>,
    obstruct_cells: Vec<Vec<usize>>,
    food_cells: Vec<Vec<usize>>,

    being_id: usize,
    ob_id: usize,
    food_id: usize,

    being_collision_count: usize,
    obstruct_collision_count: usize,
    food_collision_count: usize,

    being_deaths: Vec<(usize, (f32, f32))>,
    obstruct_deaths: Vec<(usize, (f32, f32))>,
    food_deaths: Vec<(usize, (f32, f32))>,

    age: usize,
}

impl World {
    pub fn new(n_cells: usize) -> Self {
        assert!(W_SIZE % n_cells == 0);
        World {
            beings: vec![],
            obstructs: vec![],
            foods: vec![],

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

    pub fn add_being(&mut self, radius: f32, pos: (f32, f32), rotation: f32, speed: f32, health: f32) {
        let (i, j) = pos_to_cell(pos);
        self.beings.push(Being {
            radius: radius,
            pos: pos,
            rotation: rotation,

            energy: health,
            speed: speed,
            cell: (i, j),
            id: self.being_id,

            pos_update: (0., 0.),
        });
        let ij = two_to_one((i, j));
        self.being_cells[ij].push(self.being_id);
        self.being_id += 1;
    }

    pub fn add_obstruct(&mut self, pos: (f32, f32)) {
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

    pub fn add_food(&mut self, pos: (f32, f32)) {
        let (i, j) = pos_to_cell(pos);
        self.foods.push(Food {
            pos: pos,
            age: 5.,
            val: 1.,
            eaten: false,

            id: self.food_id,

        });

        let ij = two_to_one((i, j));
        self.food_cells[ij].push(self.food_id);
        self.food_id += 1;
    }
    pub fn move_beings(&mut self, substeps: usize) {
        let s = substeps as f32;

        let rdist = Uniform::new(1., (W_SIZE as f32) - 1.);
        let mut rng = thread_rng();


        // REMEMBER TO POS_UPDATE HERE INSTEAD OF ASYNC UPDATE (CURRENT)
        for _ in 0..substeps {
            let w = W_SIZE as f32;
            self.beings.iter_mut().for_each(|being| {
                let move_vec = scale_2d(dir_from_theta(being.rotation), being.speed / s);
                let (newi, newj) = add_2d(being.pos, move_vec);

                let r = being.radius;

                if !oob((newi, newj), r) {
                    being.pos_update = move_vec;
                }
                // TEMP TEMP TEMP TEMP NOTICE TEMP
                else {
                    let (newi, newj) = (rng.sample(rdist), rng.sample(rdist));
                    being.pos = (newi, newj);
                }
            });
        }
    }

    pub fn check_collisions(&mut self, timestep: usize) {
        let w = N_CELLS as isize;

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
                        if !(ni < 0 || ni >= w || nj < 0 || nj >= w) {
                            let (ni, nj) = (ni as usize, nj as usize);
                            let nij = two_to_one((ni, nj));

                            for id2 in &self.being_cells[nij] {
                                if !(*id1 == *id2) {
                                    let (b1, b2) = self.beings.get2_mut(*id1, *id2);

                                    let b1_ref = b1.as_ref().unwrap();
                                    let b2_ref = b2.as_ref().unwrap();

                                    if beings_collide(b1_ref, b2_ref) {
                                        self.being_collision_count += 1;

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
                                let b = self.beings.get_mut(*id1); // see this line here happens 3 times for some reason because the bc won't allow non overlapping borrows from the same vec even if i use a splitmut method the second time but i was assured the compiler compiles this away so who knows. for later.
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
                                let b = self.beings.get_mut(*id1);
                                let f = self.foods.get_mut(*f_id);

                                let b_ref = b.as_ref().unwrap();
                                let f_ref = f.as_ref().unwrap();

                                if food_collide(b_ref, f_ref) && !f_ref.eaten {
                                    b.unwrap().energy += f_ref.val;
                                    self.food_deaths.push((f_ref.id, f_ref.pos));
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

    pub fn update_cells(&mut self) {
        for b in &mut self.beings {
            let new_pos = add_2d(b.pos, b.pos_update);

            if !oob(new_pos, b.radius) {
                b.pos = new_pos;
                b.pos_update = (0., 0.);

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

    pub fn tire_beings (&mut self, tire_rate: f32) {
        for b in &mut self.beings {
            b.energy -= tire_rate;

            if b.energy <= 0. {
                self.being_deaths.push((b.id, b.pos));
            }
        }

        for b in &self.being_deaths {
            self.beings.retain(|x| x.id != b.0);
            self.being_cells[two_to_one(pos_to_cell(b.1))].retain(|x| *x != b.0);
        }

        self.being_deaths.clear();
    }

    pub fn age_foods(&mut self, decay_rate: f32) {
        for f in &mut self.foods {
            f.age *= decay_rate;
            if f.age < 0.05 {
                self.food_deaths.push((f.id, f.pos));

            }
        }

        for f in &self.food_deaths {
            self.foods.retain(|x| x.id != f.0);
            self.food_cells[two_to_one(pos_to_cell(f.1))].retain(|x| *x != f.0);
        }

        self.food_deaths.clear();
    }

    pub fn age_obstructs (&mut self, decay_rate: f32) {
        for o in &mut self.obstructs {
            o.age *= decay_rate;

            if o.age < 0.05 {
                self.obstruct_deaths.push((o.id, o.pos));
            }
        }

        for o in &self.obstruct_deaths {
            self.obstructs.retain(|x| x.id != o.0);
            self.obstruct_cells[two_to_one(pos_to_cell(o.1))].retain(|x| *x != o.0);
        }

        self.obstruct_deaths.clear();
    }


    pub fn step(&mut self, substeps: usize) {
        for _ in 0..substeps {
            self.move_beings(substeps);
            self.check_collisions(self.age);
            self.update_cells();

        }

        self.tire_beings(0.001);
        self.age_foods(0.999);
        self.age_obstructs(0.999);
        
        self.age += 1;
    }
}

struct MainState {
    being_instances: graphics::InstanceArray,
    world: World,
}

impl MainState {
    fn new(ctx: &mut Context, w: World) -> GameResult<MainState> {
        let image = graphics::Image::from_path(ctx, "/pacman.jpeg")?;
        let mut instances = graphics::InstanceArray::new(ctx, image);
        instances.resize(ctx, 100);
        Ok(MainState {
            being_instances: instances,
            world: w,
        })
    }
}

impl event::EventHandler<ggez::GameError> for MainState {
    fn update(&mut self, ctx: &mut Context) -> Result<(), ggez::GameError> {
        self.world.step(1);
        if self.world.age % HZ == 0 {
            println!("{} {} {}", self.world.age, ctx.time.fps(), self.world.obstructs.len());
        }

        Ok(())
    }

    fn draw(&mut self, _ctx: &mut Context) -> Result<(), ggez::GameError> {
        let mut canvas = graphics::Canvas::from_frame(_ctx, Color::BLACK);

        self.being_instances.set(self.world.beings.iter().map(|b| {
            let (x, y) = b.pos;
            graphics::DrawParam::new()
                .dest(Vec2::new(x, y))
                .scale(Vec2::new(0.01, 0.01))
                .rotation(b.rotation)
        }));

        let param = graphics::DrawParam::new();
        canvas.draw(&self.being_instances, param);
        canvas.finish(_ctx)
    }
}

pub fn run() -> GameResult {
    assert!(W_SIZE % N_CELLS == 0);
    let mut world = World::new(N_CELLS);
    let rdist = Uniform::new(1., (W_SIZE as f32) - 1.);
    let mut rng = thread_rng();

    for i in 1..50000 {
        world.add_being(
            2.,
            (rng.sample(rdist), rng.sample(rdist)),
            rng.gen_range(-10.0..10.),

            0.1,
            5.
        );
    }

    // for i in 1..5000 {
    //     world.add_obstruct((rng.sample(rdist), rng.sample(rdist)));
    // }

    // for i in 1..2000 {
    //     world.add_food((rng.sample(rdist), rng.sample(rdist)))
    // }

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

pub fn main() {
    let _ = run();
}
