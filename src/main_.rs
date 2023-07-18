// use dashmap::{DashMap, DashSet};
// use piston_window::*;
// use rand::{distributions::Uniform, prelude::*};
// use rayon::prelude::*;
// use std::{
//     thread,
//     borrow::BorrowMut,
//     sync::{Arc, Mutex},
//     time::*,
// };

// const W_SIZE: usize = 720;

// #[derive(Debug, PartialEq)]
// struct Being {
//     id: u32,

//     pos: (f64, f64),
//     rotation: f64,

//     health: f64,
//     hunger: f64,
// }

// #[derive(Debug)]
// struct Food {
//     id: u32,
//     pos: (f64, f64),
//     val: f64,
// }

// #[derive(Debug)]
// struct Chunk {
//     pos: (u32, u32),
//     being_keys: Vec<u32>,
//     food_keys: Vec<u32>,
// }

// #[derive(Debug)]
// struct World {
//     chunk_size: f64,
//     n_chunks: u32,
//     worldsize: f64,

//     chunks: Vec<Vec<Arc<Mutex<Chunk>>>>,

//     being_keys: Vec<u32>,
//     beings: DashMap<u32, Being>,
//     foods: DashMap<u32, Food>,

//     being_speed: f64,
//     being_radius: f64,

//     beingkey: u32,
//     foodkey: u32,

//     repr: Vec<u32>,
// }

// fn normalize_2d((i, j): (f64, f64)) -> (f64, f64) {
//     let norm = (i.powi(2) + j.powi(2)).sqrt();

//     (i / norm, j / norm)
// }

// fn add_2d((i, j): (f64, f64), (k, l): (f64, f64)) -> (f64, f64) {
//     (i + k, j + l)
// }

// fn scale_2d((i, j): (f64, f64), c: f64) -> (f64, f64) {
//     (i * c, j * c)
// }

// fn dist_2d((i1, j1): (f64, f64), (i2, j2): (f64, f64)) -> f64 {
//     ((i1 - i2).powi(2) + (j1 - j2).powi(2)).sqrt()
// }

// fn one_to_two(ij: usize) -> (usize, usize) {
//     ((ij - ij % W_SIZE) / W_SIZE, ij % W_SIZE)
// }

// fn two_to_one((i, j): (usize, usize)) -> usize {
//     i * W_SIZE + j
// }

// fn dir_from_theta(theta: f64) -> (f64, f64) {
//     (theta.cos(), theta.sin())
// }

// impl World {
//     pub fn new(chunk_size: f64, n_chunks: u32) -> Self {
//         World {
//             chunk_size: chunk_size,
//             n_chunks: n_chunks,
//             worldsize: chunk_size * (n_chunks as f64),

//             chunks: (0..n_chunks)
//                 .into_par_iter()
//                 .map(|i| {
//                     (0..n_chunks)
//                         .into_iter()
//                         .map(|j| {
//                             Arc::new(Mutex::new(Chunk {
//                                 pos: (i, j),
//                                 being_keys: vec![],
//                                 food_keys: vec![],
//                             }))
//                         })
//                         .collect()
//                 })
//                 .collect(),
            
//             being_keys: Vec::new(),
//             beings: DashMap::new(),
//             foods: DashMap::new(),

//             being_speed: 0.5,
//             being_radius: (chunk_size / 2.) * 0.999,

//             beingkey: 0,
//             foodkey: 0,

//             repr: vec![],
//         }
//     }

//     fn pos_to_chunk(&self, pos: (f64, f64)) -> (usize, usize) {
//         let i = ((pos.0 - (pos.0 % self.chunk_size)) / self.chunk_size) as usize;
//         let j = ((pos.1 - (pos.1 % self.chunk_size)) / self.chunk_size) as usize;

//         (i, j)
//     }

//     pub fn add_food(&mut self, pos: (f64, f64), val: f64, age: f64) {
//         self.foods.insert(
//             self.foodkey,
//             Food {
//                 id: self.foodkey,
//                 pos: pos,
//                 val: val,
//             },
//         );

//         let (i, j) = self.pos_to_chunk(pos);
//         self.chunks[i][j]
//             .lock()
//             .unwrap()
//             .food_keys
//             .push(self.foodkey);

//         self.foodkey += 1;
//     }

//     pub fn add_being(&mut self, pos: (f64, f64), rotation: f64, health: f64) {
        
//         self.being_keys.push(self.beingkey);
//         self.beings.insert(
//             self.beingkey,
//             Being {
//                 id: self.beingkey,
//                 pos: pos,
//                 rotation: rotation,
//                 health: 10.,
//                 hunger: 0.,
//             },
//         );

//         let (i, j) = self.pos_to_chunk(pos);
//         self.chunks[i][j]
//             .lock()
//             .unwrap()
//             .being_keys
//             .push(self.beingkey);

//         self.beingkey += 1;
//     }

//     pub fn decay_food(mut self) {
//         self.foods.par_iter_mut().for_each(|mut entry| {
//             entry.value_mut().val *= 0.9;
//         });

//         self.foods.retain(|_, food| food.val > 0.05);
//     }

//     pub fn move_beings(&mut self) {
//         let mut rdist = Uniform::new(-0.1, 0.1);

//         self.beings
//             .par_iter_mut()
//             .for_each_init(thread_rng, |rng, mut entry| {
//                 let being = entry.value_mut();
//                 let mut direction = (being.rotation.cos(), being.rotation.sin());
//                 let fatigue_speed = (10. - being.hunger) / 10. * self.being_speed;

//                 let curr_pos = being.pos.clone();
//                 let new_pos = add_2d(curr_pos, scale_2d(direction, fatigue_speed));

//                 let ver_border_tresspass = new_pos.1 - self.being_radius < 1.
//                     || new_pos.1 + self.being_radius >= self.worldsize - 1.;
//                 let hor_border_tresspass = new_pos.0 - self.being_radius < 1.
//                     || new_pos.0 + self.being_radius >= self.worldsize - 1.;

//                 if !(ver_border_tresspass || hor_border_tresspass) {
//                     being.pos = new_pos;
//                     let curr_chunk = self.pos_to_chunk(curr_pos);
//                     let new_chunk = self.pos_to_chunk(new_pos);

//                     if !(curr_chunk == new_chunk) {
//                         self.chunks[curr_chunk.0][curr_chunk.1]
//                             .lock()
//                             .unwrap()
//                             .being_keys
//                             .retain(|x| x != &being.id);
//                         self.chunks[new_chunk.0][new_chunk.1]
//                             .lock()
//                             .unwrap()
//                             .being_keys
//                             .push(being.id);
//                     }
//                 } else if ver_border_tresspass {
//                     being.rotation *= -1.;
//                     let new_direction = (being.rotation.cos(), being.rotation.sin());
//                     let new_pos = add_2d(new_pos, scale_2d(new_direction, fatigue_speed));
//                     being.pos = new_pos;
//                 } else if hor_border_tresspass {
//                     being.rotation *= -1.;
//                     being.rotation += 3.14;
//                     let new_direction = (being.rotation.cos(), being.rotation.sin());
//                     let new_pos = add_2d(new_pos, scale_2d(new_direction, fatigue_speed));
//                     being.pos = new_pos;
//                 }
//             });
//     }

//     pub fn check_being_collision(&mut self) {
//         let mut bs = self.being_keys.clone();
//         let mut adjust_queue: DashMap<u32, (f64, f64)> = DashMap::new();

//         bs.par_iter().for_each(|k| {
//             let being = self.beings.get(k).unwrap();
//             let (ci, cj) =  self.pos_to_chunk(being.value().pos);

//             [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)].into_par_iter().for_each(|(di, dj)| {
//                 let (nci, ncj) = (ci as isize + di, cj as isize + dj);

//                 if !(nci.min(ncj) < 0 || nci.max(ncj) as u32 >= self.n_chunks) {
//                     let (nci, ncj) = (nci as usize, ncj as usize);

//                     self.chunks[nci][ncj].lock().unwrap().being_keys.iter().for_each(|nk| {
//                         if *nk != *k {
//                             let (a, b) = being.pos;
//                             let (c, d) = self.beings.get(nk).unwrap().pos;

//                             let dist = dist_2d((a, b), (c, d));
//                             if dist < 2. * self.being_radius {
//                                 let diff = (c - a, d - b);
//                                 let dp = scale_2d(diff, 0.5);
                                


//                                 adjust_queue.insert(*k, dp);
//                             }


//                             // println!("{}, {},  {}, {}", a, b, c, d);
//                         }
//                     })
//                 }
//             });
//         });

//         adjust_queue.into_par_iter().for_each(|(k, v)| {
//             let pos = self.beings.get(&k).unwrap().pos;
//             let new_pos = add_2d(pos, scale_2d(v, -0.5));

//             if !(new_pos.0.min(new_pos.1) - self.being_radius < 1. || new_pos.0.max(new_pos.1) + self.being_radius > self.worldsize - 1.){
//                 self.beings.get_mut(&k).unwrap().pos = new_pos;
//             }
//         });
//     }

//     // fn pacwatch(&self, (pi, pj): (f64, f64), rad: f64) -> Vec<Vec<u32>> {
//     //     let (pi, pj) = (pi as u32, pj as u32);

//     // }
// }

// fn main() {
//     let mut world = World::new(18., 40);
//     let mut rng = thread_rng();
//     let rdist = Uniform::new(1., W_SIZE as f64 - 1.);
//     let rotdist = Uniform::new(-3.14, 3.14);
//     for i in (0..1000) {
//         world.add_being(
//             (rng.sample(rdist), rng.sample(rdist)),
//             rng.sample(rotdist),
//             10.,
//         );
//     }

//     let mut window: PistonWindow = WindowSettings::new("neuralang", [W_SIZE as u32, W_SIZE as u32])
//         .exit_on_esc(true)
//         .build()
//         .unwrap();
//     let mut i = 0;
//     let r = world.being_radius;
//     loop {
//         if false {
//             {
                
//                 if let Some(e) = window.next() {
//                     window.draw_2d(&e, |c, g, device| {
//                         clear([0., 0., 0., 1.], g);

//                     });

//                     world.beings.iter().for_each(|b| {
//                         window.draw_2d(&e, |c, g, device| {
//                             ellipse([1., 0., 0., 1.], [b.pos.1, b.pos.0, 2. * r, 2. * r], c.transform, g);
//                         });
//                     });

                    
//                 }
//             }
//         }
//         i += 1;
//         println!("{}", i);

//         world.move_beings();
//         world.check_being_collision();
//     }
// }
