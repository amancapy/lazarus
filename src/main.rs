use rayon::prelude::*;
use dashmap::{DashMap, DashSet};
use std::{sync::{Arc, Mutex}, ops::Deref};
use rand::prelude::*;
use minifb::{Key, Window, WindowOptions};


const W_SIZE: usize = 720;


#[derive(Debug)]
struct Being {
    id: u32,

    pos: (f32, f32),
    rotation: f32,

    health: f32,
    hunger: f32,

}

#[derive(Debug)]
struct Food {
    id: u32,
    pos: (f32, f32),
    val: f32,
}

#[derive(Debug)]
struct Chunk {
    pos: (u32, u32),
    being_keys: DashSet<u32>,
    food_keys: DashSet<u32>,
}


#[derive(Debug)]
struct World {
    chunk_size: f32,
    n_chunks: u32,
    worldsize: f32,

    chunks: Vec<Vec<Chunk>>,

    beings: DashMap<u32, Being>,
    foods: DashMap<u32, Food>,

    being_speed: f32,
    being_radius: f32,

    beingkey: u32,
    foodkey: u32,
}


fn normalize_2d((i, j): (f32, f32)) -> (f32, f32){
    let norm = (i.powi(2) + j.powi(2)).sqrt();

    (i / norm, j / norm)
}

fn add_2d((i, j): (f32, f32), (k, l): (f32, f32)) -> (f32, f32){
    (i + k, j + l)
}

fn scale_2d((i, j): (f32, f32), c: f32) -> (f32, f32){
    (i * c, j * c)
}

fn dist_2d((i1, j1): (f32, f32), (i2, j2): (f32, f32)) -> f32{
    ((i1 - i2).powi(2) + (j1 - j2).powi(2)).sqrt()
}

fn one_to_two(ij: usize) -> (usize, usize){
    ((ij - ij % W_SIZE) / W_SIZE, ij % W_SIZE)
}

fn two_to_one((i, j): (usize, usize)) -> usize{
    i * W_SIZE + j
}

impl World {
    pub fn new(chunk_size: f32, n_chunks: u32) -> Self {

        World {
            chunk_size: chunk_size,
            n_chunks: n_chunks,
            worldsize: chunk_size * (n_chunks as f32),

            chunks: (0..n_chunks)
                .into_par_iter()
                .map(|i| {
                    (0..n_chunks)
                        .into_iter()
                        .map(|j| Chunk {
                            pos: (i, j),
                            being_keys: DashSet::new(),
                            food_keys: DashSet::new(),
                        })
                        .collect()
                })
                .collect(),
            beings: DashMap::new(),
            foods: DashMap::new(),

            being_speed: 10.,
            being_radius: chunk_size - 5.,

            beingkey: 0,
            foodkey: 0,
        }
    }

    fn pos_to_chunk(&self, pos: (f32, f32)) -> (usize, usize) {
        let i = ((pos.0 - (pos.0 % self.chunk_size)) / self.chunk_size) as usize;
        let j = ((pos.1 - (pos.1 % self.chunk_size)) / self.chunk_size) as usize;

        (i, j)
    }

    pub fn add_food(&mut self, pos: (f32, f32), val: f32, age: f32) {
        self.foods.insert(
            self.foodkey,
            Food {
                id: self.foodkey,
                pos: pos,
                val: val,
            },
        );

        let (i, j) = self.pos_to_chunk(pos);
        self.chunks[i][j].food_keys.insert(self.foodkey);

        self.foodkey += 1;
    }

    pub fn add_being(&mut self, pos: (f32, f32), rotation: f32, health: f32) {
        self.beings.insert(
            self.beingkey,
            Being {
                id: self.beingkey,
                pos: pos,
                rotation: rotation,
                health: 10.,
                hunger: 0.
            },
        );

        let (i, j) = self.pos_to_chunk(pos);
        self.chunks[i][j].being_keys.insert(self.beingkey);

        self.beingkey += 1;
    }


    pub fn decay_food(mut self){
        self.foods.par_iter_mut().for_each(|mut entry|{
            entry.value_mut().val *= 0.9;
        });

        self.foods.retain(|_, food|{
            food.val > 0.05
        });
    }

    pub fn move_beings(mut self){
        self.beings.par_iter_mut().for_each(|mut entry|{

            let being = entry.value_mut();
            let direction = (being.rotation.cos(), being.rotation.sin());
            let fatigue_speed = (10. - being.hunger) / 10. * self.being_speed;

            let curr_pos = being.pos.clone();
            let new_pos = add_2d(curr_pos, scale_2d(direction, fatigue_speed));

            if (new_pos.0 - self.being_radius < 0. || new_pos.0 + self.being_radius > self.worldsize
            || new_pos.1 - self.being_radius < 0. || new_pos.1 + self.being_radius > self.worldsize){

            }

            being.pos = new_pos;


            let curr_chunk = self.pos_to_chunk(curr_pos);
            let new_chunk = self.pos_to_chunk(new_pos);


            if !(curr_chunk == new_chunk){
                self.chunks[curr_chunk.0][curr_chunk.1].being_keys.remove(&being.id);
                self.chunks[new_chunk.0][new_chunk.1].being_keys.insert(being.id);
            }
        });
    }

    pub fn check_being_collision(mut self){
        self.beings.par_iter_mut().for_each(|being|{
            let (key, being) = (being.key(), being.value());

            let (bci, bcj) = self.pos_to_chunk(being.pos);
            for (di, dj) in [(0, 0), (0, 1), (1, 0), (1, 1)]{
                let (a, b) = (bci as i32 + di, bcj as i32 + dj);

                if !(a < 0 || b < 0 || a >= self.n_chunks as i32 || b >= self.n_chunks as i32){
                    let (a, b) = (a as usize, b as usize);
                    self.chunks[a][b].being_keys.par_iter().for_each(|key|{
                        
                        let other_pos = self.beings.get(&key).unwrap().value().pos;
                        let self_pos = being.pos;

                        if dist_2d(self_pos, other_pos) < 2. * self.being_radius{
                            // TODO
                        }

                    });
                }
            }
        })
    }
    
    pub fn get_being_pixels(mut self){
        let shared_buffer = (0..W_SIZE.pow(2)).into_par_iter().for_each(|ij|{
            let (i, j) = one_to_two(ij);
        });
    }
}

fn main() {
    let mut buffer: Vec<u32> = vec![0; W_SIZE.pow(2)];

    let mut window = Window::new(
        "Test - ESC to exit",
        W_SIZE,
        W_SIZE,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));
    while window.is_open() && !window.is_key_down(Key::Escape) {
        
        


        window
            .update_with_buffer(&buffer, W_SIZE, W_SIZE)
            .unwrap();
    }
}