use rayon::prelude::*;
use dashmap::{DashMap, DashSet};
use std::{sync::{Arc, Mutex}, ops::Deref};
use rand::prelude::*;


#[derive(Debug)]
struct Being {
    id: u32,

    pos: (f32, f32),
    velocity: (f32, f32),
    rotation: f32,

    health: f32,
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

    chunks: Vec<Vec<Chunk>>,

    beings: DashMap<u32, Being>,
    foods: DashMap<u32, Food>,

    being_speed: f32,

    beingkey: u32,
    foodkey: u32,
}

impl World {
    pub fn new(chunk_size: f32, n_chunks: u32) -> Self {
        let world_width: f32 = chunk_size * (n_chunks as f32);

        World {
            chunk_size: chunk_size,
            n_chunks: n_chunks,

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
                velocity: (0., 0.),
                rotation: rotation,
                health: 10.,
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
}


fn main() {
    let world = World::new(25., 1000);
    println!("{:?}", world);
}
