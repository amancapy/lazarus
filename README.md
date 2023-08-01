# neuralang

overview/plan: small deepRL agents evolve to communicate with "sound". includes basic 2d physics with collision resolution built from scratch, rendered using the ggez library.

current progress: I'm able to simulate upwards of 40k objects at 60Hz, on a single thread. Instead of using theads to optimize a single simulation, since the rendering happens to be thread-safe, I want to run multiple simulation worlds, one per thread. Maybe lineages could be transferred between worlds to accelerate evolution. My gpu isn't all that beefy, so I guess for now 1-2 will have to suffice.