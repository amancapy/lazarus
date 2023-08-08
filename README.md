# neuralang

overview/plan: small deepRL agents evolve to communicate with "sound". includes basic 2d physics with collision resolution built from scratch, rendered using the ggez library.

current progress: I'm able to simulate upwards of 40k objects at 60Hz, on a single thread. Instead of using theads to optimize a single simulation, since the rendering happens to be thread-safe, I want to run multiple simulation worlds, one per thread. Maybe lineages could be transferred between worlds to accelerate evolution. My gpu isn't all that beefy, so I guess for now 1-2 will have to suffice.

8/9: Currently toying with the idea of having one large net control all the creatures as batched stimulus-actions. this would create very uniform behaviour and eliminate hostile behavior entirely but given the larger capacity it will likely show much better planning and sophistication. i will defer taking this route until I am absolutely certain that n single-sample forward passes can not be made as efficient as one n-batch pass. the obvious middle way is to have a few (say 5) medium sized models control fractions of the population, color coded.