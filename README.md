# Genetic Evolution Environment
Project in the making.\
This project is composed of:
- A genric neuroevolution framework.
- A compatible dense nerual network library that implements fixed structure, direct encoded agents.
- Demos - // TODO

## Demos

Meaningful demos have not been implemented yet.

### Proof of concept

This simple demo is the current [evolution_environment_main.cpp](./EvolutionEnvironment/evolution_environment_main.cpp) main of the main branch, used for testing. In this demo, a dense neural network with [1, 16, 16, 32, 64, 1] nodes per layer, with a total of 3027 parameters, is trained to approximate the function $$ y = \sin(x / 50) * \cos(\sqrt(x / 50)) $$ in the range $ x \in [0, 1000] $.

