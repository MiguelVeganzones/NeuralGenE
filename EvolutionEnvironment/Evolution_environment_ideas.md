# Evolution environment ideas

## Process

1. The first generation of agents is created.
2. This generation is split into independent groups (demes).
3. The current generation of agents is introduced into the system. Agents are evaluated by the system based on their performance.
4. The system may evolve.
5. A set of parents is chosed based on their performance, other metrics and randomness.
6. A new generation is created from the selected parents to replace the previous population. Mutations can occur.
7. The demes may vary.
8. Repeat from step 3.

## Components of the evolution environment

- **Population**: A group of agents with the concept of generation. Divided into demes.
- **System**: Where agents will be evaluated. Can be static or dynamic.
- **Reproduction manager**: Responsible for generating the new generations. Composed of:
    - *Parent selection algorithm*: Will choose parents.
    - *Parent reproduction algorithm*: Will reproduce parents.
    - *Mutation policy*: Will mutate the new generation.

## Ideas

- **The system**: The system can be static or dynamic. In static systems mutation and elites count can be higher than in dynamic systems. Should the mutaion rate evolve with the individual? This suggests that it should.

## Mutaton

Who owns the mutation parameters? Should they be another parameter to evolve? Should the mutation rate be shared among all agents?

### Conditions

- Mutation parameters should be flexible enough to require little to no tunning.
- Mutation should be higher in environments where the system is dynamic.
    - Low mutation rate:
        - Low genetic diversity, Agents are similar
        - Hiher specialization to the environemnt
        - Less adaptability to changes
    - High mutation rate:
        - High genetic diversity
        - Population is robust to changes in the environment

- Stored in the agent:
    - Mutation parameters evolve.
    - Mutation parameters vary according to an exponential distribution defined by parent parameters.
        - Population might get stuck.
        - Agents with the fittest mutation rates will have an advantage, which should emerge over time.
            - This dynamic will be slow because a 'fit' mutation rate can be evaluated just through side effects that take place over a probably long time.    
    - Will require discontinuities to reset mutation rates if they drif to unwanted values, such as very low mutation rates that get the population stuck.
    - The best mutation rate will change over time, the population will need to adapt to these changes.
    - Optimization over two parameters:
        - Fit mutation rates will be lost due to unfit agent parameters. 
        - Fit agents with unfit mutation rates will disappear without elitism, and will propagate bad mutation rate genes with elitism.
    - Probably the most powerful algorithm, but slow.

- Mutation policy
    - Mutation rates are shared among all individuals.
    - Can use trianing data to dynamically adjust mutation rates according to some heuristics.
    - Requires hand-crafted heuristics and tuning very sensitive hyperparameters.
    - Mutation rates should evolve during training, looking for good ranges of mutation rates.
    - Mutation rates can be increased when the population fitness is not progressing.
    - Mutation rates can be lowered to focus on local minima. Probably should care less about local minima at the beginning of the training and more at the end. 
    - Different mutation rates and policies can be used in different demes to compare performance. Probably too clumsy.

### Mutation parameters

- Mutation probability
    - Mutation parameters
- Replacement probability
    - Replacement parameters

## Elitism

 - Elitism should probably not always be on. Specially at the beginning.
