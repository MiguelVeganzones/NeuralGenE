# Evoluition environment definition
## Evolution environment
### Member variables
1. Population
3. Reproduciton policy - Will generate the next population
### Static members
### Static interfaces
1. System - Will evaluate agent fitness. 


##  Population
Made up of agents
### Agents
Agents will be able to:
1. Mutate 
2. Crossover
3. React to their envirnoment

## System
Will be able to score agents based on their performance. Will give feedback, not necessarily of the constructive kind.

The system is a static part of the evolution envirnoment because:
1. Instances of the system may or may not be necessary to evaluate each agent (compare predicting data vs controlling a system). 
2. The system will need to manage its parallelism as it is who has the knowledge on how to do it. If the agents only need to look at const data parallelism is trivial, if agents need to manage an evolving system multiple approaches can be used to managfe this parallelism. Parallelism might not be an option at all.
3. The system will not change from generation to generation in the environment, so its interfcae to the environment must be const and thus a concrete instance is not needed. The system cannot show its mutable state to the envirmonemt.

The fact that the system is a static part of the environment means that - if the system could be used in paralell in multiple evolution environments - it either cannot have a mutable internal state or the internal muable state must be shared among all envirnonemnts.

There will not be a static instance of the system in the environment as that would imo give a wrong idea of what the system is. Evolution environments will know of a system, they will not have a system, not even a shared one.

## Reproduction manager
Will choose parents for next generation based on performance and diversity metrics as well as internal state.
Will return the next generation.
Will contain a mutation policy generator and will give it to newly created agents
### Reprodution policy
Will do the parent choosing
### Member functions
1. Diversity score evaluation
