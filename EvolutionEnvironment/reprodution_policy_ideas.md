## Reproduction policy ideas
### Scenario
Assume a simple concave 2D loss surface and an initial population far away from the global minimum.
### Parameters
w_diversity
w_fitness
elitism_rate
### Propposed method
1. Initialize the initial population.
2. Initialize the weights of diversity and fitness to random values in the range 0-1. Initialize elitism_  rate to 0.
3. Calcualte the distance matrix of the current generation and their fitness score.
4. Store the best fitness socre
5. Update the average distance value (?)
6. Choose parents for the next generation based on their compound fitness score f = w_diversity * diversity + w_fitness * fitness
7. Calculate the distance matrix of the current generation and their fitness score.
8. If the best fitness score is better than the previous best fitness score Goto step 9, Else go to step x.
9. Drop the weight of diversity to zero and set top_N to 2.
10. 
