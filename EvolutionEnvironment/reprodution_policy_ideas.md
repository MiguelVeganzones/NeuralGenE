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

### Parent selection
Parents of the next genration will belong to one of three groups:
1. The elite: The elite is the n top performing individials (with n in [0, 0.05 * gen_size] and elite_n = max(min_elite_count, elite_percenage * gen_size)). The elite_n will be copied directly into the next generation.
2. The survivors: the survivors are individuals chosen at random that will be copied into the next generation. The surviving chances of individals will depend on the ammount of spots in the next generation reserved for survivors and will be proportional to their fitness.
3. The progenitors: The progenitors will produce the larges portion of the new generation, the offsprings. These individuals will be created by crossovering randomly selected individuals from the previous generation. Parents will be chosen proportionally to their fitness. 

 