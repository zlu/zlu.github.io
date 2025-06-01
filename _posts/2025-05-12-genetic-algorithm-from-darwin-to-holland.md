---
layout: post
title: 'Genetic Algorithm: From Darwin to Holland'
date: 2025-05-12 18:09 +0800
tags: 
tags: 
- Optimization
- Metaheuristics
- Genetic Algorithm
- Python
- Algorithm
comments: true
---

![Genetic Algorithm: Algo-Scope.online](/assets/images/uploads/ga.png)

To view an animated version: **[Genetic Algorithm: Algo-Scope.online](https://www.algo-scape.online/ga)**

Have you ever wondered how engineers optimize aircraft wing designs for the best fuel efficiency, lift, and drag under tricky aerodynamic constraints? Or how universities manage to schedule thousands of classes, avoiding timetable clashes and making sure every course has a classroom? These are just a couple of typical applications for Genetic Algorithms (GA), a fascinating approach inspired by Charles Darwin's theory of **evolution**.

In his groundbreaking book *On the Origin of Species*, Darwin explained that variations occur during reproduction. These variations, if they give an organism a better chance of surviving and reproducing (i.e., increase its 'reproductive fitness'), are more likely to be passed on to future generations. You've probably heard this summed up as 'survival of the fittest'.

Fast-forward to 1975. John Holland, a brilliant professor at the University of Michigan, took these ideas and formalized them into what we now call Genetic Algorithms. In his influential book, *Adaptation in Natural and Artificial Systems*, he introduced key concepts like _selection_, _crossover_, and _mutation_ as mechanisms for 'evolving' solutions to problems. The beauty of GAs is that these adaptive systems can learn and find optimized solutions even in complex environments that we don't fully understand.

At its core, a GA performs the following steps:

1.  **Initialize Population:** We kick things off by creating a "population" – basically, a bunch of random potential solutions to our problem. Think of it as throwing a lot of ideas at the wall to see what sticks.
    Here's an example of how you might generate a random bit-vector (a common way to represent solutions):
    ```python
    import random # Make sure random is imported for this snippet too

    def generate_random_bit_vector(length):
        vector = []
        for _ in range(length):
            vector.append(1 if random.random() < 0.5 else 0) # 50/50 chance for 0 or 1
        return vector
    ```

2.  **Assess Fitness:** Next, we need a way to tell how "good" each solution is. This is where the "fitness function" comes in. It scores each solution – the higher the score, the "fitter" the solution.

3.  **Selection of Parents:** Now for the "survival of the fittest" part! We pick parent solutions to "breed," giving preference to those with higher fitness scores. There are a few ways to do this, like "Fitness-Proportionate Selection" (fitter ones have a higher chance of being picked) or "Tournament Selection" (a few solutions duke it out, and the best one wins parenthood).

4.  **Crossover (Reproduction):** This is where the magic happens. We take two parent solutions and mix their "genes" (parts of their solutions) to create one or more "offspring" (new solutions). A common method is "One-Point Crossover":
    ```python
    def one_point_crossover_example(parent1, parent2): # Renamed to avoid conflict if run standalone
        # Pick a random point in the solution string
        point = random.randint(1, len(parent1)-1) 
        # Create children by swapping the "tails"
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    ```
    There's also Two-Point Crossover (uses two points for swapping a middle segment) and Uniform Crossover (each bit is swapped with some probability).

5.  **Mutation:** To keep things interesting and avoid getting stuck, we introduce random changes, or "mutations," into the offspring. This adds diversity to the population and helps explore new parts of the solution space. A simple "Bit-Flip Mutation" does this:
    ```python
    def bit_flip_mutation_example(vector, prob): # Renamed for clarity
        # prob is the mutation probability
        return [1 - bit if random.random() < prob else bit for bit in vector]
    ```

6.  **Replace and Repeat:** Finally, the new generation of offspring replaces the old population (or part of it). We often keep track of the best solution found so far (a strategy called "elitism"). Then, the whole cycle (steps 2-6) repeats for many generations, or until we find a solution that's 'good enough,' or we simply run out of time/iterations.

A classic example for GA is perhaps less exciting than optimizing aircraft wings or car bodies, but it's a great way to illustrate how the algorithm works. We'll optimize a binary string! The goal is to maximize a fitness function: $$f(x) = \text{sum}(x)$$, where $$x$$ is a binary string of length 20. The optimal solution is a string of all 1s (which would give a fitness of 20).

Let's see it in action with some Python code:

```python
import random

# Parameters
POP_SIZE = 50
STRING_LENGTH = 20
MUTATION_PROB = 0.05
GENERATIONS = 100

# Initialize population (Algorithm 21: Generate a Random Bit-Vector)
def init_population():
    return [[random.choice([0, 1]) for _ in range(STRING_LENGTH)] for _ in range(POP_SIZE)]

# Fitness function: sum of bits
def assess_fitness(individual):
    return sum(individual)

# Tournament selection (Algorithm 32: Tournament Selection)
def tournament_selection(population, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    # Return the individual with the highest fitness in the tournament
    return max(tournament, key=assess_fitness)

# One-point crossover (Algorithm 23: One-Point Crossover)
def one_point_crossover(parent1, parent2):
    # Ensure crossover point is not at the very ends
    point = random.randint(1, STRING_LENGTH - 1) 
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Bit-flip mutation (Algorithm 22: Bit-Flip Mutation)
def bit_flip_mutation(individual):
    return [1 - bit if random.random() < MUTATION_PROB else bit for bit in individual]

# Main GA loop (Algorithm 20: The Genetic Algorithm)
def genetic_algorithm():
    population = init_population()
    overall_best_individual = None
    overall_best_fitness = -1 # Initialize with a very low fitness

    for generation_num in range(GENERATIONS):
        # Assess fitness of current population and find the current generation's best
        # Also, update the overall best if a new champion emerges
        current_gen_best_fitness = -1
        for individual in population:
            fitness = assess_fitness(individual)
            if overall_best_individual is None or fitness > overall_best_fitness:
                overall_best_individual = list(individual) # Store a copy
                overall_best_fitness = fitness
            if fitness > current_gen_best_fitness:
                current_gen_best_fitness = fitness
        
        # Optional: Print progress for the current generation
        # print(f"Generation {generation_num+1}: Best Fitness = {overall_best_fitness}")


        # Create new population
        new_population = []
        # Elitism: Optionally carry over the best individual(s) directly
        # if overall_best_individual is not None:
        #    new_population.append(list(overall_best_individual)) 

        # Fill the rest of the population with offspring
        # Adjust loop range if using elitism to maintain POP_SIZE
        for _ in range(POP_SIZE // 2): 
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = one_point_crossover(parent1, parent2)
            new_population.extend([bit_flip_mutation(child1), bit_flip_mutation(child2)])
        
        # If POP_SIZE is odd, we might be one short, can add another mutated child or handle as needed
        if len(new_population) < POP_SIZE and POP_SIZE % 2 != 0:
             parent1 = tournament_selection(population)
             parent2 = tournament_selection(population)
             child1, _ = one_point_crossover(parent1, parent2)
             new_population.append(bit_flip_mutation(child1))


        population = new_population[:POP_SIZE] # Ensure population size is maintained

        # Early stopping if optimal solution found
        if overall_best_fitness == STRING_LENGTH:
            print(f"Optimal solution found in generation {generation_num+1}!")
            break
            
    return overall_best_individual, overall_best_fitness

# Run GA
best_solution, best_fitness = genetic_algorithm()
print(f"Best solution found: {best_solution}, Fitness: {best_fitness}")
```