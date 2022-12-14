import random
import math
import matplotlib.pyplot as plt
from numpy.random import choice

# Get cities info.
def getCity():
    cities = []
    f = open("TSP51.txt")
    for i in f.readlines():
        node_city_val = i.split()
        cities.append(
            [node_city_val[0], float(node_city_val[1]), float(node_city_val[2])]
        )

    return cities


# Calculating distance of the cities.
def calcDistance(cities):
    total_sum = 0
    for i in range(len(cities) - 1):
        cityA = cities[i]
        cityB = cities[i + 1]

        d = math.sqrt(
            math.pow(cityB[1] - cityA[1], 2) + math.pow(cityB[2] - cityA[2], 2)
        )

        total_sum += d

    # Adds the distance also between the first and the last targets.
    cityA = cities[0]
    cityB = cities[-1] # The last target.
    d = math.sqrt(math.pow(cityB[1] - cityA[1], 2) + math.pow(cityB[2] - cityA[2], 2))

    total_sum += d

    return total_sum


# Selecting the population.
def selectPopulation(cities, size):
    population = []

    for i in range(size): # size = number of possible paths.
        c = cities.copy() # Copy in order to not change the original order of targets.
        random.shuffle(c) # Get a random path between the targets.
        distance = calcDistance(c) # Calculate the fitness value (= total distance between the targets).
        population.append([distance, c]) # Adds the path (= chromosome) and its total distance to the papulation.
    fittest = sorted(population)[0] # Takes the fittest (= shortest path).

    return population, fittest # Returns the current population and the shortest path.

# The Genetic Algorithm.
def geneticAlgorithm(
    population,
    lenCities,
    STEADY_STATE_SELECTION_RATE, # The precentage of entire population
    TARGET,
):
    gen_number = 0
    size_of_fittest_population = int(len(population)*STEADY_STATE_SELECTION_RATE)
    for i in range(200):
        new_population = []
        sorted_population = sorted(population)
        # The fittest individuals are copied to the new population.
        for i in range(size_of_fittest_population):
            new_population.append(sorted_population[i])
        num_of_least_fittest = len(population)-size_of_fittest_population # Size of the population to be replaced
        for i in range(int(num_of_least_fittest / 2)):
            # SELECTION (Steady State)
            # Two fittest individuals are selected randomally to produce offspring
            fittest_parent1_index = random.randint(0, size_of_fittest_population)
            fittest_parent2_index = random.randint(0, size_of_fittest_population)

            parent_chromosome1 = population[fittest_parent1_index]
            parent_chromosome2 = population[fittest_parent2_index]
                
            # CROSSOVER (Order Crossover Operator)
            point = random.randint(0, lenCities - 1) # Selects a random index.
            # First child.
            child_chromosome1 = parent_chromosome1[1][0:point] # Selects a sub-path (from its beginning - to the "point" index).
            for j in parent_chromosome2[1]: # Adds the missing targets from the second parent.
                if (j in child_chromosome1) == False:
                    child_chromosome1.append(j)
            # Second child.
            child_chromosome2 = parent_chromosome2[1][0:point]
            for j in parent_chromosome1[1]:
                if (j in child_chromosome2) == False:
                    child_chromosome2.append(j)


            # MUTATION (Swap Mutation)
            point1 = random.randint(0, lenCities - 1)
            point2 = random.randint(0, lenCities - 1)
            child_chromosome1[point1], child_chromosome1[point2] = ( # Selects 2 random genes and exchanges them.
                child_chromosome1[point2],
                child_chromosome1[point1],
            )

            point1 = random.randint(0, lenCities - 1)
            point2 = random.randint(0, lenCities - 1)
            child_chromosome2[point1], child_chromosome2[point2] = (
                child_chromosome2[point2],
                child_chromosome2[point1],
            )

            new_population.append([calcDistance(child_chromosome1), child_chromosome1])
            new_population.append([calcDistance(child_chromosome2), child_chromosome2])
        
        population = new_population
        
        gen_number += 1

        if gen_number % 10 == 0: # Prints shortest path every 10 rounds.
            print(gen_number, sorted(population)[0][0])

        if sorted(population)[0][0] < TARGET: # TO BE REMOVED!!! We can't know what is the real shortest path.
            break

    answer = sorted(population)[0] # Prints shortest path found.

    return answer, gen_number


# Draw cities and answer map.
def drawMap(city, answer):
    for j in city: # Draws the targets.
        plt.plot(j[1], j[2], "ro") # "ro" = red marking for each target.
        plt.annotate(j[0], (j[1], j[2])) # Adds the index for each target: j[0] = index, j[1] = x, j[2] = y.

    for i in range(len(answer[1])): # Draws the line between the targets.
        try:
            first = answer[1][i]
            secend = answer[1][i + 1]
            plt.plot([first[1], secend[1]], [first[2], secend[2]], "gray")
        except: # In case there is an out of range exception (because of i+1).
            continue

    # Draws the line between the first and the last targets.
    first = answer[1][0]
    secend = answer[1][-1]
    plt.plot([first[1], secend[1]], [first[2], secend[2]], "gray")

    plt.show()


def main():
    # Initial values.
    POPULATION_SIZE = 2000 # = number of possible paths.
    STEADY_STATE_SELECTION_RATE = 0.6
    TARGET = 450.0

    cities = getCity()
    firstPopulation, firstFittest = selectPopulation(cities, POPULATION_SIZE)
    answer, genNumber = geneticAlgorithm(
        firstPopulation,
        len(cities),
        STEADY_STATE_SELECTION_RATE,
        TARGET,
    )

    print("\n----------------------------------------------------------------")
    print("Generation: " + str(genNumber))
    print("Fittest chromosome distance before training: " + str(firstFittest[0]))
    print("Fittest chromosome distance after training: " + str(answer[0]))
    print("Target distance: " + str(TARGET))
    print("----------------------------------------------------------------\n")

    drawMap(cities, answer)


main()
