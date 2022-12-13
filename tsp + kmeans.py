import random
import math
import matplotlib.pyplot as plt
from random import shuffle
from numpy.random import choice
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.cluster import KMeans

# Get cities info.
def getCity():
    cities = []
    f = open("TSP51.txt")
    for i in f.readlines():
        node_city_val = i.split()
        cities.append(
            [float(node_city_val[0]), float(node_city_val[1])]
        )

    return cities


# Calculating distance of the cities.
def calcDistance(cities):
    total_sum = 0
    for i in range(len(cities) - 1):
        cityA = cities[i]
        cityB = cities[i + 1]

        d = math.sqrt(
            math.pow(cityB[0] - cityA[0], 2) + math.pow(cityB[1] - cityA[1], 2)
        )

        total_sum += d

    # Adds the distance also between the first and the last targets.
    cityA = cities[0]
    cityB = cities[-1] # The last target.
    d = math.sqrt(math.pow(cityB[0] - cityA[0], 2) + math.pow(cityB[1] - cityA[1], 2))

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

# Tournament Selection.
def tournamentSelection(population, TOURNAMENT_SELECTION_SIZE):
    parent_chromosome1 = sorted( # First parent.
                    random.choices(population, k=TOURNAMENT_SELECTION_SIZE)
                )[0] # Selects k random paths, sorts them, and choose the shortest one.
    parent_chromosome2 = sorted( # Second parent.
                    random.choices(population, k=TOURNAMENT_SELECTION_SIZE)
                )[0]
    return parent_chromosome1, parent_chromosome2

def truncationSelection(trunc, population):
    new_population = []
    sorted_fitness = sorted(population, key=lambda x: int(x[0]))
    for i in range(0, len(population)):
        r = random.randint((1 - trunc) * len(population), len(population) - 1)
        new_population.append(sorted_fitness[r])
    return sorted_fitness[0], sorted_fitness[1]     

# Roulette Wheel Selection.
# https://gist.github.com/rocreguant/b14ab2c2ecb58f98ee44b4d75785b8af
def rouletteWheelSelection(population):
    # Computes the totallity of the population fitness.
    population_fitness = 0
    for i in range(len(population)):
        population_fitness += population[i][0]
    
    # Computes for each chromosome the probability.
    chromosome_probabilities = []
    chromosomes = chromosomes = list(range(len(population))) # A list of the chromosomes' indexes.
    
    for i in range(len(population)):
        chromosome_probability = population[i][0]/population_fitness # Calculate each chromosome's probablity.
        chromosome_probabilities.append(chromosome_probability)
    
    # Temporarily sort both in the order of chromosome_probabilities
    chromosome_probabilities, chromosomes = zip(*sorted(zip(chromosome_probabilities, chromosomes)))
    
    # Correct probablities by swap the fitness values (the highest becomes the lowest etc...).
    #chromosomes = list(chromosomes)
    chromosome_probabilities = list(chromosome_probabilities)
    last_index = len(chromosome_probabilities)-1
    for i in range(int(len(chromosomes) / 2)):
        # Swap
        chromosome_probabilities[i], chromosome_probabilities[last_index] = chromosome_probabilities[last_index], chromosome_probabilities[i]
        #chromosomes[i], chromosomes[last_index] = chromosomes[last_index], chromosomes[i]
        last_index-=1
    
    # Correct probablities by swap the fitness values (the highest becomes the lowest etc...).
    #chromosome_probabilities = list(chromosome_probabilities)
    #chromosomes = list(chromosomes)
    #last_index = len(chromosome_probabilities)-1
    #for i in range(int(len(chromosome_probabilities) / 2)):
        # Swap
     #   chromosome_probabilities[i], chromosome_probabilities[last_index] = chromosome_probabilities[last_index], chromosome_probabilities[i]
      #  chromosomes[i], chromosomes[last_index] = chromosomes[last_index], chromosomes[i]
       # last_index-=1
    
    # Restore the original order of the chromosomes.
    #chromosomes, chromosome_probabilities = zip(*sorted(zip(chromosomes, chromosome_probabilities)))
    
    # Selects two chromosomes based on the computed probabilities.
    # This NumPy's "choice" function that supports probability distributions.
    # choice(list_of_candidates, number_of_items_to_pick, replace=False, p=probability_distribution)
    # "replace=False" to change the behavior so that drawn items are not replaced,
    # Default is True, meaning that a value of "a" can be selected multiple times.
    chromosome1_index, chromosome2_index = choice(chromosomes, 2, replace=False, p=chromosome_probabilities)
    parent_chromosome1 = population[int(chromosome1_index)]
    parent_chromosome2 = population[int(chromosome2_index)]
    
    return parent_chromosome1, parent_chromosome2


def rankSelection(population):
    sorted_population = sorted(population)
    ranked_population = []
    
    # Add rank to each chromosome.
    # The fittest gets the highest rank.
    # That because will make its probability the highest.
    sum_of_probablities = 0
    rank = len(population)
    for i in range(len(population)):
        ranked_population.append([rank, sorted_population[i]])
        sum_of_probablities += rank
        rank -= 1
    
    # Computes for each chromosome the probability.
    chromosome_probabilities = []
    chromosomes = list(range(len(population))) # A list of the chromosomes' indexes.
    for i in range(len(population)):
        chromosome_probability = ranked_population[i][0]/sum_of_probablities # Calculate each chromosome's probablity.
        chromosome_probabilities.append(chromosome_probability)
    
    # Selects two chromosomes based on the computed probabilities.
    # This NumPy's "choice" function that supports probability distributions.
    # choice(list_of_candidates, number_of_items_to_pick, replace=False, p=probability_distribution)
    # "replace=False" to change the behavior so that drawn items are not replaced,
    # Default is True, meaning that a value of "a" can be selected multiple times.
    chromosome1_index, chromosome2_index = choice(chromosomes, 2, replace=False, p=chromosome_probabilities)
    parent_chromosome1 = sorted_population[int(chromosome1_index)]
    parent_chromosome2 = sorted_population[int(chromosome2_index)]
    
    return parent_chromosome1, parent_chromosome2
    
def swapMutation(child_chromosome, lenCities):
    point1 = random.randint(0, lenCities - 1)
    point2 = random.randint(0, lenCities - 1)
    child_chromosome[point1], child_chromosome[point2] = ( # Selects 2 random genes and exchanges them.
        child_chromosome[point2],
        child_chromosome[point1],
    )  
    return child_chromosome
     
def inversionMutation(child_chromosome):
    point = random.randint(0, len(child_chromosome))
    child_chromosome[0:point] = reversed(child_chromosome[0:point])
    child_chromosome[point:len(child_chromosome)] = reversed(child_chromosome[point:len(child_chromosome)])
    return child_chromosome     
  
def scrambleMutation(child_chromosome):
    point1 = random.randint(0, len(child_chromosome))
    point2 = random.randint(0, len(child_chromosome))
    random.shuffle(child_chromosome[point1:point2])
    return child_chromosome  

# The Genetic Algorithm.
def geneticAlgorithm(
    population,
    lenCities,
    TOURNAMENT_SELECTION_SIZE,
    TRUNC_SELECTION_SIZE,
    MUTATION_RATE,
    CROSSOVER_RATE,
    #TARGET,
):
    gen_number = 0
    count = 0
    for i in range(200):
        new_population = []
        for i in range(int((len(population) - 2) / 2)):
            # SELECTION
            random_number = random.random() # Returns a random number between 0.0 - 1.0.
            if random_number < CROSSOVER_RATE:
                parent_chromosome1, parent_chromosome2 = tournamentSelection(population, TOURNAMENT_SELECTION_SIZE)
                #parent_chromosome1, parent_chromosome2 = truncationSelection(TRUNC_SELECTION_SIZE, population) 
                #parent_chromosome1, parent_chromosome2 = rouletteWheelSelection(population)
                #parent_chromosome1, parent_chromosome2 = rankSelection(population)
                
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

            # If crossover not happen
            else: # Choose two random paths.
                child_chromosome1 = random.choices(population)[0][1]
                child_chromosome2 = random.choices(population)[0][1]
            
            # MUTATION
            if random.random() < MUTATION_RATE:
                #Swap Mutation
                child_chromosome1 = swapMutation(child_chromosome1, lenCities)
                child_chromosome2 = swapMutation(child_chromosome2, lenCities)
                
                #Inversion Mutation
                #child_chromosome1 = inversionMutation(child_chromosome1)
                #child_chromosome2 = inversionMutation(child_chromosome2)
                
                #Scramble Mutation
                #child_chromosome1 = scrambleMutation(child_chromosome1)
                #child_chromosome2 = scrambleMutation(child_chromosome2)
                
            new_population.append([calcDistance(child_chromosome1), child_chromosome1])
            new_population.append([calcDistance(child_chromosome2), child_chromosome2])
            
        # REPLACEMENT
        # Selecting two of the best options we have (elitism).
        new_population.append(sorted(population)[0])
        new_population.append(sorted(population)[1])
        
        if new_population.sort() == population.sort(): # Increase count when there's no change in population.
            count += 1
        else:
            count = 0
        
        if count == 50: # If 10 generations stay the same, no need to continue.
            break
        
        population = new_population

        gen_number += 1
        if gen_number % 10 == 0: # Prints shortest path every 10 rounds.
            print(gen_number, sorted(population)[0][0])

        #if sorted(population)[0][0] < TARGET: # TO BE REMOVED!!! We can't know what is the real shortest path.
            #break

    answer = sorted(population)[0] # Prints shortest path found.

    return answer, gen_number

# Draw cities and answer map.
def drawMap(city, answer, color):
    city_index = 1
    for j in city: # Draws the targets.
        plt.plot(j[0], j[1], "ro") # "ro" = red marking for each target.
        plt.annotate(city_index, (j[0], j[1])) # Adds the index for each target: j[0] = index, j[1] = x, j[2] = y.
        city_index += 1

    for i in range(len(answer[1])): # Draws the line between the targets.
        try:
            first = answer[1][i]
            second = answer[1][i + 1]
            #plt.plot([first[1], second[1]], [first[2], second[2]], "gray")
            plt.plot([first[0], second[0]], [first[1], second[1]], color)
        except: # In case there is an out of range exception (because of i+1).
            continue

    # Draws the line between the first and the last targets.
    first = answer[1][0]
    second = answer[1][-1]
    #plt.plot([first[1], second[1]], [first[2], second[2]], "gray")
    plt.plot([first[0], second[0]], [first[1], second[1]], color)

    #plt.show()

# Gets cluster of cities (targets) with the same label given by KMeans.
def getCluster(cities, labels, label_index):
    cluster = []
    cluster_index = np.where(labels == label_index)
    for index in cluster_index[0]:
        cluster.append(cities[index])
    return cluster
       
def main():
    # Initial values.
    POPULATION_SIZE = 2000 # = number of possible paths.
    TOURNAMENT_SELECTION_SIZE = 4
    TRUNC_SELECTION_SIZE = 0.1
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.9
    #TARGET = 450.0 # Length of shortest path between all cities.
    K = 6
    results = []
    color = ""
    
    cities = getCity()

    for i in range(100):
        # Clustering the targets using KMeans
        kmeans = KMeans(n_clusters = K)
        #cities = df[['col2', 'col3']]
        kmeans.fit(cities)
        labels = kmeans.labels_
        kmeans.predict(cities)
        clusters = {i: getCluster(cities, labels, i) for i in range(kmeans.n_clusters)}
                    
        #################### LOOP
        sum_subgroups = 0
        j = 0
        for clusterOfCities in clusters.values():
            firstPopulation, firstFittest = selectPopulation(clusterOfCities, int(POPULATION_SIZE/K))
            answer, genNumber = geneticAlgorithm(
                firstPopulation,
                len(clusterOfCities),
                TOURNAMENT_SELECTION_SIZE,
                TRUNC_SELECTION_SIZE,
                MUTATION_RATE,
                CROSSOVER_RATE,
                #TARGET,
            )
            print("\n----------------------------------------------------------------")
            print("Generation: " + str(genNumber))
            print("Fittest chromosome distance before training: " + str(firstFittest[0]))
            print("Fittest chromosome distance after training: " + str(answer[0]))
            #print("Target distance: " + str(TARGET))
            print("----------------------------------------------------------------\n")
            if j == 0:
                color = "blue"
            elif j == 1:
                color = "green"
            elif j == 2:
                color = "purple"
            elif j == 3:
                color = "gray"
            elif j == 4:
                color = "pink"
            else:
                color = "brown"
            j += 1
            drawMap(cities, answer, color)
            sum_subgroups += answer[0]
        #################### END OF LOOP
        plt.show()
        results.append(sum_subgroups)

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(results, 'bo-')
    run = 0
    for res in results:
        plt.annotate(str(round(res, 2)), (run, res))
        run += 1
    plt.xlabel('Runs')
    plt.ylabel('Distances [m]')
    plt.xticks(np.arange(len(results)), np.arange(1, len(results)+1))
    plt.show()

main()
