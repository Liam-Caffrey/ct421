
from collections import Counter
import numpy as np
import random
import concurrent.futures
import matplotlib.pyplot as plt
import time
import random

noCities = 0 # Global variable

# -------- read in file ----------------------
def readtsplib(filename):

    global noCities
    with open(filename, 'r') as file:
        lines = file.readlines()

    nodes = False #make sure we are not at the nodes yet
    cities = []

    for line in lines:
        line = line.strip()

        if line.startswith("DIMENSION"): #get the length
            noCities = int(line.split(":")[1].strip())


        if line == "NODE_COORD_SECTION":
            nodes = True
            continue
        elif line == "EOF":
            break # between NODE_COORD_SECTION and EOF read off

        if nodes:
            parts = line.split() #get the x and y values for each
            if len(parts) == 3:
                _, x, y = parts
                cities.append((float(x), float(y)))

    return cities

# -------- an array of random path (2d array) ----------------------
def initalizePop(noCity, generationSize):

    initalGen = [] #initalise out population with a set of random paths

    for _ in range(generationSize):
        tempPath = np.random.permutation(noCity).tolist()
        initalGen.append(tempPath)

    return initalGen

#--------- distance matrix ------------
# using this so don't have to recalculate
def getdistmatrix(cities):
    numcities = len(cities)
    distmat = np.zeros((numcities,numcities))

    for i in range(numcities):
        for j in range(i + 1, numcities):
            dist = np.linalg.norm(np.array(cities[i]) - np.array(cities[j]))
            distmat[i][j] = distmat[j][i] = dist #symetrical so do out distances once

    return distmat

#--------- fitness -------------------------

def fitnessfunction(path, distmatrix):
    totalDistance = 0

    for i in range(len(path)):
        city1 = path[i]
        city2 = path[(i + 1) % len(path)]
        totalDistance += distmatrix[city1][city2]
#calculates the distance of each path
    return totalDistance

#----- cross over --------------------------------------------------------------------------------------------------

def crossover1(parent1, parent2):
    #this takes 1 cuts it at a point and does the same to 2, thn adds the one to the other
    #then if the child isnt empty (-1), add whats left in parent 2 and add them on one by one

    size= len(parent1)
    child =  np.full(size, -1)

    start, end = sorted(random.sample(range(size), 2))

    child[start:end] = parent1[start:end]

    p2 = 0
    for i in range(size):
        if child[i] == -1:
            while p2 < size and parent2[p2] in child:
                p2 += 1
            if p2 < size:
                child[i] = parent2[p2]

    return child


def crossover2(parent1, parent2):
# process akin to crossover 1 however seemed to work out to be more efficient
    size = len(parent1)
    child = np.full(size, -1)

    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]

#create a list for each city not already in child
    p2 = [city for city in parent2 if city not in child]

    for i in range(size):
        if child[i] == -1:
            child[i] = p2.pop(0)

    return child

# ------------------- mutate ----------------------------
#swap two things

def mutate1(path, mutationRate):
#swap two random cities
    if random.random() < mutationRate:
        i, j = np.random.choice(len(path), 2, replace=False)
        path[i], path[j] = path[j], path[i]
    return path


def mutate2(path):
# swap sub sections of areas
    size = len(path)
    start, end = sorted(random.sample(range(size), 2))

    path[start:end] = path[start:end][::-1]

    return path

# ------------------- check if array is entirely unique ----------------------------
def isUnique(arr):
    return set(arr) == set(range(len(arr)))
#make sure the child contains all the correct cities and is correct length

#--------- selection ------------------------------------------------------------------------------------------------
# torunament selection and eliteism



def selection(population, fitVals, elitismRate, tournamentSize):

    mutationRate = 0.2
    popSize = len(population)
    noElite = int(popSize * elitismRate) #number of elites

    #elitism section
    eliteIns = np.argpartition(fitVals, noElite)[:noElite]
    elites = [population[i] for i in eliteIns]

    noSelected = popSize - noElite
    selected = []

    #tournament selection
    for _ in range(noSelected):

        tournamentIndices = np.random.choice(len(population), tournamentSize, replace=False)
        tournament = [population[i] for i in tournamentIndices]
        tournamentFitness = [fitVals[i] for i in tournamentIndices]


        bestIndex = tournamentIndices[tournamentFitness.index(min(tournamentFitness))]
        selected.append(population[bestIndex])



    newPopulation = elites[:]


    while len(newPopulation) < popSize:

        p1, p2 = random.choices(population, k=2)

        while True:
            child = crossover1(p1, p2)  # Generate a child
            if isUnique(child):  # Check if child has all cities
                child = mutate1(child, mutationRate)
                #child = mutate2(child)
                newPopulation.append(child)  # Add child to new population
                break


    return newPopulation




#--------- main ---------------------------------------------------------------------------------------------------
if 1 == 1:

    startTime = time.time()

    filename = "berlin52.tsp"; popSize = 500; maxGenerations = 2000 #small set
    #filename = "kroA100.tsp"; popSize = 600; maxGenerations = 2000 #medium set
    #filename = "pr1002.tsp"; popSize = 150; maxGenerations = 5000 #large set
    cities = readtsplib(filename)
    distmatrix = getdistmatrix(cities)
    noImprovementLimit =50

        #get a history of best fitness
    bestFitnessHistory = []
    avgFitnessHistory = []
    elitismRate = 0.1 #percentage of data kept
    tournamentSize = 4
    noImprovementCount = 0


        #print out cords test
    #print("Cords:")
    #for i, city in enumerate(cities):
    #    print(f"{i + 1}: {city}")
    #print("")

        #print out num cities test
    #print(f"{noCities}")
    #print("")

    #get the starting gen (genomes) and the fitness two arrays
    currentGen = initalizePop(noCities,popSize)
    currentGenFit = []

        #print distance matrix test
    #print(distmatrix)
    #print("")

    # initial population fitness
    for i, path in enumerate(currentGen):
        #print(f"Chromosome {i + 1}: {path}")
        currentGenFit.append(fitnessfunction(path, distmatrix))
    print("")

    #print out inital population fitness to see where we are
    #for i, fitness in enumerate(currentGenFit):
    #    print(f"{i + 1}: {fitness}")
    #print("")
    #print("")

    for gen in range(maxGenerations):

    #for each path in the current gen get a fitness value
        currentGenFit = [fitnessfunction(path, distmatrix) for path in currentGen]

    #keep track of the best fitness and average fitness
        bestFitness = min(currentGenFit)
        avgFitness = sum(currentGenFit) / popSize
        bestFitnessHistory.append(bestFitness)

    #if no improvement is seen end
        if gen > 1 and bestFitnessHistory[-1] >= bestFitnessHistory[-2]:
            noImprovementCount += 1
        else:
            noImprovementCount = 0

        if noImprovementCount >= noImprovementLimit:
            print(f"No improvement for {noImprovementLimit} generations, stopping...")
            break

        print(f"Generation {gen}: Best = {bestFitness:.2f}, Avg = {avgFitness:.2f}")



        currentGen = selection(currentGen, currentGenFit, elitismRate, tournamentSize)

    print(f"Algorithm finished at gen {gen}")
    endTime = time.time()
    print(f"Total Computational Time: {endTime - startTime:.2f} seconds")

    #testing of final genes
    print("")
    print("")
    #currentGenFit = []
    #for k, path in enumerate(currentGen):
        #print(f"Chromosome {i + 1}: {path}")
    #    currentGenFit.append(fitnessfunction(path, distmatrix))
    #print("")
    #for k, fitness in enumerate(currentGenFit):
    #    print(f"{k + 1}: {fitness}")

    plt.plot(bestFitnessHistory, label="Best Fitness")
    plt.plot(avgFitnessHistory, label="Average Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness (Total Distance)")
    plt.title("Fitness over Generations")
    plt.legend()
    plt.show()

