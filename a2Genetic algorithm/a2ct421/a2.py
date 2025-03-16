import random
import numpy as np
import matplotlib.pyplot as plt


# Function to generate population of strategies
def generatepop(length, popsize):
    return ["".join(random.choice(["0", "1"]) for _ in range(length)) for _ in range(popsize)]


def fixedstrat3(lastinput):
    if lastinput == "C":
        return "D"
    else:
        return "C"


def fixedstrat5(rounds=15):
    return [random.choice(['C', 'D']) for _ in range(rounds)]


# -------------------------------calculate the move player will make-------------------------------------------
def getmymove(opmoves, strat):
    mymove = ""

    last5 = opmoves[-5:]  # Based on the previous 5 moves
    # Convert last five moves into 1s and 0s
    # 0s if C and 1s if D
    binarystr = ''.join(['0' if move == 'C' else '1' for move in last5])
    # Convert this binary string to decimal
    decimal = int(binarystr, 2)
    response = strat[decimal]
    # Allocated spot is the response
    if response == "0" or response == 0:
        mymove = "C"
    else:
        mymove = "D"

    return mymove


# ----------------------------------------------------------game logic------------------------------------------------
def prisonersdilemma(mymove, opmove):
    if mymove == 'C' and opmove == 'C':
        return 3
    elif mymove == 'C' and opmove == 'D':
        return 0
    elif mymove == 'D' and opmove == 'C':
        return 5
    elif mymove == 'D' and opmove == 'D':
        return 1


# --------------------------------------------------------- Fitness function -----------------------------------------------------
def fitnessfunction(inputnum, population, rounds, strat, probablitynoise, randomopmoves=None):
    score = 0
    mymoves = ["C"]
    opmoves = []

    # Generate the opponent moves based on the chosen strategy
    if randomopmoves:
        opmoves = randomopmoves
    else:
        if inputnum == 1:
            opmoves = ["C"] * rounds
        elif inputnum == 2:
            opmoves = ["D"] * rounds
        elif inputnum == 3:
            opmoves = [random.choice(["C", "D"])]
        elif inputnum == 4:
            opmoves = ["C", "D"] * (rounds // 2)

    # Apply noise to the opponent's moves just once, and make sure it's consistent for every genome
    noisyinopmoves = []
    for opmove in opmoves:
        if random.random() < probablitynoise:
            noisyinopmoves.append('C' if opmove == 'D' else 'D')
        else:
            noisyinopmoves.append(opmove)

    # Now, play the game, move by move, using the same noisy opponent moves for every genome
    for round_num in range(rounds):
        opmove = noisyinopmoves[round_num]  # Always use the same noisy opponent moves for every genome

        # Use your strategy to decide the player's move
        mymove = getmymove(noisyinopmoves, strat)

        # Record the move of the opponent and the player's move
        mymoves.append(mymove)
        noisyinopmoves.append(opmove)

        # Calculate score based on the moves
        score += prisonersdilemma(mymove, opmove)

        # For strategy 3, update the opponent's next move based on their last move
        if inputnum == 3:
            noisyinopmoves.append(fixedstrat3(opmove))  # Update with the opponent's next move based on last one

    return score


# ---------------------------------------------------------cross over---------------------------------------------------
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return child


# ---------------------------------------------------------mutation---------------------------------------------------
def mutation(bitstr, mutationRate=0.2):
    if random.random() < mutationRate:
        point = random.randint(0, len(bitstr) - 1)
        bitstr = list(bitstr)
        bitstr[point] = '1' if bitstr[point] == '0' else '0'
        return ''.join(bitstr)
    return bitstr


# ---------------------------------------------------------selection---------------------------------------------------
def selection(population, currentgenfit, elitismRate, tournamentSize):
    noElite = int(len(population) * elitismRate)
    sortedindices = np.argsort(currentgenfit)[::-1]
    sortedpopulation = [population[i] for i in sortedindices]

    elites = sortedpopulation[:noElite]
    newpopulation = elites[:]

    noSelected = len(population) - noElite
    selected = []

    for _ in range(noSelected):
        tournamentIndices = np.random.choice(len(population), tournamentSize, replace=False)
        tournament = [currentgenfit[i] for i in tournamentIndices]
        bestIndex = tournamentIndices[tournament.index(max(tournament))]
        selected.append(population[bestIndex])

    while len(newpopulation) < len(population):
        p1, p2 = random.choices(sortedpopulation, k=2)
        child = crossover(p1, p2)
        child = mutation(child)
        newpopulation.append(child)

    return newpopulation


# ---------------------------------------------------------main---------------------------------------------------
length = 32  # 2^5 possible states
popsize = 50
currentGen = generatepop(length, popsize)
currentGenFit = []

# GA settings
generation = 0
maxgenerations = 500
noImprovementLimit = 100
bestFitnessHistory = []
avgFitnessHistory = []
elitismRate = 0.1
tournamentSize = 4
noImprovementCount = 0
rounds = 15
probablitynoise = 0.5

print("Choose strategy to play against:")
print("1 for all C")
print("2 for all D")
print("3 for what opponent did last round")
print("4 for random")
print("5 for random 15 moves (same for every generation)")

inputnumber = input("Enter: ")

if inputnumber.isdigit():
    inputnumber = int(inputnumber)
else:
    exit()

print(f"Initial Population: {currentGen}")

# Generate random opponent moves if strategy 5 is selected
randomopmoves = None
if inputnumber == 5:
    randomopmoves = fixedstrat5(rounds)

# Calculate initial fitness
for i, strat in enumerate(currentGen):
    currentGenFit.append(fitnessfunction(inputnumber, popsize, rounds, strat, probablitynoise, randomopmoves))

# Print initial fitness scores
print("\nFitness Scores:")
for i, score in enumerate(currentGenFit):
    print(f"Chromosome {i + 1} Fitness: {score}")

# Genetic Algorithm loop
for gen in range(maxgenerations):
    bestFitness = max(currentGenFit)
    bestFitnessHistory.append(bestFitness)

    if gen > 1 and bestFitnessHistory[-1] <= bestFitnessHistory[-2]:
        noImprovementCount += 1
    else:
        noImprovementCount = 0

    if noImprovementCount >= noImprovementLimit:
        print(f"No improvement for {noImprovementLimit} generations, stopping...")
        break

    print(f"Generation {gen}: Best = {bestFitness:.2f}")

    # Selection, crossover, mutation, and next generation
    currentGen = selection(currentGen, currentGenFit, elitismRate, tournamentSize)

    # Recalculate fitness for the new generation
    currentGenFit = []
    for strat in currentGen:
        currentGenFit.append(fitnessfunction(inputnumber, popsize, rounds, strat, probablitynoise, randomopmoves))

# Plot fitness over generations
plt.plot(bestFitnessHistory)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Fitness Over Generations')
plt.show()
