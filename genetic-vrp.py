"""
Team members:

1. Nguyen Lam Khanh Quynh - 1813776
2. Nguyen Dinh Thanh - 1813968

Solving the Vehicle Routing Problem (VRP) with multiple constraints using Genetic Algorithm.
"""

# Them cac thu vien neu can
import random
from random import randrange
from time import time
import math
import multiprocessing 


"""
Data structure to store information about each city.
- location: An ordered pair (x,y) contains the co-ordinates of the point in which the city is located.
- volume: The volume of supplies which will be shipped by the shipper at the city.
- weight: The weight of supplies which will be shipped by the shipper at the city.
"""

class City():

    def __init__(self, location, volume, weight):
        self.location = location
        self.volume = volume
        self.weight = weight


"""
Class to represent the problems to be solved by Genetic Algorithm.
- genes: a list of possible genes in a chromosome
- chroLen: the length of each chromosome
- decode: the method that receive the chromosome as input and returns the solution
to the original problem represented by the chromosome
- fitness: the method that returns the evaluation of a chromosome (i.e. how "good" 
a solution is to the original problem)
- crossOver: function that implements the crossover operator over two chromosomes
- mutation: function that implements the mutation operator over a chromosome
"""

class Genetic():
    
    def __init__(self, genes, noOfTrucks, decode, fitness):
        self.genes = genes
        self.noOfTrucks = noOfTrucks
        self.decode = decode
        self.fitness = fitness

    def checkChro(self, chro):
        dec = self.decode(chro)
        flat_dec = [item for sublist in dec for item in sublist]
        isDuplicate = any(flat_dec.count(element) > 1 for element in flat_dec)
        return not ([] in dec or len(dec) < self.noOfTrucks or isDuplicate)

    """
    Implement the One-Point Crossover operator over two chromosomes.
    For each pair of two parent chromosomes (parents), a random crossover point is selected
    and the tail of its two parents are swapped to get new off-springs.
    0 1 2 3 | 4 5 6        ->        0 1 2 3 | 6 0 1
    4 5 3 2 | 6 0 1        ->        4 5 3 2 | 4 5 6
    In our VRP, each city will be visited only one time. So, we need to implement a method to
    fix the repeated genes, if only, in the off-springs.
    """

    def crossOver(self, parent1, parent2):

        def fixRepeatedGene(child, parent):
            counter1 = 0
            for gene1 in child:
                if gene1[0] != 'truck':
                    repeat = child.count(gene1)
                    if repeat > 1:
                        for gene2 in parent[pos:]:
                            if gene2 not in child:
                                child[counter1] = gene2
                                break
                counter1 += 1

            return child

        new1 = []
        new2 = []

        while True:
            pos = random.randrange(1, len(parent1) - 1)
            child1 = parent1[:pos] + parent2[pos:]
            child2 = parent2[:pos] + parent1[pos:] 
            new1 = fixRepeatedGene(child1, parent1)
            new2 = fixRepeatedGene(child2, parent2)
            if self.checkChro(new1) and self.checkChro(new2):
                break


        return [new1, new2]

    """
    Implement the Inversion Mutation opeartor over a chromosome.
    It is used to maintain and introduce diversity in the genetic population 
    and is usually applied with a low probability: Mutation Probability
    In inversion mutation, we select a random subset of adjacent genes in the chosen chromosome,
    then merely invert the entire string in the subset.

    0 1 | 2 3 4 5 | 6 7 8    ->    0 1 | 5 4 3 2 | 6 7 8
    """

    def mutation(self, chro, prob):

        def inversion(chro):
            start = randrange(0, len(chro))
            end = randrange(start, len(chro))

            chroMid = chro[start:end]
            chroMid.reverse()

            return chro[0:start] + chroMid + chro[end:]

        newChro = []

        for _ in range(len(chro)):
            if random.random()  < prob:
                while True:
                    newChro = inversion(chro)
                    if self.checkChro(newChro):
                        break

        return newChro


def assign(file_input, file_output):
    # read input

    f = open(file_input, "r")
    lines = [line.rstrip('\n') for line in f]
    lines = [tuple(map(int, line.split(" "))) for line in lines]
    # Depot is the special city with volume and weight 0.
    depot = City(lines[0], volume = 0, weight = 0)
    noOfCities = lines[1][0]
    noOfTrucks = lines[1][1] 

    cities = [depot] + [City((lines[i][0], lines[i][1]), lines[i][2], lines[i][3]) for i in range(2, len(lines))]
    trucks = ['truck' for i in range(0, noOfTrucks - 1)]


    #The distances between locations are calculated using Euclid distance, 
    # in which the distance between two points, (x1, y1) and (x2, y2) is defined to be sqrt((x1 - x2)^2 + (y1 - y2)^2).
    dist = lambda p1, p2: math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    distance_matrix = [[dist(p1, p2) for p2 in [city.location for city in cities]] for p1 in [city.location for city in cities]]

    f.close()

    # run algorithm

    """
    Implement the functions that the Genetic Algorithm needs to work.

    The Genetic Algorithm function receives these attributes as input:
    - Genetic: an instance of Genetic class, with contains the problem to be solved
    by the Genetic Algorithm
    - selectionSize: number of participants on the selection tournaments.
    - noOfGens: number of generations (which is also the halting condition)
    - populationSize: number of chromosomes (i.e. individuals) for each generation
    - crossRatio: portion of the population which will be obtained by means of crossovers.
    - mutateProb: probability that a chromosome will be mutated

    The Genetic Algorithm function also has these method:
    - initialPopulation: Populate the initial population with completely random solutions.
    - newGeneration: Populate new generation based on tournament selection, crossover and mutation
    operator.
    """


    def genetic_algo(Genetic, selectionSize, noOfGens, populationSize, crossRatio, mutateProb):

        def initialPopulation(Genetic, populationSize):

            sampleChro = [gene for gene in Genetic.genes]

            
            def generateChro():
                citiesNo = [_ for _ in range(1, noOfCities + 1)]
                truckList = [[] for _ in range(0, noOfTrucks)]

                for truck in truckList:
                    choice = random.choice(citiesNo)
                    truck.append(choice)
                    citiesNo.remove(choice)

                for city in citiesNo:
                    choice = random.choice(truckList)
                    choice.append(city)
                
                chroNew = []

                for truck in truckList:
                    for i in truck:
                        chroNew += [gene for gene in sampleChro if gene[0] == i]
                    chroNew += [sampleChro[-1]]
            
                return chroNew[:-1]
            
            return [generateChro() for _ in range(populationSize)]   



        
        def newGeneration(Genetic, selectionSize, population, noOfParents, noOfDirects, mutateProb):

            def selection(Genetic, population, selectionSize, n):

                winners = []
                for _ in range(n):
                    ele = random.sample(population, selectionSize)
                    winners.append(min(ele, key = Genetic.fitness))
                return winners

            def crossOver(Genetic, parents):
                childs = []
                for i in range(0, len(parents), 2):
                    childs.extend(Genetic.crossOver(parents[i], parents[i+1]))
                return childs

            def mutation(Genetic, population, prob):
                
                for chro in population:
                    Genetic.mutation(chro, prob)
                return population

            directs = selection(Genetic, population, selectionSize, noOfDirects)
            crosses = crossOver(Genetic, selection(Genetic, population, selectionSize, noOfParents))
            mutations = mutation(Genetic, crosses, mutateProb)

            return directs + mutations


        population = initialPopulation(Genetic, populationSize)
        noOfParents = round(populationSize*crossRatio)
        noOfParents = (noOfParents if noOfParents % 2 == 0 else noOfParents - 1)
        noOfDirects = populationSize - noOfParents
        for _ in range(noOfGens):
            tempPopulation = newGeneration(Genetic, selectionSize, population, noOfParents, noOfDirects, mutateProb)
            if Genetic.fitness(min(tempPopulation, key=Genetic.fitness)) < Genetic.fitness(min(population, key=Genetic.fitness)):
                population = tempPopulation
        bestChro = min(population, key = Genetic.fitness)
        return (Genetic.decode(bestChro), Genetic.fitness(bestChro))

    """
    Implement decode function for VRP
    Our genes which represents the cities will have this form: (<city_index>, <volume>, <weight>)
    Other genes which represents the shippers will have this form: ("truck", '--', '--')
    Hence, each chromosome will be a list of above genes.
    The decode function will take a chromosome as an input, and returns a list contains the list of cities will be
    visited by each shipper.
    """

    def decode(chro):

        lst = []
        sol = []
        for i in range(0, len(chro)):
            if chro[i][0] not in trucks[:(noOfTrucks)]:
                lst += [(chro[i][0]) - 1]
            if (chro[i][0] in trucks[:(noOfTrucks - 1)]) or (i == len(chro) - 1):
                sol += [lst]
                lst = []

        return sol

    """
    Implement fitness function for VRP
    The revenue for each shipping order will be calculated using this formula: 5 + volume + weight*2
    The revenue of each shipper will be the total revenue for all shipping orders that the shipper has been done
    The cost of each shipper will be calculated using the formula: <total_distance>/40*20 + 10
    Hence, the income of each shipper is his/her revenue, minus the cost.
    Our mission is finding a set of delivery routes giving minimal gaps between the incomes of all shippers.
    """

    def fitness(chro):
        
        fitVal = 0
        chroDecode = decode(chro)
        fitList = [0 for i in range(noOfTrucks)]
        truckNo = 0
        idx = 0
        for truck in chroDecode:
            cost = 0
            rev = 0
            if truck:
                cost += distance_matrix[0][truck[0] + 1]
                for _ in truck:
                    rev += 5 + chro[idx][1] + chro[idx][2]*2
                    idx += 1
                for i in range(len(truck) - 1):
                    cost += distance_matrix[truck[i] + 1][truck[i+1] + 1]
                idx += 1
            else:
                idx += 1

            fitList[truckNo] = (rev - (cost/40 * 20 + 10)) if truck else 0
            truckNo += 1        

        for x in range(0, len(fitList)):
            for y in range(x, len(fitList)):
                fitVal += abs(fitList[x] - fitList[y])  

        return fitVal*2


    def VRP(noOfInstances):

        PROBLEM = Genetic([(i, cities[i].volume, cities[i].weight) for i in range(1, noOfCities + 1)] + [(trucks[0], '--', '--') for i in range(noOfTrucks - 1)],
        noOfTrucks, lambda x : decode(x), lambda y: fitness(y))

        sol = []
        for _ in range(0, noOfInstances):
            sol.append(genetic_algo(Genetic = PROBLEM, selectionSize = 2, noOfGens = 100, populationSize = 25, crossRatio = 0.9, mutateProb= 0.01))
        return min(sol, key = lambda t: t[1])
    
    sol = VRP(noOfInstances = 1000)

    # write output
    
    with open(file_output, "w") as f:
        for truck in sol[0]:
            f.write(' '.join(map(str, truck)) + '\n')
        f.write(str(sol[1]))

    return
    


if __name__ == '__main__':

    processes = []
    for i in range(0, 10):
        p = multiprocessing.Process(target = assign, args=('input.txt', 'output.txt',))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()