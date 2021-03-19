# Solving Vehicle Routing Problem using Genetic Algorithm

Ever wondered how logistic service providers determine a set of rules for their vehicles to deliver things, such that all customers' requirements and operational constraints are satisfied? That's where Vehicle Routing Problem comes up.

## Prerequisite

Python3 is required to run the executing file. Also, we recommend that you have installed the most updated version of NumPy and Matplotlib to render the routing map smoothly. All you need to do is opening the terminal and typing the following

```bash
python3 -m pip install numpy matplotlib
```

## Input and output file format

### Input file

The input file begins with the position of the depot (that's where all vehicles start their route), represented by a pair of (x,y) coordinate. A line below contains the number of cities (or customers' positions) and the number of vehicles, respectively.

From line 2, each line contains the coordinate (x,y) of the city, the volume, and the weight of delivered things to that city, respectively.

An example is given below, and also note that all numbers are separated by whitespace.

```text
8 8
13 5
1 8 2 3
15 20 3 3
9 5 2 5
3 16 2 5
2 18 3 5
11 19 2 1
6 14 1 3
11 20 1 1
10 13 1 5
3 18 3 3
20 15 1 3
8 20 1 4
18 0 3 5
```

### Output file

In the output, each line is the route for the vehicle, represented by an ordered set of cities' indexes, corresponding with the appeared order in the input file. Also note that each vehicle will start from the depot, then go to the first city, and so on. But they will not go back to the depot after finishing the delivery process.

An example output file for the above input:

```text
0 10 5 7
3 2
4 11
12 6 1
8 9
```

## Modify the objective function

In our implement, the goal is to minimize the sum of all the difference between vehicles' costs and revenues. The objective function can be vary different depending on the particular application of the result, and you can get the desired result by modifing the fitness function.

```python
def fitness(chromosome):
    # Your code goes here

    return # the fitness value
```

## Parameters of Genetic Algorithm

You can easily change the parameters of Genetic Algorithm, such as the number of participants on the selection tournaments, the number of generations, the ratio for cross-over, or the probability that a chromosome will be mutated, etc. by modifing the following snippet of code

```python
def VRP(noOfInstances):
    # ...

    for _ in range(0, noOfInstances):
        sol.append(genetic_algo(
            Genetic = PROBLEM,
            selectionSize = 2,
            noOfGens = 100,
            populationSize = 25,
            crossRatio = 0.9,
            mutateProb= 0.01))

    return min(sol, key = lambda t: t[1])
```

## Examples

![Testcase 1](https://github.com/ngdthanh2000/genetic-vrp/blob/main/images/1.png?raw=true)
![Testcase 10](https://github.com/ngdthanh2000/genetic-vrp/blob/main/images/10.png?raw=true)
![Testcase 20](https://github.com/ngdthanh2000/genetic-vrp/blob/main/images/20.png?raw=true)
