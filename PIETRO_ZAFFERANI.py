'''Summative Assignment - by Pietro Zafferani'''

'''These are all the libraries exploited for the completion of the tasks.'''

import random
from codetiming import Timer
import numpy
import scipy.optimize as optimization
import scipy

'''This function creates the list of elements, each element has an ID and a weight associated to.'''


def create_equipment(n: int) -> list:
    L = [(i, round(random.random(), 3)) for i in range(1, n + 1)]
    return L


'''This function sorts the list in ascendantly order based on the elements weight'''


def Sorting(List: list) -> list:
    return sorted(List, key=lambda i: i[1], reverse=False)


'''This function is the core of the Greedy algorithm. It takes as only argument a list, then the list gets ordered ascendantly
    by weight and the function fills the containers starting from the last elements of the ordered list.'''


def Greedy(L: list, check=True) -> list:
    if check:
        objects = Sorting(L)
    else:
        objects = L

    # max capacity of the container
    max_weight = round(numpy.sqrt(len(objects)), 2)
    S = []
    # index to search through the input list
    i = len(objects) - 1

    # loop executed until each element has been placed in the containers
    while i >= 0:

        container = []
        max_container = max_weight

        # check whether there is enough capacity in the container
        while max_container >= objects[i][1]:
            # subtract the newly-added element's weight to the remaining capacity
            max_container -= objects[i][1]
            # add element's ID to container list
            container.append(objects[i][0])
            i -= 1
            # check whether there are elements left to be processed
            if i < 0:
                # if not the case exit the scope if the loop (linexx)
                break

        # the container is full so it is inserted in S
        S.append(container)

    # eventually return the final list
    return S


'''this function generates all possible permutations of the input list.'''


def permutation(List):
    # base case of recursion
    if List == []:
        # generate an empty list
        yield []
    # repeat the recursion for every elements of the input list
    for Object in List:
        # we create a new list without the 'object' element
        new_l = [y for y in List if not y == Object]
        # repeat recursion for every elements of the newly-truncated list inside the upper loop
        for p in permutation(new_l):
            # generate one of the permuted list at a time
            yield ([Object] + p)


'''This function is the core of ExhaustiveSearch algorithm , it goes through all the possible ways of filling
     the containers and returns the combination that uses the fewest number of containers possible.'''


def ExhaustiveSearch(L: list) -> list:
    # number of containers that would be used in the worst case
    best = len(L)
    # S is set by default as empty
    S = []
    # generator of all possible permutations of L
    search_space = permutation(L)

    # go through every single permutation
    for single in search_space:
        # the list is given to Greedy() and a set of containers is returned
        # check=False means that the list is not sorted before being processed
        current_single = Greedy(single, check=False)
        # print(current_single)

        # if all the elements fit in only one container we found the best solution possible
        if len(current_single) == 1:
            return current_single

        # if a solution with fewer container is found, it is substituted to the former best one
        if len(current_single) < best:
            best = len(current_single)
            S = current_single

    return S


'''This function returns the average running time of the Greedy algorithm for a given input size (n). 
    The trial is repeated 1000 times.'''


def TestGreedy(n: int, repetitions=1000) -> float:
    # number of trials
    for i in range(repetitions):
        # create the list of elements
        Input = create_equipment(n)

        # record the running time
        with Timer(name='GreedyAlgorithm', logger=None):
            Greedy(Input)

    # return the average
    return round(Timer.timers.mean('GreedyAlgorithm'), 9)


'''This function returns the average running time of the Exhaustive Search algorithm for a given input size (n). 
    The trial is repeated 1000 times.'''


def TestExhaustive(n: int, repetitions=1) -> float:
    # number of trials
    for i in range(repetitions):
        # create the list of elements
        Input = create_equipment(n)

        # record the running time
        with Timer(name='ExhaustiveSearch', logger=None):
            ExhaustiveSearch(Input)

    # return the average
    return round(Timer.timers.mean('ExhaustiveSearch'), 5)


'''TFor a given array of input sizes, this function returns the average running times of the Greedy algorithm for each
    input size in the form of a list'''


def Time_setsGreedy(input_array: list) -> list:
    GreedyRes = list()
    # iterating over the list of input sizes
    for Input in input_array:
        # test the running time for each input size
        result = TestGreedy(Input)
        GreedyRes.append(result)
    # return the an array of running times
    return GreedyRes


'''TFor a given array of input sizes, this function returns the average running times of the Exhaustive Search algorithm for each
    input size in the form of a list'''


def Time_setsExhaustive(input_array: list) -> list:
    EXRes = list()
    # iterating over the list of input sizes

    for Input in input_array:
        # test the running time for each input size
        result = TestExhaustive(Input)
        EXRes.append(result)

    # return the an array of running times
    return EXRes


'''This function returns the coefficients to be given to the theoretical model in order to fit better the emprical data.
The model represents the time complexity in terms of big O notation for the Greedy Algorithm.'''


def fit_greedy(Set, times):
    # define the mathematical function
    def curve(x, a, b):
        return a * x * numpy.log2(x) + b

    # return the coefficients that shape the curve in order to fit the data
    # Set represents the X-axis coordinates
    # times represents the Y-xis coordinates
    return optimization.curve_fit(curve, Set, times)


'''This function returns the coefficients to be given to the theoretical model in order to fit better the emprical data.
The model represents the time complexity in terms of big O notation for the Exhaustive Search Algorithm.'''


def fit_exhaustive(Set, times):
    # define the mathematical curve
    def factorial(x, a, b):
        return a * x * numpy.log2(x) * scipy.special.factorial(x) + b

    # return the coefficients used to fit the model in the set of the parameters
    # Set represents the X-axis coordinates
    # times represents the Y-xis coordinates
    return optimization.curve_fit(factorial, Set, times)


'''This function takes two arrays as arguments and returns a set of pairs in which each element of the pair comes from 
    one of the two initial arrays.'''


def printPairs(Set, time_data) -> print:
    for (n, t) in zip(Set, time_data):
        print((n, t), end=' ')


'''This function takes as only paramenter a list of integers, each of them is the input size of a list of elements.
    The function tests, for each of the input sizes, if the Greedy algorithm found the optimal solution compared to
     the Exhaustive Search one'''


def routine(List: list) -> print:
    # for each element in the list
    for n in List:
        # create a list of n elements
        elements = create_equipment(n)
        # tests both algorithms on the same list
        greedy = Greedy(elements)
        ex = ExhaustiveSearch(elements)

        print('list of elements: ' + str(elements))
        print('Greedy result: ' + str(greedy))
        print('Exhaustive result: ' + str(ex) + '\n')

        # the Greedy did not find the optimal solution
        if len(greedy) != len(ex):
            print('Greedy algorithm did not perform well' + '\n')

        # the Greedy did find the optimal solution
        else:
            print('Greedy algorithm performed well' + '\n')


'''This function takes as argument one of the two algorithms, and tests it for a pre-defined series of size inputs'''


def benchmarking(function) -> print:
    for n in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        elements = create_equipment(n)

        # test the Greedy algorithm for every input size in the loop
        if function == 'Greedy':
            result = Greedy(elements)
            print(result)

        # test the exhaustive Search algorithm for every input size in the loop
        elif function == 'Exhaustive':
            result = ExhaustiveSearch(elements)
            print(result)


if __name__ == '__main__':
    # execute both algorithms on 6 elements
    a = create_equipment(6)
    print(a)
    print()
    print(Greedy(a))
    print(ExhaustiveSearch(a))
