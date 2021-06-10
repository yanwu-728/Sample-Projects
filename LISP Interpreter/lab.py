import sys
sys.setrecursionlimit(5000)

import doctest
# NO ADDITIONAL IMPORTS!


###########################
# Snek-related Exceptions #
###########################

class SnekError(Exception):
    """
    A type of exception to be raised if there is an error with a Snek
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """
    pass


class SnekSyntaxError(SnekError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """
    pass


class SnekNameError(SnekError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """
    pass


class SnekEvaluationError(SnekError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SnekNameError.
    """
    pass


############################
# Tokenization and Parsing #
############################


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Snek
                      expression
    >>> test = '(foo (bar 3.14))'
    >>> tokenize(test)
    ['(', 'foo', '(', 'bar', '3.14', ')', ')']
    """
    new = ''
    i = 0
    while i < len(source):
        string = source[i]
        # Modify '(' and ')' so that split method could be used.
        if string == '(':
            new += ' ( '
        elif string == ')':
            new += ' ) '
        elif string == ';':
            # Handles the comment
            try:
                i += source[i:].index('\n')
            except:
                i = len(source)
        else: 
            new += string
        i += 1
    
    return new.split()
    


def parse(tokens, start = True):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    >>> token = ['(', 'cat', '(', 'dog', '(', 'tomato', ')', '1', ')', ')']
    >>> parse(token)
    ['cat', ['dog', ['tomato'], 1 ] ]
    """
    # Count the whether the number of parentheses match.
    num_left = 0
    num_right = 0
    for i in tokens:
        if i == '(':
            num_left += 1
        if i == ')':
            if num_left <= num_right:
                raise SnekSyntaxError
            else:
                num_right += 1
    if num_left != num_right:
        raise SnekSyntaxError

    # Base Case 1
    if tokens == []:
        return []
    # Base Case 2
    if len(tokens) == 1:
        try:
            return int(tokens[0])
        except:
            try:
                return float(tokens[0])
            except:
                return tokens[0]
    # If the first element is not '('
    if tokens[0] != '(':
        raise SnekSyntaxError

    # Assigning the variable to either a number or a function
    if tokens[1] == ':=':
        if type(tokens[2]) != str and (type(tokens[3]) != int or type(tokens[3]) != float or tokens[3] != '('):
            raise SnekSyntaxError

    def parse_helper(tokens, i = 0, start = True):
        result = []
        while i < len(tokens):
            if tokens[i] == '(' and i != 0:
                # Recursively build the list inside
                inside, next_i = parse_helper(tokens[i:], 1, False)
                i += next_i - 1
                result.append(inside)
            elif tokens[i] != '(':
                # try to convert the result to integer or float
                try:
                    result.append(int(tokens[i]))
                except:
                    try:
                        result.append(float(tokens[i]))
                    except:
                        if tokens[i] != ')':
                            result.append(tokens[i])
                        else:
                            # if reach a ')', ends
                            if result == []:
                                return [], i+1
                            if result[0] == ':=':
                                # Checking the format if the expression assigns value
                                if len(result) != 3:
                                    raise SnekSyntaxError
                                if type(result[1]) != str and type(result[1]) != list:
                                    raise SnekSyntaxError
                                if result[1] == []:
                                    raise SnekSyntaxError
                                for n in result[1]:
                                    if type(n) != str:
                                        raise SnekSyntaxError
                            if result[0] == 'function': 
                                # Checking the format if the expression is a function
                                if len(result) != 3:
                                    raise SnekSyntaxError
                                if type(result[1]) != list:
                                    raise SnekSyntaxError
                                for n in result[1]:
                                    if type(n) != str:
                                        raise SnekSyntaxError
                            if result[0] == 'if':
                                if len(result) != 4:
                                    raise SnekSyntaxError
                            if result[0] == 'cons':
                                if len(result) != 3:
                                    raise SnekSyntaxError
                            if result[0] == 'car' or result[0] == 'cdr':
                                if len(result) != 2:
                                    raise SnekSyntaxError
                            return result, i+1
            i += 1
        if result == []:
            raise SnekSyntaxError
        return result, i+1

    result = parse_helper(tokens)

    return result[0]

######################
# Built-in Functions #
######################

def multiply(lst):
    if lst == []:
        return None
    if len(lst) == 1:
        return lst[0]
    start = 1
    for i in lst:
        start *= i
    return start

def divide(lst):
    if len(lst) >= 2:
        return lst[0]/multiply(lst[1:])
    else:
        return None

def all_pairs(lst):
    result = set()
    for i in range(len(lst)-1):
        pair = (lst[i], lst[i+1])
        result.add(pair)
    return result

def ands(args, environment):
    for x in args:
        if evaluate(x, environment) == '#f':
            # if anything evaluates to false, return false without further evaluation
            return '#f'
    return '#t'

def ors(args, environment):
    for x in args:
        if evaluate(x, environment) == '#t':
            # if anything evaluates to true, return true without further evaluation
            return '#t'
    return '#f'

snek_builtins = {
    '+': sum,
    '-': lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    '*': multiply,
    '/': divide,
    '=?': lambda args: '#t' if all(x == args[0] for x in args) else '#f',
    '>': lambda args: '#t' if all(x[0] > x[1] for x in all_pairs(args)) else '#f',
    '>=': lambda args: '#t' if all(x[0] >= x[1] for x in all_pairs(args)) else '#f',
    '<': lambda args: '#t' if all(x[0] < x[1] for x in all_pairs(args)) else '#f',
    '<=': lambda args: '#t' if all(x[0] <= x[1] for x in all_pairs(args)) else '#f',
    'and': ands,
    'or': ors,
    'not': lambda arg: '#t' if arg[0] == '#f' else '#f',
    '#t': True,
    '#f': False,
    'nil': None,
    'length': lambda x: len(x[0]) if x[0] is not None else 0,
    'elt-at-index': lambda x, index: x[0].elt_at_index(index),
    'concat': lambda args: evaluate(['concat'] + args)
}

class Pair():
    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr

    def __len__(self):
        length = 0
        while self is not None:
            length += 1
            self = self.cdr
        return length
    
    def elt_at_index(self, index):
        if index == 0:
            return self.car
        if self.cdr != None and index-1 >= 0:
            # recursively find the element at index-1 in self.cdr
            return self.cdr.elt_at_index(index-1)
        else:
            raise SnekEvaluationError
    
    def copy(self):
        if self.cdr is None:
            return Pair(self.car, None)
        else:
            if type(self.cdr) != Pair:
                raise SnekEvaluationError
            return Pair(self.car, self.cdr.copy())
    
    def find_end(self):
        end = self.copy()
        while end.cdr is not None:
            end = end.cdr
            if type(end) != Pair:
                raise SnekEvaluationError
        return end
    
    def map(self, function):
        new = self.copy()
        # make a copy of self so that it doesn't modify the original list
        current = new
        # find the evaluated value for current.car
        current.car = calling_functions(function, [current.car])
        while current.cdr is not None:
            current = current.cdr
            # recursively go down the list and call the function on each car
            current.car = calling_functions(function, [current.car])
        return new
    
    def filter(self, function):
        current = self.copy()
        new = Pair(None, None)
        # start a new list
        new_copy = new
        if calling_functions(function, [current.car]) == '#t':
            new.car = current.car
            # initialize the new if the first element is included
        while current.cdr is not None:
            current = current.cdr
            # recursively go down in current and update new by modifying new_copy
            if calling_functions(function, [current.car]) == '#t':
                new_copy.cdr = Pair(current.car, None)
                new_copy = new_copy.cdr
            else:
                pass
        if new.car is None and new.cdr is None:
            return None
        return new
    
    def reduce(self, function, init):
        current = self.copy()
        # initialize the result by calling function on the initial value and the first element
        result = calling_functions(function, [init, current.car])
        while current.cdr is not None:
            current = current.cdr
            # recursively go down and update the result
            result = calling_functions(function, [result, current.car])
        return result


        

##############
# Evaluation #
##############

class Environment():
    def __init__(self, parent = None):
        self.value = {}
        self.parent = parent
    
    def assign(self, var, val):
        self.value[var] = val
        return val
    
    def copy(self):
        if self.parent is None:
            new = Environment()
        else:
            new = Environment(self.parent.copy())
        new.value = self.value.copy()
        return new
    
    def find(self, var):
        new_env = self.copy()
        if type(var) == str:
            while new_env is not None:
                if var in new_env.value:
                    return new_env.value[var]
                else:
                    # Keep looking up in the parent if not found in current environment.
                    new_env = new_env.parent
            raise SnekNameError
        else:
            return var
    
    def delete(self, var):
        if var in self.value:
            # remove the key and return its value
            return self.value.pop(var)
        else:
            raise SnekNameError
    
    def sets(self, var, val):
        if type(var) == str:
            while self is not None:
                if var in self.value:
                    # if the variable is in the current value, set it to be val.
                    self.value[var] = val
                    return val
                else:
                    # Keep looking up in the parent if not found in current environment.
                    self = self.parent
            raise SnekNameError
        else:
            return var

snek_builtins_environment = Environment()
snek_builtins_environment.value = snek_builtins

class Function():
    def __init__(self, parameters, function, parent):
        self.para = parameters
        self.parent = parent
        self.function = function
    


def evaluate(tree, environment = None):
    """
    Evaluate the given syntax tree according to the rules of the Snek
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    if environment == None:
        # the environment is children of the builtins
        environment = Environment(snek_builtins_environment)
    if type(tree) != list:
        if type(tree) == int or type(tree) == float or type(tree) == Function or type(tree) == Pair:
            return tree
        elif type(tree) == str:
            if tree in ['#t', '#f']:
                return tree
            else:
                # find the value of a variable
                return environment.find(tree)
        elif tree == None:
            return []
        else:
            raise SnekNameError
    else:
        if tree == []:
            raise SnekEvaluationError
        if tree[0] == 'if':
            if tree[1] == '#t' or evaluate(tree[1], environment) == '#t':
                return evaluate(tree[2], environment)
            else:
                return evaluate(tree[3], environment)
        if tree[0] == ':=':
            if type(tree[1]) == list:
                # convert the simple format
                tree = function_format_convert(tree)
                # assign the variable to the value of the assigned part 
            return environment.assign(tree[1], evaluate(tree[2], environment))
        else:
            if tree[0] == 'function':
                # build a function
                return building_function(tree, environment)
            if tree[0] == 'cons':
                return building_pair(tree)
            if tree[0] == 'car':
                try:
                    return evaluate(tree[1], environment).car
                except:
                    if tree[1] == None:
                        return 0
                    raise SnekEvaluationError
            if tree[0] == 'cdr':
                try:
                    return evaluate(tree[1], environment).cdr
                except:
                    if tree[1] == None:
                        return 0
                    raise SnekEvaluationError
            if tree[0] == 'list':
                result = building_list(tree, environment)
                return result
            if tree[0] in ['and', 'or']:
                return environment.find(tree[0])(tree[1:], environment)
            if tree[0] == 'length':
                try:
                    return len(evaluate(tree[1], environment))
                except:
                    if tree[1] == ['list']:
                        return 0
                    else:
                        raise SnekEvaluationError
            if tree[0] == 'elt-at-index':
                try:
                    return evaluate(tree[1], environment).elt_at_index(tree[2])
                except:
                    raise SnekEvaluationError
            if tree[0] == 'concat':
                if len(tree) < 2:
                    return None
                base = evaluate(tree[1], environment)
                for i in range(1, len(tree)-1):
                    base = concat(base, evaluate(tree[i+1], environment))
                return base
            if tree[0] == 'map':
                func = evaluate(tree[1], environment)
                lst = evaluate(tree[2], environment)
                if lst is None:
                    return None
                else:
                    if type(lst) == Pair:
                        return lst.map(func)
                    else:
                        raise SnekEvaluationError
            if tree[0] == 'filter':
                func = evaluate(tree[1], environment)
                lst = evaluate(tree[2], environment)
                if lst is None:
                    return None
                else:
                    if type(lst) == Pair:
                        return lst.filter(func)
                    else:
                        raise SnekEvaluationError
            if tree[0] == 'reduce':
                function = evaluate(tree[1], environment)
                lst = evaluate(tree[2], environment)
                if lst is None:
                    return tree[3]
                else:
                    if type(lst) == Pair:
                        return lst.reduce(function, tree[3])
                    else:
                        raise SnekEvaluationError
            if tree[0] == 'begin':
                new = tree[1:].copy()
                for i, arg in enumerate(new):
                    if i == len(new) - 1:
                        return evaluate(arg, environment)
                    # evaluate the argument in the environment, return it if it is the last expression
                    evaluate(arg, environment)
            if tree[0] == 'del':
                return environment.delete(tree[1])
            if tree[0] == 'let':
                return let(tree[1:], environment)
            if tree[0] == 'set!':
                return environment.sets(tree[1], evaluate(tree[2], environment))


            result = []
            for i in range(len(tree)):
                result.append(evaluate(tree[i], environment))
            try:
                # try to evaluate the function
                return calling_functions(result[0], result[1:])
            except TypeError:
                raise SnekEvaluationError

def function_format_convert(function):
    """
    Converts a function in simplified format to the original format
    """
    new_func = [':=', function[1][0], ['function']]
    if len(function[1]) > 1:
        var = [i for i in function[1][1:]]
        new_func[2].append(var)
    else:
        var = []
        new_func[2].append(var)
    new_func[2].append(function[2])
    return new_func

def building_function(exp, environment):
    """
    Build a function given an expression and an parent environment
    """
    new_environment = Environment(environment)
    return Function(exp[1], exp[2], new_environment)

def calling_functions(operator, parameters):
    """
    Evaluate the function given some value of the parameters
    """
    if type(operator) == Function:
        if len(operator.para) != len(parameters):
            # the number of values given does not match the number of variable in the function
            raise SnekEvaluationError
        # create a new environment from the parent of the operator and assign the value of the variable.
        new_environment = Environment(operator.parent)
        for i in range(len(operator.para)):
            evaluate([':=', operator.para[i], parameters[i]], new_environment)
        # evaluate the function in the newly created environment
        return evaluate(operator.function, new_environment)
    else:
        return operator(parameters)

def building_list(args, environment):
    if type(args) in [int, float, Pair]:
        return args
    else:
        if args is None or len(args) == 0:
            return None
        elif type(args) == list and args[0] == 'list':
            return building_list(args[1:], environment)
        elif len(args) == 1:
            return Pair(building_list(evaluate(args[0], environment), environment), None)
        else:
            return Pair(building_list(evaluate(args[0], environment), environment), building_list(args[1:], environment))

def concat(lst1, lst2):
    if lst1 == None or lst1 == []:
        return lst2.copy()
    if lst2 == None or lst2 == []:
        return lst1.copy()
    if type(lst1) != Pair or type(lst2) != Pair:
        raise SnekEvaluationError
    new = lst1.copy()
    current = new
    while current.cdr is not None:
        current = current.cdr
        if type(current) != Pair:
            raise SnekEvaluationError
    current.cdr = lst2.copy()
    return new

def building_pair(args):
    if args == 'nil':
        return None
    elif type(args) == int or type(args) == float:
        return args
    else:
        if type(args) == str:
            return Pair(args, None)
        return Pair(args[1], building_pair(args[2]))

def evaluate_file(string, environment = None):
    fil = open(string)
    exp = ' '.join(line.strip() for line in fil)
    fil.close()
    return evaluate(parse(tokenize(exp)), environment)

def let(args, environment):
    env = Environment(environment)
    for lst in args[0]:
        # assign the first element to be the evaluated result of the second element
        evaluate([':=', lst[0], evaluate(lst[1], environment)], env)
    return evaluate(args[1], env)

            
def result_and_env(tree, environment = None):
    """
    Evaluate a given tree in the given environment.
    """
    if environment is None:
        environment = Environment(snek_builtins_environment)
    return evaluate(tree, environment), environment
    
def REPL(environment = None):
    """
    Takes in an input, tokenizes and parses it, evaluates the expression and prints
    the result.
    >>> inp = (:= x (+ 2 3))
     out > 5
    >>> inp = (:= x2 (* x x))
     out > 25
    """
    if environment is None:
        environment = Environment(snek_builtins_environment)
    inp = input('in > ')
    while inp != 'QUIT':
        # try:
        result = evaluate(parse(tokenize(inp)), environment)
        print(' out >', evaluate(parse(tokenize(inp)), environment))
        # except:
        #     print(' out >', 'Error Happened')
        inp = input('in > ')


if __name__ == '__main__':
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    # doctest.testmod()
    
    env = Environment(snek_builtins_environment)
    lst = sys.argv[1:]
    for arg in lst:
        evaluate_file(arg, env)

    REPL(env)
