# Python language subset

The parser in Caddiepy is modifed to be able to exept .py scripts files instead of the .cad files.

The Python language subset is only very small part of Python, for the input python files for Caddiepy to read. Caddie is written for functional declarations in Standard ML looking syntax. To get it to work with limited Python syntax, which is a dynamically typed language, some formal changes had to made.

The input .py files to Caddipy, are legal python syntax, which means the that .py scripts can be run in the python interpreter and can also be used as input scripts to caddiepy for the parser. Caddiepy can take the derivative of the .py script inputs and by using combinatory automatic differentiation. 


# Comments

Comments are not allowed in .py files for caddiepy.

Standard Python comments with '#' is not allowed. 
Comments using SML syntax like (* comment *) cannot be run in the python intepreter. Changing the comment syntax is beyond scope of the project. It has to be done in SML-parser library.


# Function declarations

| SML       | Python      |
| --------- | ----------- |
| fn f x =  | def f(x):   |
|   ...     |   return .. |        

Function declarations in .cad files look like SML. In the .py syntax functions are declared with 'def' f(x): return...' The 'return' statement is necessary for Python functions to work. 


# Variable bindings (let-bindings)

| SML          | Python     |
| ------------ | ---------- |
| let val = e  | val = e;   | 
|  in e        | ...        |
|  end         |            |

Indentation, which is normal in Python to seperate expression is not allowed.
Instead ';' simicolon is used. 

Let bindings do not exist in Python. Instead for variable bingings we use 'x = e;'
and end the statment with a ';'.


# Operators

|       | SML       | Python    |
|-------| --------- | --------- |
|       | abs       | abs       |
|       | sin       | math.sin  |
|       | cos       | math.cos  |
|       | tan       | math.tan  |
|       | exp       | math.exp  |
|       | ln        | math.log  |
|       | pow       | pow       |
|       | pi        | math.pi   |
|       | +         | +         |
|       | -         | -         |
|       | *         | *         |
|       | /         | /         |
| Neg   | ~         | -         |
| Prj   | #1 x      | x[0]      |
| Prj   | #2 x      | x[1]      |
| Smul  | *>        | multiply(e, []) |


multiply is np.multiply for scalar multiplication (element wise)



# Functions

| SML       | Python              |
| --------- | ------------------- |
| Iota      | list(range(x))      |
| Map       | list(map(fun, xs))  |
| Red       | reduce(fun, xs, n)  |
| Array     |                     |
| Range     |                     |
| Tuple     |                     |




# Lamda function delcarations

| SML       | Python      |
| --------- | ----------- |
| fn e =>   | lambda x:   |
|   e       |   e |  

Lambda expression in Python do not allow for variable assignment in the expression: 
    lambda dec: exp

In .cad  fn x => let a = x + x in a + x is allowed 
In Python lambda x: a = x + x; a + x is not possible.

A way around this is to use nested lambdas in the expression like:

lambda x: (lambda a = x + x: a + x)() 
