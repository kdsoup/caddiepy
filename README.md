<!-- # caddie [![CI](https://github.com/diku-dk/caddie/workflows/CI/badge.svg)](https://github.com/diku-dk/caddie/actions) -->
# Caddiepy

Combinatory Automatic Differentiation in a Python context.

This project is a Python context implementation of the standalone tool [Caddie](https://github.com/diku-dk/caddie). 
The aim of the implementation is to enable Combinatory Automatic Differentiation to be used in a Python programming context. 
The Caddiepy tool can differentiate Python programs to linear map derivative code using a forward-mode and a reverse-mode automatic differentiation.

Consider the Python function:

    def f(x): return x[0] * sin(x[1])

Using Caddiepy the function differentiates to:

    def f_diff(x1,x2,dx1,dx2): v1 = sin(x2); return ((dx1*v1) + (x1*(cos(x2)*dx2)))

Most of the code for Caddiepy is adapted from the implementation of Caddie. The theory of Combinatory Automatic Differentiation is based on the paper [Combinatory Adjoints and Differentiation](https://elsman.com/pdf/msfp22.pdf) [1].  

# Setup

To be able to run Caddiepy, the Standard ML compiler toolkit [MLKit](https://github.com/melsman/mlkit) is required. 
The MLKit repository provides instructions for installing the toolkit. 
For macOS ARM computers, MLKit has to be installed using Rosetta.  

When MLKit is installed, download or clone Caddiepy. 
In the terminal, navigate to the ```caddiepy``` folder and type the command ```make```. 
The makefile command will compile the Caddiepy source code to an executable program using MLKit. 
The executable named ```./cad``` is available in the src folder.  

[Python](https://www.python.org/) and [NumPy](https://numpy.org/) are required to support Python functionality.

# How to use

To run the program, navigate to the ```src``` folder in the terminal, and type 

    ./cad --Pdiffu some-file.py

With this command, Caddiepy will compute the unlinearised linear map derivative of the input python file ```some-file.py```, and print the result to the terminal. 
To get the adjoint linear map of an input program, type
    
    ./cad -r --Pdiffu some-file.py
    
where ```-r``` is the reverse-mode option command.  
Caddiepy has the following options, which is listed by typing ```./cad --help```

    -r
      Apply reverse mode AD.
    --verbose
      Be verbose.
    -e ()
      Expression to be evaluated after loading of program files.
    --help
      Print usage information and exit.
    --version
      Print version information and exit.
    --Ptyped
      Print program after type inference.
    --Pexp
      Print internal expression program.
    --Ppointfree
      Print point free internal expression program.
    --Pdiff
      Print differentiated program.
    --Pdiffu
      Print unlinearised differentiated program.


# References

[1] Martin Elsman, Fritz Henglein, Robin Kaarsgaard, Mikkel K. Mathiesen, and Robert Schenck. Combinatory Adjoints and Differentiation. In Ninth Workshop on Mathematically Structured Functional Programming (MSFP 2022). Munich, Germany. April, 2022. [PDF](https://elsman.com/pdf/msfp22.pdf).
