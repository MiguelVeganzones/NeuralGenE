# TODO
## Learn cmake
https://code.visualstudio.com/docs/cpp/cmake-linux
1. cmake quick start
2. cmake scan for kits
3. cmake select a variant
4. cmake configure

## make a higher order function that generates mutation policies and evolve agents and mutation policies simultaneously

agents share a static mutation policy generator that creates a new policy based on current parameters and a mutation policy.

Agents will have their own mutation policy then

The reproduction guide will contain a metric for the distamce between memebrs.
    using agent_norm_calculator_type   = distance_matrix_type (*)(generation_container_type const&);

## Unit tests
## create a nullable type
## Add a stateless activation function version of layer to exploit [[pure]] and [[const]] functions.
## Issues
Define all functions outside class
## Conventions
### [GCC](https://gcc.gnu.org/wiki/CppConventions)
1. All data members should be private
2. All data members should have names ending with an underscore
3. Template parameter names should have a leading upper case
4. When defining a class, first define all public types, then all public constructors, then the public destructor, then all public methods. 
