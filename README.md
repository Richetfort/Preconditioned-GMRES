# Preconditioned-GMRES
Here is stored different preconditioning routines for the GMRES algorithm such as ILU(k), ILU(t,p) and an hybrid version of them.

Incomplete LU like algorithms are perfectly well explained in <i>Iterative Methods for Sparse Linear Systems</i> p. 321 written by <i>Youssef Saad</i>, and I recommand to read the relative chapter (named <i>Preconditioning Techniques</i>) for a good understanding of how these approaches work.

Everything has been developed by my own but (obviously) routines imported from <a href=https://docs.julialang.org/en/v1/stdlib/SparseArrays/index.html>SparseArrays</a> and <a href=https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/>LinearAlgebra</a>.

Feel free to use and comment on my work! :-)

## Prerequisites
* Julia
* SparseArrays.jl
* LinearAlgebra.jl
* MatrixMarket.jl

## Running the code

```shell
git clone https://github.com/Protoniac/Preconditioned-GMRES
cd Preconditioned-GMRES/src/
julia main.jl
```
By default, the code is ran on an Harwel-Boeing matrix <i>orsirr_1.mtx</i>.

I discussed with the Professor <i>Serge G. Petiton</i> and he adviced me to use the <a href=https://smg2s.github.io/>Parallel Generator of Non-Hermitian Matrices Computed from Given Spectra</a> he developed himself with <i><a href=https://github.com/brunowu>Xinzhe Wu<a></i>. This can be a very useful tool for benchmarking GMRES and Preconditioners given a set of eigenvalues, and will be called soon. 

## Output 
<p align='center'>
<img src="images/example.png"/>
</p>

Output returns performances of :
* Direct solver method x = A\b
* Restarted GMRES without preconditioning
* Restarted GMRES with the hybrid preconditioning algorithm explained below (but other preconditioners can be chosen).

## In coming ...

* Improve the Hybrid Incomplete LU factorization and think about other approaches
* Implement parallelism for ILU routines described in the chapter called <i>Parallel Preconditioners</i> of the book <i>Iterative Methods for Sparse Linear Systems</i>.
