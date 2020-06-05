using SparseArrays, LinearAlgebra, MatrixMarket
include("sparse_lu.jl")
include("sparse_gmres.jl")
include("sparse_hybrid.jl")

A = MatrixMarket.mmread("../data/orsirr_1.mtx")
b = sprand(Float64,A.n,0.5)*100.0

println("=====> Target Solution : ")
xDirect = @time Array(A)\Vector(b)
println(xDirect[A.n-2:A.n])


println("=====> Restarted-GMRES : ")
xGMRES,iter,β = @time restarted_gmres(A,b,10)
println(xGMRES[A.n-2:A.n],"\nIter = ",iter," | β = ",β)


println("=====> Preconditioning phase : ")
hybrid_pattern(A,1.0e-3,500)
L,U = @time sparse_scatter_lu_sparse(sparse_ilu(A))
droptol!(A,1.0e-59)

println("=====> Preconditioned Restarted-GMRES : ")
xGMRES,iter,β = @time restarted_gmres(A,b,10,L,U)
println(xGMRES[A.n-2:A.n],"\nIter = ",iter," | β = ",β)
