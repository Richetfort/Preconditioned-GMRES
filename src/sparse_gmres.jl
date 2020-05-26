using SparseArrays, LinearAlgebra

include("utilities.jl")
include("matrix_basic.jl")
include("sparse_lu.jl")
include("sparse_solvers.jl")

function plu_restarted_gmres(A::SparseMatrixCSC{Float64,Int64},b::SparseVector{Float64},m::Integer,
			 L::SparseMatrixCSC{Float64,Int64},
			 U::SparseMatrixCSC{Float64,Int64})
	@time sparse_solve_matrix_lt(L,A)

#	println(A)
	
	@time sparse_solve_matrix_ut(U,A) #Bottleneck => an idea is to drop elements which are smaller than a given threshold

#	println(A)

        println("Κ(M) = ",cond(Array(A),2))

	sparse_solve_vector_lt(L,b)

	sparse_solve_vector_ut(U,b)

	return restarted_gmres(A,b,m)
end

function restarted_gmres(A::SparseMatrixCSC{Float64},b::SparseVector{Float64},m::Integer)

	nrow,ncol = size(A)
	nrow == ncol || error("A is not square.")
	n = nrow
	error = 0.001
	optimum = false

	Hm = Matrix{Float64}(undef,m,m-1)
	Vm = Matrix{Float64}(undef,n,m)

	x0 = zeros(Float64,n)
	x0[1] = 1.0

	iter = 1
	while(!optimum)

		r0 = b - A*x0
		β = norm(r0,2)
		
		e1 = zeros(Float64,m)
		e1[1] = β
		Vm[:,1] = r0/β

		for k in 1:m-1
			w = A*Vm[:,k]

			for i in 1:k
				Hm[i,k] = transpose(Vm[:,i])*w
				w -= Hm[i,k]*Vm[:,i]
			end

			Hm[k+1,k] = norm(w,2)

			if Hm[k+1,k] > 0
				Vm[:,k+1] = w/Hm[k+1,k]
			end
		end

		for i in 1:m-1
			ci = Hm[i,i]/norm(Hm[i:i+1,i],2)
			si = Hm[i+1,i]/norm(Hm[i:i+1,i],2)
			
			Hm[i,i] = Hm[i,i]*ci + Hm[i+1,i]*si
			Hm[i+1,i] = 0

			for j in i+1:m-1

				Temp1 = Hm[i,j]*ci + Hm[i+1,j]*si
				Temp2 = Hm[i+1,j]*ci - Hm[i,j]*si
				Hm[i,j] = Temp1
				Hm[i+1,j] = Temp2
			end

			Temp1 = e1[i]*ci + e1[i+1]*si
			Temp2 = e1[i+1]*ci - e1[i]*si

			e1[i] = Temp1
			e1[i+1] = Temp2

		end

		pop!(e1)
		matrix_backward(Hm[1:m-1,:],e1,m-1)
		x0 = x0 + Vm[:,1:m-1]*e1

		if(norm(b-A*x0,2)<= error)
			optimum = true
		else
			iter+=1
		end
	end
	return x0,iter,norm(b-A*x0,2)

end

function sparse_gmres(A_cs::SparseMatrixCSC,b_cs::SparseVector,n::Integer)
	m = 10 
	
	error = 0.01
	Hm = fill!(Matrix{Float64}(undef,m,m-1),0)
	x0 = fill!(Vector{Float64}(undef,n),0.0)

	x0[1] = 1.0

	r0 = Array(b_cs - A_cs*x0)
	β = norm(r0,2)
	e1 = fill!(Vector{Float64}(undef,m),0.0)
	e1[1] = β
	Vm = r0/β
	for k in 1:m-1
		w = A_cs*Vm[:,k]
		for i in 1:k
			Hm[i,k] = transpose(Vm[:,i])*w
			w -= Hm[i,k]*Vm[:,i]
		end
		Hm[k+1,k] = norm(w,2)
		if Hm[k+1,k] > 0
			Vm = hcat(Vm,w/Hm[k+1,k])
		end
	end
	for i in 1:m-1
		ci = Hm[i,i]/norm(Hm[i:i+1,i],2)
		si = Hm[i+1,i]/norm(Hm[i:i+1,i],2)
		for j in i:m-1
			Temp1 = Hm[i,j]*ci + Hm[i+1,j]*si
			Temp2 = Hm[i+1,j]*ci - Hm[i,j]*si
			Hm[i,j] = Temp1
			Hm[i+1,j] = Temp2
		end
		Temp1 = e1[i]*ci + e1[i+1]*si
		Temp2 = e1[i+1]*ci - e1[i]*si
		e1[i] = Temp1
		e1[i+1] = Temp2

	end
	ρm = Hm\(e1)
	return x0 + Vm[:,1:m-1]*ρm
end

A,b,n = system_from_mtx_file("/home/crichefo/Downloads/orsirr_1.mtx")
println("Κ(A) = ",cond(A,2))

#=A = [1.0 0.0 2.0 0.0 0.0 0.0;
     0.0 2.0 3.0 0.0 0.0 5.0;
     0.0 0.0 3.0 0.0 0.0 1.0;
     1.0 2.0 0.0 1.0 0.0 3.0;
     0.0 2.0 0.0 0.0 4.0 0.0;
     3.0 0.0 0.0 1.0 0.0 6.0]

b = [1.0;2.0;3.0;0.0;0.0;-1.0]

n = 6=#


println("========== Direct Method ==========")
@time x=A\b
println("2 last values of x : ",x[n-1:n],"\n")

A_cs = SparseMatrixCSC(A)
b_cs = SparseVector(b)

#Sparse A and b, 10 = Number of Krylov Subspaces Elements Computed
println("=========== RESTARTED GMRES ===========")

x,iter,β = @time restarted_gmres(A_cs,b_cs,10)

println("RESTARTED GMRES | Number of iteration : ",iter,"| β : ",β,"\n")
println("2 last values of x : ",x[n-1:n],"\n")

println("======> Preconditioning step : ")
lu_cs = @time sparse_ilut(A_cs,1.0e-5,5) #Incomplete LU factorization with threshold strategy ie. ILUT

#lu_cs = sparse_ilu(A_cs) #Incomplete LU factorization with 0 fill-in ie. ILU(0)
#p = 1
#lu_cs = sparse_ilu(A_cs,p) #Incommplete LU factorization with p fill-in ie. ILU(p)

L_cs,U_cs = sparse_scatter_lu_sparse(lu_cs) #Return sparse lower and upper matrices from a given matrix, here the lu_matrix

#Sparse A, Sparse b, 10 = Number of Krylov Subspaces Elements Computed, Sparse L and U
println("=========== Preconditioned RESTARTED GMRES ===========")

x,iter,β = plu_restarted_gmres(A_cs,b_cs,10,L_cs,U_cs) #This function erase A and b, to be fixed ?

println("Preconditioned RESTARTED GMRES | Number of iteration : ",iter,"| β : ",β,"\n")
println("2 last values of x : ",x[n-1:n],"\n")

