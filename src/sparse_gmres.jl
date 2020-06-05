using SparseArrays, LinearAlgebra

include("matrix_basic.jl")
include("sparse_solvers.jl")

function restarted_gmres(A::SparseMatrixCSC{Float64,Int64},b::SparseVector{Float64},m::Integer,
			 L::SparseMatrixCSC{Float64,Int64},
			 U::SparseMatrixCSC{Float64,Int64})
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

		sparse_solve_vector_lt(L,r0)
		sparse_solve_vector_ut(U,r0)

		β = norm(r0,2)
		
		e1 = zeros(Float64,m)
		e1[1] = β
		Vm[:,1] = r0/β

		for k in 1:m-1
			w = sparse(A*Vm[:,k])
			sparse_solve_vector_lt(L,w)
			sparse_solve_vector_ut(U,w)
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
