using LinearAlgebra

function matrix_multiplication(A::Array{Float64,2},B::Array{Float64,2},n::Integer,m::Integer,l::Integer)
	AB = Array{Float64,2}(undef,n,l)
	for i in 1:n
		for j in 1:l
			AB[i,j] = 0.0
			for k in 1:m
				AB[i,j] = AB[i,j] + A[i,k]*B[k,j]
			end
		end
	end
	return AB
end

function matrix_solve_matrix(L::LowerTriangular{Float64},A::Matrix{Float64},n)
	#Compute M = L^{1} × A without reversing L.
	#Since L is Lower Triangular, it is possible to compute n forwards.
	M = Matrix{Float64}(undef,n,n)
	for k in 1:n
		M[:,k] = A[:,k]
		for i in 1:n
			s = 0.0
			for j in 1:i-1
				s = s + L[i,j]*M[j,k]
			end
			M[i,k] = (M[i,k] - s)/L[i,i]
		end
	end
	return M
end

function matrix_solve_matrix(U::UpperTriangular{Float64},A::Matrix{Float64},n)
	#Compute M = U^{1} × A without reversing U.
	#Since U is Upper Triangular, it is possible to compute n backwards.
	M = Matrix{Float64}(undef,n,n)
	M[:,:] = A[:,:]
	for k in 1:n	
		for i in n:-1:1
			s = 0.0
			for j in n:-1:i+1
				s = s + U[i,j]*M[j,k]
			end
			M[i,k] = (M[i,k] - s)/U[i,i]
		end
	end
	return M
end

function matrix_forward(A::AbstractMatrix,b::Vector{Float64},n::Integer)
	for i in 1:n
		s=0.0
		for j in 1:i-1
			s = s + A[i,j]*b[j]
		end
		b[i] = (b[i] - s)/A[i,i]
	end
end

function matrix_backward(A::AbstractMatrix,b::Vector{Float64},n::Integer)
	for i in n:-1:1
		s = 0.0
		for j in n:-1:i+1
			s = s + A[i,j]*b[j]
		end
		b[i] = (b[i] - s)/A[i,i]
	end
end

function print_matrix(M::Array{Float64,2},nrow::Integer,ncol::Integer,name::String)
	print("Matrix ",name," (",nrow,"x",ncol,"), :\n"),
	for i in 1:nrow
		print(M[i,:],"\n")
	end
end
