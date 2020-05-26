using SparseArrays, LinearAlgebra

function sparse_recursive_elimination_tree(L::SparseMatrixCSC,b::SparseVector)
	#Recursive elimination tree algorithm for lower triangular systems
	#/!\ Stack overflow /!\
	χ = []
	w = fill!(Vector{Integer}(undef,L.n),0)
	for i in 1:length(b.nzind)
		if w[b.nzind[i]] != 1
			cs_dsfr(b.nzind[i],L,b,w,χ)
		end
	end
	return χ
end

function sparse_dsfr(j::Integer,L::SparseMatrixCSC,b::SparseVector,w::Vector{Integer},χ)
	w[j] = 1
	for i in L.colptr[j]:L.colptr[j+1]-1
		if w[L.rowval[i]] != 1
			cs_dsfr(L.rowval[i],L,b,w,χ)
		end
	end
	pushfirst!(χ,j)	
end

function sparse_elimination_tree_ut(U::SparseMatrixCSC,b::SparseVector)
	#Direct elimination tree algorithm for upper triangular systems
	w = zeros(Float64,length(U.nzval))
	χ = []
	queue = []
	for i in b.nzind[length(b.nzval)]:-1:b.nzind[1]
		if(w[i] != 1)
			current = i
			pushfirst!(queue,i)
			while !isempty(queue)
				current = queue[1]
				next = U.colptr[current+1] - 2
				while next >= U.colptr[current]
					if w[U.rowval[next]] != 1
						pushfirst!(queue,U.rowval[next])
						current = queue[1]
						next = U.colptr[U.rowval[next]+1]-2
					else
						next -= 1
					end
				end
				pushfirst!(χ,queue[1])
				w[queue[1]] = 1
				popfirst!(queue)
			end
		end
	end
	return χ
end


function sparse_elimination_tree_lt(L::SparseMatrixCSC,b::SparseVector)
	#Direct elimination tree algorithm for lower triangular systems
	w = zeros(Float64,length(L.nzval))
	χ = []
	queue = []
	for i in b.nzind[1]:b.nzind[length(b.nzval)]
		if(w[i] != 1)
			current = i
			pushfirst!(queue,i)
			while !isempty(queue)
				current = queue[1]
				next = L.colptr[current] +1
				while next < L.colptr[current+1]
					if w[L.rowval[next]] != 1
						pushfirst!(queue,L.rowval[next])
						current = queue[1]
						next = L.colptr[L.rowval[next]]+1
					else
						next += 1
					end
				end
				pushfirst!(χ,queue[1])
				w[queue[1]] = 1
				popfirst!(queue)
			end
		end
	end
	return χ
end

function sparse_solve_matrix_ut(U::SparseMatrixCSC,A::SparseMatrixCSC)
	@inbounds for k in 1:U.n
		χ = sparse_elimination_tree_ut(U,A[:,k])
		@inbounds for i in 1:length(χ)
			A[χ[i],k] = A[χ[i],k] / U.nzval[U.colptr[χ[i]+1]-1]
			@inbounds for p::Int64 in U.colptr[χ[i]+1]-2:-1:U.colptr[χ[i]]
				A[U.rowval[p],k] -= U.nzval[p]*A[χ[i],k]
			end
		end
		dropzeros!(A[:,k])
	end
	droptol!(A,1.0e-20) #Remove close to 0 entries to free memory, seems durty => Need to be improved
	#This dirty method seems to reduce M = U⁻¹L⁻¹A computation time
	#Isn't it the same approach than in ILUT?
end

function sparse_solve_matrix_lt(L::SparseMatrixCSC,A::SparseMatrixCSC)
	@inbounds for k in 1:L.n
		χ = sparse_elimination_tree_lt(L,A[:,k])
		@inbounds for i in 1:length(χ)
			A[χ[i],k] = A[χ[i],k] / L.nzval[L.colptr[χ[i]]]
			@inbounds for p in L.colptr[χ[i]]+1:L.colptr[χ[i]+1]-1
				A[L.rowval[p],k] -= L.nzval[p]*A[χ[i],k]
			end
		end
		dropzeros!(A[:,k])
	end
	droptol!(A,1.0e-20) #Same observation than above
end

function sparse_solve_vector_ut(U::SparseMatrixCSC,b::SparseVector)
	χ = sparse_elimination_tree_ut(U,b)
	for i in 1:length(χ)
		b[χ[i]] = b[χ[i]] / U.nzval[U.colptr[χ[i]+1]-1]
		for p in U.colptr[χ[i]+1]-2:-1:U.colptr[χ[i]]
			b[U.rowval[p]] -= U.nzval[p]*b[χ[i]]
		end
	end
	droptol!(b,0.0000001)
end

function sparse_solve_vector_lt(L::SparseMatrixCSC,b::SparseVector)
	χ = sparse_elimination_tree_lt(L,b)
	for i in 1:length(χ)
		b[χ[i]] = b[χ[i]] / L.nzval[L.colptr[χ[i]]]
		for p in L.colptr[χ[i]]+1:L.colptr[χ[i]+1]-1
			b[L.rowval[p]] -= L.nzval[p]*b[χ[i]]
		end
	end
	droptol!(b,0.0000001)
end
#=
U = [1.0 0.0 1.0 3.0 0.0 1.0;
     0.0 2.0 0.0 4.0 0.0 2.0;
     0.0 0.0 1.0 0.0 3.0 0.0;
     0.0 0.0 0.0 2.0 0.0 0.0;
     0.0 0.0 0.0 0.0 1.0 3.0;
     0.0 0.0 0.0 0.0 0.0 4.0]

A = [1.0 0.0 0.0 2.0 0.0 0.0;
     4.0 6.0 3.0 0.0 0.0 1.0;
     0.0 0.0 2.0 0.0 0.0 0.0;
     5.0 2.0 0.0 2.0 0.0 3.0;
     1.0 2.0 0.0 0.0 4.0 0.0;
     1.2 0.3 0.0 0.0 0.0 6.0]

b = [0.0;1.0;0.0;0.0;0.0;0.0]

L = transpose(U)

A_cs = SparseMatrixCSC(A)

U_cs = SparseMatrixCSC(U)

L_cs = SparseMatrixCSC(L)

b_cs = SparseVector(b)

#@time χ = sparse_elimination_tree_lt(U_cs,b_cs)

#println(χ)

@time sparse_solve_matrix_ut(U_cs,A_cs)

println(A_cs)=#
#=
println(Array(A_cs))

@time x = inv(L)*A

println(x)=#
