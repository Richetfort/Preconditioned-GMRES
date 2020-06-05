using SparseArrays, LinearAlgebra

include("sparse_solvers.jl")

function lower_and_upper_nentries(A_cs::SparseMatrixCSC{Float64,Int64},index::Integer)

	nzl::Integer,nzu::Integer,diag = 0,0,A_cs.colptr[index]

	while A_cs.rowval[diag] != index
		diag = diag + 1
	end

	for j in A_cs.colptr[index]:diag-1
		nzl = nzl + 1
	end

	for j in diag+1:A_cs.colptr[index+1]-1
		nzu = nzu + 1
	end

	return nzl,nzu
end

function sparse_augmented_pattern(A_cs::SparseMatrixCSC{Float64,Int64},p::Integer)

	Aₚ = SparseMatrixCSC(transpose(A_cs))
	lev = SparseMatrixCSC(transpose(A_cs)) 
	fill!(lev.nzval,0.1)

	L_csr = spzeros(Float64,A_cs.m,A_cs.n)
	U_csc = spzeros(Float64,A_cs.m,A_cs.n)

	for i in 1:p

		L_csr,U_csc = sparse_scatter_lu_sparse(Aₚ)
		L_csr = SparseMatrixCSC(transpose(L_csr))

		for k in 1:A_cs.m
			for l in 1:A_cs.n
				j = L_csr.colptr[k]
				while(j < L_csr.colptr[k+1] && iszero(Aₚ[k,l]))
					if(!iszero(U_csc[L_csr.rowval[j],l]))
						Aₚ[k,l] = 1.0e-60
						lev[k,l] = 99
					end
					j = j + 1
				end
			end
		end
	end

	lev = Vector(lev.nzval)
	for i in 1:length(lev)
		if lev[i] == 0.1
			lev[i] = 0.0
		end
	end
	#We return Aₚ and lev in CSR format
	return Aₚ,lev
end

function sparse_ilut(A_cs::SparseMatrixCSC{Float64,Int64},τ::Float64,p::Int64)
	
	#ILUT Approach based on "Iterative Solvers for Sparse Linear Systems" by Youssef Saad p.321-327
	#∀j∈[1,A_cs.n] where wⱼ< τ×||w||₂ will be replaced by 0.0 ⇾ Gain in computation time
	#Only the number of nonzero + p largest + the diagonal entry in w will be keeped. → Gain in memory
	#This approach can inspire sparse_matrix_solver_ut/lt routines in sparse_solvers.jl, where elements below a given threshold would be droped
	#/!\ We here assume that diagonal entries are not equals to 0.0 /!\
	
	#IKJ version of the Gaussian elimination is used here
	
	w::Vector{Float64} = zeros(Float64,A_cs.m)

	l_pattern::Vector{Int64} = zeros(Int64,A_cs.n) #Store indexes corresponding to elements stored in l_val
	u_pattern::Vector{Int64} = zeros(Int64,A_cs.n) #Store indexes corresponding to elements stored in u_val

	u_val::Vector{Float64} = zeros(Float64,A_cs.n) #Store 0 in lower part and its elements in upper part
	l_val::Vector{Float64} = zeros(Float64,A_cs.n) #Store 0 in upper part and its elements in lower part

	index = Vector{Int64}(undef,A_cs.n)

	for i in 1:A_cs.n
		index[i] = i
	end

	w_cs::SparseVector{Float64} = spzeros(Float64,A_cs.m)

	lu_csr = SparseMatrixCSC(transpose(A_cs)) #Sparse L and U are stored in a single data structure in CSR format

	for i in 1:lu_csr.m #Loop line by line accross the sparse matrix lu_csr 
		
		#We first copy all nz elements of the line i into w
		for j in lu_csr.colptr[i]:lu_csr.colptr[i+1]-1
			w[lu_csr.rowval[j]] = lu_csr.nzval[j]
		end

		k = lu_csr.rowval[lu_csr.colptr[i]] #k is egual to index of the first element in line i
		while k < i
			
			#We look for the diagonal entry, assuming it is not equal to 0.0
			diag = lu_csr.colptr[k] 
			while lu_csr.rowval[diag] != k
				diag = diag + 1
			end
			
			#We store the pivot into the lower part of w
			w[k] = w[k] / lu_csr.nzval[diag]

			if abs(w[k]) < τ*norm(w,2) #If this pivot is smaller than a given threshold, than we drop it
				w[k] = 0.0
			else #Else we compute the upper part
				for j in diag+1:lu_csr.colptr[lu_csr.rowval[diag]+1]-1
					w[lu_csr.rowval[j]] -= w[k]*lu_csr.nzval[j]
				end
			end

			k = k + 1
		end

		nentries_l, nentries_u = lower_and_upper_nentries(lu_csr,i) #Are returned the nz elements in l and u
		nentries_l, nentries_u = min(i-1,nentries_l+p),min(n-i+1,nentries_u+p) #We then choose the number of elements to keep

		l_pattern[:] = index[:]
		l_val[1:i-1] = w[1:i-1] #l receives the lower part of w

		u_pattern[:] = index[:]
		u_val[i:A_cs.n] = w[i:A_cs.n] #u receives the upper part of w + diagonal entry
		
		diag_val = w[i] #We store the diag value to keep it even if it is small
		
		quick_split(l_val,l_pattern,nentries_l,1,length(l_val)) #nentries biggest elements are sorted at the end of l_val
		quick_split(u_val,u_pattern,nentries_u,1,length(u_val)) 

		w = zeros(Float64,A_cs.n) #We drop all elements in w
		
		for j in 1:nentries_l
			w[l_pattern[A_cs.n-j+1]] = l_val[A_cs.n-j+1] #We store the nentries largest elements 
		end
		for j in 1:nentries_u
			w[u_pattern[A_cs.n-j+1]] = u_val[A_cs.n-j+1]
		end

		w[i] = diag_val #We store the diagonal element to guarantee that it is still present at this end of line factorization

		w_cs = SparseVector(w) #We drop all zero entries 
		droptol!(w_cs,τ*norm(w_cs,2)) #We drop all elements smaller than the given threshold

		lu_csr[:,i] = w_cs #Don't forget that lu_csr is in CSR format, we need to store the sparse vector w_cs in lu_csr i th column

		l_val = zeros(Float64,A_cs.n) #All temporary vectors are reseted
		u_val = zeros(Float64,A_cs.n)

		w = zeros(Float64,A_cs.n)
	end

	return SparseMatrixCSC(transpose(lu_csr)) #We finally return lu_csr but in CSC format
end

function sparse_ilu(A_cs::SparseMatrixCSC{Float64,Int64},p::Integer)
	
	lu_csr,lev = sparse_augmented_pattern(A_cs,p) 

	for i in 2:lu_csr.n
		k = lu_csr.colptr[i]
		while k < lu_csr.colptr[i+1] && lu_csr.rowval[k] < i
			pivot = lu_csr.colptr[lu_csr.rowval[k]] 
			while lu_csr.rowval[pivot] != lu_csr.rowval[k]
				pivot = pivot + 1
			end
			lu_csr.nzval[k] = lu_csr.nzval[k]/lu_csr.nzval[pivot]
			j = k+1
			while j < lu_csr.colptr[i+1]
				lu_kj = lu_csr[lu_csr.rowval[j],lu_csr.rowval[k]]
				if !iszero(lu_kj) #Need to check if aₖⱼ∈ NZ(A₀), to be improved ?
					index = lu_csr.colptr[lu_csr.rowval[k]]
					while lu_csr.rowval[index] != lu_csr.rowval[j]
						index = index + 1
					end
					lu_csr.nzval[j] -= lu_csr.nzval[k] * lu_kj
					lev[j] = min(lev[j],lev[k]+lev[index]+1)
				end
				j = j + 1
			end
			k = k + 1
		end
	end
	for i in 1:length(lev)
		if lev[i] > p
			lu_csr.nzval[i] = 1.0e-60
		end
	end
	droptol!(lu_csr,1.0e-59)
	return SparseMatrixCSC(transpose(lu_csr))
end

function sparse_ilu(A_cs::SparseMatrixCSC{Float64,Int64})
	lu_csr::SparseMatrixCSC{Float64,Int64} = SparseMatrixCSC(transpose(A_cs))
	#This function compute ILU(0)
	#ILU(p) will be implemented
	#This function use CSR format, user need to give SparseMatrixCSC(transpose(A_cs)) in parameter
	#To be improved ?
	for i in 2:lu_csr.n
		k = lu_csr.colptr[i]
		while k < lu_csr.colptr[i+1] && lu_csr.rowval[k] < i
			pivot = lu_csr.colptr[lu_csr.rowval[k]] 
			while lu_csr.rowval[pivot] != lu_csr.rowval[k]
				pivot = pivot + 1
			end
			lu_csr.nzval[k] = lu_csr.nzval[k]/lu_csr.nzval[pivot]
			j = k+1
			while j < lu_csr.colptr[i+1]
				lu_kj = lu_csr[lu_csr.rowval[j],lu_csr.rowval[k]]
				if !iszero(lu_kj) #Need to check if aₖⱼ∈ NZ(A₀), to be improved ?
					lu_csr.nzval[j] -= lu_csr.nzval[k] * lu_kj
				end
				j = j + 1
			end
			k = k + 1
		end
	end
	return SparseMatrixCSC(transpose(lu_csr)) #Return LU in CSC format (in a single data structure however)
end

function sparse_scatter_lu_sparse(lu_cs::SparseMatrixCSC{Float64,Int64})

	#To be improved, L_cs and U_cs should respectively get the lower and upper pattern of lu_cs

	L_cs = spzeros(Float64,lu_cs.m,lu_cs.n)
	U_cs = spzeros(Float64,lu_cs.m,lu_cs.n)

	for p in 1:lu_cs.n
		i = lu_cs.colptr[p]
		while(i<lu_cs.colptr[p+1] && lu_cs.rowval[i]<=p)
			U_cs[lu_cs.rowval[i],p] = lu_cs.nzval[i]
			i = i + 1
		end
		while(i < lu_cs.colptr[p+1] && lu_cs.rowval[i]<=lu_cs.n)
			L_cs[lu_cs.rowval[i],p] = lu_cs.nzval[i]
			i = i + 1
		end
		L_cs[p,p] = 1.0
	end

	return L_cs,U_cs
end

function sparse_scatter_lu_dense(lu_cs::SparseMatrixCSC{Float64,Int64})
	
	L = Matrix{Float64}(undef,lu_cs.m,lu_cs.n)
	U = Matrix{Float64}(undef,lu_cs.m,lu_cs.n)

	for p in 1:lu_cs.n
		i = lu_cs.colptr[p]
		while(i<lu_cs.colptr[p+1] && lu_cs.rowval[i]<=p)
			U[lu_cs.rowval[i],p] = lu_cs.nzval[i]
			i = i + 1
		end
		while(i < lu_cs.colptr[p+1] && lu_cs.rowval[i]<=lu_cs.n)
			L[lu_cs.rowval[i],p] = lu_cs.nzval[i]
			i = i + 1
		end
		L[p,p] = 1.0
	end

	return LowerTriangular(L), UpperTriangular(U)
end

function sparse_lu_left_preconditioning(A::SparseMatrixCSC{Float64,Int64},b::SparseVector{Float64},L::SparseMatrixCSC{Float64,Int64},U::SparseMatrixCSC{Float64,Int64})
	
	M::SparseMatrixCSC{Float64,Int64} = sparse(A)
	v::SparseVector{Float64} = sparse(b)

	@time sparse_solve_matrix_lt(L,M)

	@time sparse_solve_matrix_ut(U,M) #Bottleneck

	sparse_solve_vector_lt(L,v)

	sparse_solve_vector_ut(U,v)

	return M,v
end
