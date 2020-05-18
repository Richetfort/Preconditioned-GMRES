using SparseArrays, LinearAlgebra

function cs_augmented_A(A_cs::SparseMatrixCSC{Float64,Int64},p::Integer)
	#Will compute the pattern of augmented A according to p for ILU(p)
	for i in 1:A_cs.n
		k = A_cs.colptr[i] + 1
		while k < A_cs.colptr[i+1]
		
		end
	end
end

function cs_ilu(A_csr::SparseMatrixCSC{Float64,Int64})
	lu_csr::SparseMatrixCSC{Float64,Int64} = A_csr
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

function cs_scatter_lu(lu_cs::SparseMatrixCSC{Float64,Int64})
	
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
	#Return dense matrices, to be improved
	return LowerTriangular(L), UpperTriangular(U)
end

#=
n = 4

A = [1.0 1.0 0.0 0.0;
     1.0 2.0 0.0 0.0;
     1.0 0.0 -1.0 4.0;
     0.0 2.0 0.0 3.0]

A_cs = SparseMatrixCSC(A)

LU_csr = cs_ilu(SparseMatrixCSC(transpose(A)))

print(LU_csr)

L,U = cs_scatter_lu(LU_csr)

println(L*U)

println(A)
=#
