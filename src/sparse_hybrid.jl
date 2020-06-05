using SparseArrays, LinearAlgebra, MatrixMarket, PyPlot, Krylov

function diagIndex(A::SparseMatrixCSC{Float64,Int64},row::Integer)
	diag::Integer = A.colptr[row]
	while A.rowval[diag] != row
		diag = diag + 1
	end
	return diag
end

function hybrid_pattern(A::SparseMatrixCSC{Float64,Int64},τ::Float64,p::Int64)
	Aₚ = SparseMatrixCSC(transpose(A))
	p = min(p,A.n)
	diag::Integer = 0
	for i in 1:p
		diag = diagIndex(Aₚ,i)
		k = i+1
		while k <= A.m
			j = Aₚ.colptr[k]
			while Aₚ.rowval[j] < i
				j = j + 1
			end
			if Aₚ.rowval[j] == i
				Aₚ.nzval[j] /= Aₚ.nzval[diag]
				norm2Row = norm(Aₚ[k,Aₚ.rowval[j]:Aₚ.n],2)
				l = diag + 1
				while l < Aₚ.colptr[i+1]
					if iszero(Aₚ[Aₚ.rowval[l],k]) && abs(Aₚ.nzval[l] * Aₚ.nzval[j]) > τ*norm2Row
						Aₚ[Aₚ.rowval[l],k] = -(Aₚ.nzval[l] * Aₚ.nzval[j])
						A[k,Aₚ.rowval[l]] = 1.0e-60
					end
					l = l + 1
				end
			end
			k = k + 1
		end
	end
end
