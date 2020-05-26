struct cs_sparse
	nrow::Integer
	ncol::Integer
	p::Vector{Int64}
	i::Vector{Int64}
	x::Vector{Float64}
	nbElements::Integer
	nbElementsMax::Integer
end

function cs_init_from_matrix(M::Array{Float64,2},nrow::Integer,ncol::Integer)
	cs_p,cs_i,cs_x = [],[],[]
	nbElements = 0
	push!(cs_p,1)
	for i in 1:ncol
		for j in 1:nrow
			if M[j,i] != 0
				nbElements = nbElements + 1
				push!(cs_x,M[j,i])
				push!(cs_i,j)
			end
		end
		push!(cs_p,length(cs_i)+1)
	end
	M_cs = cs_sparse(nrow,ncol,cs_p,cs_i,cs_x,nbElements,nrow*ncol)	
	return M_cs
end

function cs_print(M_cs::cs_sparse)
	print("Number of row : ",M_cs.nrow,"\n")
	print("Number of col : ",M_cs.ncol,"\n")
	print("Number of Elements : ",M_cs.nbElements,"\n")
	print("Number of Elements Maximum : ",M_cs.nbElementsMax,"\n")
	print("p : ",M_cs.p,"\n")
	print("i : ",M_cs.i,"\n")
	print("x : ",M_cs.x,"\n")
end

function cs_mxpy(M_cs::cs_sparse,x::Vector{Float64},y::Vector{Float64})
	for i in 1:M_cs.ncol
		for j in M_cs.p[i]:M_cs.p[i+1]-1
			y[M_cs.i[j]] = y[M_cs.i[j]] + M_cs.x[j]*x[i]
		end
	end
end

function cs_transpose(M_cs::cs_sparse)
	Tp = Vector{Int64}(undef,0)
	Ti = Vector{Vector}(undef,M_cs.nrow)
	Tx = Vector{Vector}(undef,M_cs.nrow)

	Mt_csi = Vector{Int64}(undef,M_cs.nbElements)
	Mt_csx = Vector{Float64}(undef,M_cs.nbElements)

	count::Integer = 1
	for i in 1:M_cs.nrow
		Ti[i] = []
		Tx[i] = []
	end
	for i in 1:M_cs.ncol
		for j in M_cs.p[i]:M_cs.p[i+1]-1
			push!(Ti[M_cs.i[j]],i)
			push!(Tx[M_cs.i[j]],M_cs.x[j])
		end
	end
	for i in 1:M_cs.nrow
		push!(Tp,count)
		for k in 1:length(Ti[i])
			Mt_csi[count] = Ti[i][k]
			Mt_csx[count] = Tx[i][k]
			count += 1
		end
	end
	push!(Tp,count)

	return cs_sparse(M_cs.nrow,M_cs.ncol,Tp,Mt_csi,Mt_csx,M_cs.nbElements,M_cs.nbElementsMax)
end

n = 3
A = Array{Float64,2}(undef,n,n)
A = [1.0 0.0 -1.0
     0.0 0.0 0.0
     2.0 3.0 0.0]

A_cs = cs_init_from_matrix(A,n,n)
cs_print(A_cs)
At_cs = cs_transpose(A_cs)
cs_print(At_cs)
