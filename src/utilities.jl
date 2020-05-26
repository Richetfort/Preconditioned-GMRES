function matrix_from_mtx_file(filename::String)
	file = open(filename)
	lines = readlines(file)
	popfirst!(lines)
	dim = split(lines[1])
	nrow = parse(Int64,dim[1])
	ncol = parse(Int64,dim[2])
	matrix = Matrix{Float64}(undef,nrow,ncol)
	popfirst!(lines)
	for ln in lines
		ln = split(ln)
		i = parse(Int64,ln[1])
		j = parse(Int64,ln[2])
		matrix[i,j] = parse(Float64,ln[3])
	end
	return matrix
end

function system_from_mtx_file(filename::String)
	#Generate linear system from a given mtx file containing header on line 1
	#The right-hand side is sparse and randomly generated
	density = 0.5
	file = open(filename)
	lines = readlines(file)
	popfirst!(lines)
	dim = split(lines[1])
	nrow = parse(Int64,dim[1])
	ncol = parse(Int64,dim[2])
	matrix = Matrix{Float64}(undef,nrow,ncol)
	vector = sprand(Float64,nrow,density)*100
	popfirst!(lines)
	for ln in lines
		ln = split(ln)
		i = parse(Int64,ln[1])
		j = parse(Int64,ln[2])
		matrix[i,j] = parse(Float64,ln[3])
	end
	return matrix,Vector(vector),nrow
end

function quick_split(v::Vector{Float64},index::Vector{Int64},p::Int64,left::Int64,right::Int64)
	#Incomplete Quick Sort Algorithm which sorts the p largest elements at the end of v
	#The index vector is sorting accordingly
	#Based on "Iterative Solvers for Sparse Linear Systems" by Youssef Saad p.326
	pivot = right
	i = left-1

	for j in left:right-1
		if abs(v[j]) < abs(v[pivot])
			i = i + 1
			swap = v[j]
			v[j] = v[i]
			v[i] = swap
			swap = index[j]
			index[j] = index[i]
			index[i] = swap
		end
	end

	val = v[pivot]
	ind = index[pivot]

	v[i+2:right] = v[i+1:right-1]
	index[i+2:right] = index[i+1:right-1]

	pivot = i + 1

	v[i+1] = val
	index[i+1] = ind

	if pivot < length(v) - p
		quick_split(v,index,p,pivot+1,right)
	elseif pivot > length(v) - p + 1
		quick_split(v,index,p,1,pivot-1)
	end	
end

#matrix,vector,n = system_from_mtx_file("data/orsirr_1.mtx")

