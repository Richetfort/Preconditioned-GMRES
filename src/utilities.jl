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
	density = 0.5
	file = open(filename)
	lines = readlines(file)
	popfirst!(lines)
	dim = split(lines[1])
	nrow = parse(Int64,dim[1])
	ncol = parse(Int64,dim[2])
	matrix = Matrix{Float64}(undef,nrow,ncol)
	vector = sprand(Float64,nrow,density)
	popfirst!(lines)
	for ln in lines
		ln = split(ln)
		i = parse(Int64,ln[1])
		j = parse(Int64,ln[2])
		matrix[i,j] = parse(Float64,ln[3])
	end
	return matrix,Vector(vector),nrow
end

#matrix,vector,n = system_from_mtx_file("data/orsirr_1.mtx")

