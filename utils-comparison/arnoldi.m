function [V, H] = arnoldi(A, V, H, m)
	% [V, H] = arnoldi(A, V, H, m)
	% Constructs Krylov subspace basis with the Arnoldi algorithm
	%
	% Input:
	%	A		matrix
	%	V		starting vector or current orthonormal Krylov basis with k+1 columns
	%	H		(k+1) x k Hessenberg matrix that satisfies the Arnoldi relation
	%				(set H = [] if V is the starting vector)
	%	m		number of iterations of Arnoldi
	%
	% Output:
	% 	V		orthonormal Krylov basis with k+m+1 columns
	%	H		(k+m+1) x (k+m) Hessenberg matrix that satisfies the Arnoldi relation
	
	if size(V, 2) == 1
		V = V/norm(V);
		H = zeros(1, 0);
	end
	k = size(V, 2)-1;
	n = size(V, 1);
	V = [V, zeros(n, m)];
	H = [H, zeros(k+1, m);
		zeros(m, k+m)];

	W = V(:, k+1);
	% Arnoldi iterations
	for j = 1:m
		W = A*W;
		% mgs
		for i = 1:k+j
			H(i, k+j) = V(:, i)'*W;
			W = W - V(:, i)*H(i, k+j);
		end
		% % reorth
		% for i = 1:k+j
		% 	W = W - V(:, i)'*(V(:, i)*W);
		% end
		H(k+j+1, k+j) = norm(W);
		W = W / H(k+j+1, k+j);
		V(:, k+j+1) = W;
	end

end