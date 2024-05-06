function [V, H] = arnoldi_block(A, V, H, m, bs)
% [V, H] = arnoldi_block(A, V, H, m)
% Constructs Krylov subspace basis with the block Arnoldi algorithm
%
% Input:
%	A		matrix
%	V		starting vector or current orthonormal Krylov basis with (k+1)*bs columns
%	H		(k+1)*bs x k*bs block Hessenberg matrix that satisfies the Arnoldi relation
%				(set H = [] if V is the starting vector)
%	m		number of iterations of Arnoldi
%
%   bs      block size (default 1)
%
%
%
% Output:
% 	V		orthonormal Krylov basis with k+m+1 columns
%	H		(k+m+1)*bs x (k+m)*bs block Hessenberg matrix that satisfies the Arnoldi relation

if nargin <= 4
    bs = 1;
end
if size(V, 2) == bs
    [V,~] = qr(V,0);
    H = zeros(bs, 0);
end
k = size(V, 2)/bs-1;
n = size(V, 1);
V = [V, zeros(n, m*bs)];
H = [H, zeros((k+1)*bs, m*bs);
    zeros(m*bs, (k+m)*bs)];

W = V(:, k*bs+1:(k+1)*bs);
% Arnoldi iterations
for j = 1:m
    W = A*W;
    % mgs
    for i = 1:k+j
        H((i-1)*bs+1:i*bs,(k+j-1)*bs+1:(k+j)*bs) = V(:, (i-1)*bs+1:i*bs)'*W;
        W = W - V(:, (i-1)*bs+1:i*bs)*H((i-1)*bs+1:i*bs, (k+j-1)*bs+1:(k+j)*bs);
    end
    [W,H((k+j)*bs+1:(k+j+1)*bs, (k+j-1)*bs+1:(k+j)*bs)] = qr(W,0);
    V(:, (k+j)*bs+1:(k+j+1)*bs) = W;
end

end