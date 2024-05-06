function [y, iter, errhist] = lanczos_fAb_block(A, b, f, maxit, tol)
% [y, iter, errhist] = lanczos_fAb_block(A, b, f, tol, options)
% Implementation of block Lanczos for f(A)*b for b block vector
% the solution is computed every iteration
%
% Input:
%	A			matrix or function handle that computes matvecs with A
%	b			block vector
%	f			function; we assume that f can compute matrix arguments
% 	tol			relative stopping tolerance
%   maxit		maximum number of iterations
%
% Output:
%	y			final approximate solution
%	iter		number of iterations
%	errhist		error norm history (estimates)

if ~isa(A, 'function_handle')
    mult = @(v) A*v;
end
k = maxit;
if nargout == 3
    errhist = [];
end

n = size(b,1);
bs = size(b,2);

% Run iterations of rational Lanczos:
W = zeros(n, (k+1)*bs);
alpha = cell(1, k+1);
beta = cell(1, k+1);
j_old = 0;
yhat_old = [];

for j = 1:k
    if j == 1
        [W(:, 1:bs),R] = qr(b,0);
        [W(:, bs+1:2*bs), alpha{1}, beta{1}] = short_recurrence_Arnoldi_in_block(mult, zeros(n,bs), W(:,1:bs), 0);
    elseif j == 2
        [W(:, 2*bs+1:3*bs), alpha{2}, beta{2}] = short_recurrence_Arnoldi_in_block(mult, W(:,1:bs), W(:,bs+1:2*bs), beta{1});
    else
        [W(:, j*bs+1:(j+1)*bs), alpha{j}, beta{j}] = short_recurrence_Arnoldi_in_block(mult, W(:,(j-2)*bs+1:(j-1)*bs), W(:,(j-1)*bs+1:j*bs), beta{j-1});
    end
    T = blkdiag(alpha{1:j});
    for i = 1: j-1
        T((i-1)*bs+1:i*bs,i*bs+1:(i+1)*bs) = beta{i}';
        T(i*bs+1:(i+1)*bs, (i-1)*bs+1:i*bs) = beta{i};
    end
    yhat = f(T)*[R; zeros((j-1)*bs, bs)];
    % Check convergence:
    errest = norm(yhat - [yhat_old; zeros((j - j_old)*bs, bs)],"fro");
    if nargout == 3
        errhist(end+1, :) = [j, errest/norm(yhat,"fro")];
    end
    if errest < tol*norm(yhat,"fro")
        y = W(:, 1:j*bs)*yhat;
        iter = j;
        return;
    else
        yhat_old = yhat;
        j_old = j;
    end
end
iter = k;
if ~exist('y', 'var')
    y = W(:, 1:j_old)*yhat;
end
end