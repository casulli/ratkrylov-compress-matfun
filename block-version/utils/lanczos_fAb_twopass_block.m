function [y, iter, errhist] = lanczos_fAb_twopass_block(A, b, f, maxit, tol, options)
% [y, iter, errhist] = lanczos_fAb_twopass_block(A, b, f, maxit, tol, options)
% Implementation of block Lanczos for f(A)*b with two-pass strategy
% the solution is only computed in every iteration.
%
% Input:
%	A			matrix or function handle that computes matvecs with A
%	b			vector
%	f			function; we assume that f can compute matrix arguments
% 	tol			relative stopping tolerance
%   maxit		maximum number of iterations
%	options		struct with fields:
%		options.fast2pass		boolean; if true, use fast second pass implementation that does not recompute orthogonalization coefficients (default = true)
%
% Output:
%	y			final approximate solution
%	iter		number of iterations
%	errhist		error norm history (estimates)

if ~isa(A,'function_handle')
    mult = @(v) A*v;
end
k = maxit;
n = size(b,1);
bs = size(b,2);
if nargin < 6
    options = [];
end
if isfield(options, 'fast2pass') == 0
    options.fast2pass = true;
end
if nargout == 3
    errhist = [];
end

% First pass:
W = zeros(n, 3*bs);		% contains only last three columns of the Krylov basis
alpha = cell(1, k+1);
beta = cell(1, k+1);
j_old = 0;
yhat_old = [];

for j = 1:k
    if j == 1
        [W(:, bs+1:2*bs),R] = qr(b,0);
        [W(:, 2*bs+1:3*bs), alpha{1}, beta{1}] = short_recurrence_Arnoldi_in_block(mult, zeros(n,bs), W(:,bs+1:2*bs), 0);
    elseif j == 2
        [W(:, 2*bs+1:3*bs), alpha{2}, beta{2}] = short_recurrence_Arnoldi_in_block(mult, W(:,1:bs), W(:,bs+1:2*bs), beta{1});
    else
        [W(:, 2*bs+1:3*bs), alpha{j}, beta{j}] = short_recurrence_Arnoldi_in_block(mult, W(:,1:bs), W(:,bs+1:2*bs), beta{j-1});
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
        errhist(end+1, :) = [j, errest/ norm(yhat,"fro")];
    end
    if errest < tol*norm(yhat,"fro")
        % Convergence reached: start second pass to compute solution
        iter = j;
        break;
    else
        yhat_old = yhat;
        j_old = j;
    end
    W(:, 1:2*bs) = W(:, bs+1:3*bs);		% discard old basis vector
end

iter = size(yhat,1)/bs;

% Second pass:
if options.fast2pass
    % Faster implementation:
    y = two_pass_reconstruction_block(mult, b, alpha(1:iter), beta(1:iter), yhat);
else
    % Basic implementation that recomputes everything:
    y = zeros(n, bs);
    for j = 1:iter-1
        % Recompute basis:
        if j == 1
            [W(:, bs+1:2*bs),~] = qr(b,0);
            y = y + W(:,bs+1: 2*bs)*yhat(1:bs,1:bs);
            [W(:, 2*bs+1:3*bs), alpha{1}, beta{1}] = short_recurrence_Arnoldi_in_block(mult, zeros(n,bs), W(:,bs+1:2*bs), 0);
        elseif j == 2
            [W(:, 2*bs+1:3*bs), alpha{2}, beta{2}] = short_recurrence_Arnoldi_in_block(mult, W(:,1:bs), W(:,bs+1:2*bs), beta{1});
        else
            [W(:, 2*bs+1:3*bs), alpha{j}, beta{j}] = short_recurrence_Arnoldi_in_block(mult, W(:,1:bs), W(:,bs+1:2*bs), beta{j-1});
        end
        y = y + W(:, 2*bs+1:3*bs)*yhat(j*bs+1:(j+1)*bs,1:bs);	% update solution
        W(:, 1:2*bs) = W(:, bs+1:3*bs);		% discard old basis vector
    end
end
end