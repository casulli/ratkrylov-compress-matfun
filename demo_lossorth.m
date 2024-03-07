% Compares loss of orthogonality of RKcompress and Lanczos
% Corresponds to Figure 3 in the paper

addpath("utils");
addpath("utils-comparison");

n = 2000;
lambda = -logspace(-4, 4, n);
D = diag(lambda);
[Q, ~] = qr(randn(n));
M = Q*D*Q';
[P, A] = hess(M);		% A is now tridiagonal
A = tril(A,1);			% remove numerical zeroes
A = (A+A')/2;			% ensure symmetry
b = randn(n, 1);
nrmb = norm(b);

options.tol = 0;			% to run for a fixed number of iterations
options.isreal = true;
% options.maxit = 400;

k = 25; 		% length poles exp
stepsize = 25;
nsteps = 18;
options.m = stepsize;

time = zeros(nsteps, 3);
err = zeros(nsteps, 3);
iter = zeros(nsteps, 3);

% Exact solution:
FAb = expm(A)*b;

for i = 1:nsteps
	fprintf("%d\n", i);
	infpoles = inf*ones(1, 2*stepsize*i);
	maxit = k + stepsize*(i-1);
	options.maxit = maxit;

	% RKcompress: 
	fprintf("RKcompress\n");
	[y, iter(i, 1)] = RKcompress_fAb(A, b, "exp", inf, @expm, options.tol, options);
	err(i, 1) = norm(y-FAb)/norm(FAb);

	% Lanczos:
	fprintf("lanczos\n");
	[y, iter(i, 2)] = lanczos_fAb(A, b, infpoles, @expm, options.tol, options);
	err(i, 2) = norm(y-FAb)/norm(FAb);

	fprintf("\n");
end

% Full orthogonalization Arnoldi
maxit = 350;
fprintf("arnoldi\n");
i = 1;
m = stepsize;
s = k;
fprintf("%d\n", s);
[V, H] = arnoldi(A, b, [], s);
haty = expm(H(1:s, 1:s)) * (eye(s, 1)*nrmb);
y = V(:, 1:s)*haty;
iter(i, 3) = s;
err(i, 3) = norm(y-FAb)/norm(FAb);
haty_old = zeros(0,1);
errest = norm(haty - [haty_old; zeros(s,1)]);
while (errest > options.tol * norm(haty) && s < maxit)
	haty_old = haty;
	[V, H] = arnoldi(A, V, H, m);
	s = s + m;
	i = i+1;
	fprintf("%d\n", s);
	haty = expm(H(1:s, 1:s)) * (eye(s, 1)*nrmb);
	errest = norm(haty - [haty_old; zeros(m,1)]);
	y = V(:, 1:s)*haty;
	iter(i, 3) = s;
	err(i, 3) = norm(y-FAb)/norm(FAb);
end
y = V(:,1:s)*haty;
err(i, 3) = norm(y-FAb)/norm(FAb);

% plot data
dlmwrite('output-data/lossorth_low-mem.dat', [iter(:, 1), err(:, 1)],'\t');
dlmwrite('output-data/lossorth_lanczos.dat', [iter(:, 2), err(:, 2)],'\t');
dlmwrite('output-data/lossorth_arnoldi.dat', [iter(1:i, 3), err(1:i, 3)],'\t');






