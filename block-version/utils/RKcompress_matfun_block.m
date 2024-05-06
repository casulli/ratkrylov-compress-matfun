function [y, iter, k, m, errest] = RKcompress_matfun_block(mult, b, xi, f, options)
% [y, iter, k, m, errest] = RKcompress_matfun(mult, b, xi, f, options)
% Computes f(A)b for A Hermitian and b block vector, employing the
% low-memory block Lanczos algorithm (only polynomial) with rational
% Krylov compression described in [1]
%
% [1] A. A. Casulli , Block rational Krylov methods for matrix
% equations and matrix functions, Phd Thesis, 2024.
%
% see RKcompress_fAb.m for input and output details


n = size(b,1);
bs = size(b,2);
k = length(xi);

if nargin < 5
    options = [];
end
if isfield (options, 'maxit') == 0
    options.maxit = n;
end
if isfield (options, 'm') == 0
    m = k;
else
    m = options.m;
end

iter=0;
s_conv = 1;

errest = zeros(options.maxit, 2);		% error estimates
check_idx = 0;							% number of convergence checks

alpha = cell(1, k);
beta = cell(1, k);
y = zeros(n,bs);
zz = zeros(k*bs,bs);
V = zeros(n, (k+m+2)*bs);
[V(:, 1:bs), R] = qr(b,0);
T = [];
w = [];
c=eye((k+m)*bs,bs)*R;
z_conv = [];
maxnrmz = 0;

[V(:,bs+1:2*bs),alpha{1},beta{1}] = short_recurrence_Arnoldi_in_block(mult,zeros(n,bs),V(:,1:bs),0);
iter=iter+1;

[V(:,2*bs+1:3*bs),alpha{2},beta{2}] = short_recurrence_Arnoldi_in_block(mult,V(:,1:bs),V(:,bs+1:2*bs),beta{1});
iter=iter+1;

check_idx = check_idx + 1;
errest(check_idx, 1) = iter;
[check_conv, T, w, z_conv, errest(check_idx, 2)] = check_convergence_block(alpha(s_conv:2), beta(s_conv:2),...
    T, w, f, c(1:2*bs,1:bs), z_conv, options.tol, maxnrmz);
s_conv = 3;
if iter >= options.maxit
    fprintf("maximum number of iterations (%d) reached \n", iter)
    z = V(:,1:2*bs)*(z_conv);
    y = y + z;
    return
end
if check_conv
    z = V(:,1:2*bs)*(z_conv);
    y = y + z;
    return
end

for i=3:k-1
    [V(:,i*bs+1:(i+1)*bs),alpha{i},beta{i}] = short_recurrence_Arnoldi_in_block(mult,V(:,(i-2)*bs+1:(i-1)*bs),V(:,(i-1)*bs+1:i*bs),beta{i-1});
    iter=iter+1;

    check_idx = check_idx + 1;
    errest(check_idx, 1) = iter;
    [check_conv, T, w, z_conv, errest(check_idx, 2)] = check_convergence_block(alpha(s_conv:i), beta(s_conv:i),...
        T, w, f, c(1 : i*bs, 1:bs), z_conv, options.tol, maxnrmz);
    s_conv = i+1;
    if iter >= options.maxit
        fprintf("maximum number of iterations (%d) reached \n", iter)
        z = V(:,1:i*bs)*(z_conv);
        y = y + z;
        return
    end
    if check_conv
        z = V(:,1:i*bs)*(z_conv);
        y = y + z;
        return
    end
end

j=k-1;
[V(:, k*bs+1:(k+1)*bs),alpha{k},beta{k}] = short_recurrence_Arnoldi_in_block(mult,V(:,(k-2)*bs+1:(k-1)*bs),V(:,(k-1)*bs+1:k*bs),beta{k-1});

j = j+1;

iter=iter+1;
check_idx = check_idx + 1;
errest(check_idx, 1) = iter;
[check_conv, T, w, z_conv, errest(check_idx, 2)] = check_convergence_block(alpha(s_conv:k), beta(s_conv:k),...
    T, w, f, c(1 : k*bs,1:bs), z_conv, options.tol, maxnrmz);
if iter >= options.maxit
    fprintf("maximum number of iterations (%d) reached \n", iter)
    z = V(:,1:k*bs)*(z_conv);
    y = y + z;
    return
end
if check_conv
    z = V(:,1:k*bs)*(z_conv);
    y = y + z;
    return
end

s_conv = 1;
hatbeta = beta{k};
alpha = cell(1, m);
beta = cell(1, m);
j=j+1;
[V(:,(k+1)*bs+1:(k+2)*bs),alpha{1},beta{1}] = short_recurrence_Arnoldi_in_block(mult,V(:,(k-1)*bs+1:k*bs),V(:,k*bs+1:(k+1)*bs),hatbeta);
iter=iter+1;

check_idx = check_idx + 1;
errest(check_idx, 1) = iter;
[check_conv, T, w, z_conv, errest(check_idx, 2)] = check_convergence_block(alpha(s_conv:1), beta(s_conv:1),...
    T, w, f, c(1 : (k+1)*bs,1:bs), z_conv, options.tol, maxnrmz);
s_conv = 2;
if iter >= options.maxit
    fprintf("maximum number of iterations (%d) reached \n", iter)
    z = V(:,1:(k+1)*bs)*(z_conv);
    y = y + z;
    return
end
if check_conv
    z = V(:,1:(k+1)*bs)*(z_conv);
    y = y + z;
    return
end


while j < options.maxit && ~check_conv
    j=j+1;
    [V(:,(k+2)*bs+1:(k+3)*bs),alpha{2},beta{2}] = short_recurrence_Arnoldi_in_block(mult,V(:,k*bs+1:(k+1)*bs),V(:,(k+1)*bs+1:(k+2)*bs),beta{1});
    iter=iter+1;

    check_idx = check_idx + 1;
    errest(check_idx, 1) = iter;
    [check_conv, T, w, z_conv, errest(check_idx, 2)] = check_convergence_block(alpha(s_conv:2), beta(s_conv:2),...
        T, w, f, c(1 : (k+2)*bs,1:bs), z_conv, options.tol, maxnrmz);
    s_conv = 3;
    if iter >= options.maxit
        fprintf("maximum number of iterations (%d) reached \n", iter)
        z = V(:,1:(k+2)*bs)*(z_conv- [zz; zeros(size(z_conv,1)-size(zz,1),bs)]);
        y = y + z;
        return
    end
    if check_conv
        z = V(:,1:(k+2)*bs)*(z_conv- [zz; zeros(size(z_conv,1)-size(zz,1),bs)]);
        y = y + z;
        return
    end

    for i=3:m-1
        j=j+1;
        [V(:,(k+i)*bs+1:(k+i+1)*bs),alpha{i},beta{i}] = short_recurrence_Arnoldi_in_block(mult,V(:,(k+i-2)*bs+1:(k+i-1)*bs),V(:,(k+i-1)*bs+1:(k+i)*bs), beta{i-1});
        iter=iter+1;

        check_idx = check_idx + 1;
        errest(check_idx, 1) = iter;
        [check_conv, T, w, z_conv, errest(check_idx, 2)] = check_convergence_block(alpha(s_conv:i), beta(s_conv:i),...
            T, w, f, c(1 : (k+i)*bs,1:bs), z_conv, options.tol, maxnrmz);
        s_conv = i+1;
        if iter >= options.maxit
            fprintf("maximum number of iterations (%d) reached \n", iter)
            z = V(:,1:(k+i)*bs)*(z_conv- [zz; zeros(size(z_conv,1)-size(zz,1),bs)]);
            y = y + z;
            return
        end
        if check_conv
            z = V(:,1:(k+i)*bs)*(z_conv- [zz; zeros(size(z_conv,1)-size(zz,1),bs)]);
            y = y + z;
            return
        end
    end
    [V(:,(k+m)*bs+1:(k+m+1)*bs),alpha{m},beta{m}] = short_recurrence_Arnoldi_in_block(mult,V(:,(k+m-2)*bs+1:(k+m-1)*bs),V(:,(k+m-1)*bs+1:(k+m)*bs), beta{m-1});

    j = j+1;


    iter=iter+1;
    check_idx = check_idx + 1;
    errest(check_idx, 1) = iter;
    [check_conv, T, w, z_conv, errest(check_idx, 2)] = check_convergence_block(alpha(s_conv:m), beta(s_conv:m),...
        T, w, f, c(1 : (k+m)*bs,1:bs), z_conv, options.tol, maxnrmz);
    z1 = (z_conv- [zz; zeros(size(z_conv,1)-size(zz,1),bs)]);
    z = V*[z1; zeros(2*bs, bs)];
    y = y + z;
    if iter >= options.maxit
        fprintf("maximum number of iterations (%d) reached \n", iter);
        return
    end
    if check_conv
        return
    end

    % ---------------- compression

    maxnrmz = max(maxnrmz, norm(z1,"fro"));
    U = rational_krylov_block(T, w, xi, options);
    zz = (f(U'*T*U)*(U'*c));
    j=j+1;

    hatbeta = beta {m};
    alpha = cell(1, m);
    beta = cell(1, m);
    s_conv = 1;
    [V(:,(k+m+1)*bs+1:(k+m+2)*bs),alpha{1},beta{1}] = short_recurrence_Arnoldi_in_block(mult,V(:,(k+m-1)*bs+1:(k+m)*bs),V(:,(k+m)*bs+1:(k+m+1)*bs), hatbeta);
    iter=iter+1;
    V(:, 1:(k+2)*bs) = V*blkdiag(U, eye(2*bs));
    c(1:k*bs,1:bs) = U' * c;
    T = U'*T*U;
    w = U'*w;
    z_conv = zz;

    check_idx = check_idx + 1;
    errest(check_idx, 1) = iter;
    [check_conv, T, w, z_conv, errest(check_idx, 2)] = check_convergence_block(alpha(s_conv:1), beta(s_conv:1),...
        T, w, f, c(1 : (k+1)*bs,1:bs), z_conv, options.tol, maxnrmz);
    s_conv = 2;

    if iter >= options.maxit
        fprintf("maximum number of iterations (%d) reached \n", iter)
        z = V(:,1:(k+1)*bs)*(z_conv- [zz; zeros(size(z_conv,1)-size(zz,1),bs)]);
        y = y + z;
        return
    end
    if check_conv
        z = V(:,1:(k+1)*bs)*(z_conv- [zz; zeros(size(z_conv,1)-size(zz,1),bs)]);
        y = y + z;
        return
    end
end
end





