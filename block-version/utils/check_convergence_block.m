function [check, T, w, z_conv2, relerrest] = check_convergence_block(alpha, beta, T, w, f, c, z_conv, tol, maxnrm)
	% Checks convergence and updates T within RKcompress

	m = length(alpha);
    bs = size(alpha{m},1);
	k = size(T,1)/bs;
	T = blkdiag(T, blkdiag(alpha{:}));
    T(1:k*bs, k*bs+1:k*bs+size(w,2)) = w;
    T(k*bs+1:k*bs+size(w,2), 1:k*bs) = w';
    for i = 1:m-1
	T((k+i-1)*bs+1:(k+i)*bs,(k+i)*bs+1:(k+i+1)*bs) = beta{i}';
	T((k+i)*bs+1:(k+i+1)*bs,(k+i-1)*bs+1:(k+i)*bs) = beta{i};
    end
    w = [zeros((k+m-1)*bs,bs); beta{m}'];
	z_conv2 = f(T)*c;
	relerrest = norm(z_conv2 - [z_conv ; zeros(m*bs,bs)],"fro") / max(norm(z_conv2,"fro"),maxnrm);
	if (relerrest < tol)
		check = true;
	else
		check = false;
	end
end