function poles = poles_Markov_functions(a, b, alpha, beta, m)
	% poles = poles_Markov_functions(a, b, alpha, beta, m)
	% Computes quasi-optimal poles for rational approximation of Markov functions from [1]
	% rational approximation with m poles on [a,b] for Markov functions with integration domain [alpha, beta]
	% 
	% [1] B. Beckermann, L. Reichel, Error estimates and evaluation of matrix functions via the Faber transform, SIAM J. Numer. Anal., 2009

	phi = @(x) (((2.*x-(b+a))/(b-a))-sqrt(((2.*x-(b+a))/(b-a)).^2-1));
	psi =  @(x) (b-a)/4*(x+1./x) + (b+a)/2; 	% inverse of phi
	if abs(alpha) == Inf
		kappa = -1/phi(beta);
	else
		kappa = (phi(alpha)-phi(beta))/(1-phi(alpha)*phi(beta));
	end
	% k = ((1-sqrt(1-kappa^2))/kappa)^2;
	k = ( kappa / ( 1 + sqrt(1-kappa^2) ) )^2;	% version with less cancellation
	T1 = @(x)(1+phi(beta)*x)/(x+phi(beta));
	T2 = @(x) (sqrt(k).*x-1)./(x-sqrt(k));
	T = @(x) T1(T2(x));							% corresponds to T^{-1} from [1]
	K = ellipke(k^2);
	poles = zeros(1,m);
	for i = 1:m
		poles(i) = sqrt(k) * ellipj(K*(m+1-2*i)/m,k^2); 
		% poles(i) corresponds to \hat{w}_i from [1]
		poles(i) = T(poles(i));
		poles(i) = psi(poles(i));
	end

end