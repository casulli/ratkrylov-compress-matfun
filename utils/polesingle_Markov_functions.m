function xi = polesingle_Markov_functions(a, b, alpha, beta)
	% xi = polesingle_Markov_functions(a, b, alpha, beta)
	% Computes quasi-optimal single repeated pole for rational approximation of Markov functions from [1]
	% rational approximation on [a,b] for Markov functions with integration domain [alpha, beta]
	% 
	% [1] B. Beckermann, L. Reichel, Error estimates and evaluation of matrix functions via the Faber transform, SIAM J. Numer. Anal., 2009

	phi = @(x) (((2.*x-(b+a))/(b-a))-sqrt(((2.*x-(b+a))/(b-a)).^2-1));
	psi =  @(x) (b-a)/4*(x+1./x) + (b+a)/2;		% inverse of phi
	if abs(alpha) == Inf
		w = phi(beta) - sqrt(phi(beta)^2 - 1);
		xi = psi(w);
	else
		kappa = (phi(alpha)-phi(beta))/(1-phi(alpha)*phi(beta));
		yopt = -1/kappa - sqrt(1/kappa^2 - 1);
		w = (1 + phi(alpha)*yopt) / (phi(alpha) + yopt);
		xi = psi(w);
	end
	
end