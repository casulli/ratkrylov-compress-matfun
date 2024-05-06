function x = two_pass_reconstruction_block(mult, b, alpha, beta, y)
% Reconstructs solution for two-pass Lanczos efficiently in the second pass

bs = size(b,2);
[v2, ~] = qr(b,0);
tildey = mult(v2);
w = tildey - v2*alpha{1};
v = w / beta{1};
x = v2*y(1:bs,1:bs) + v*y(bs+1:2*bs,1:bs);
v1 = v2;
v2 = v;
beta1 = beta{1};
for i = 3:size(y,1)/bs
    tildey = mult(v2)-v1*beta1';
    w = tildey - v2*alpha{i-1};
    v = w / beta{i-1};
    x = x + v*y((i-1)*bs+1:i*bs,1:bs);
    v1 = v2;
    v2 = v;
    beta1 = beta{i-1};
end
end