function y = softmax(z)
    z_exp = exp(z);
    y = z_exp ./ sum(z_exp, 'all');
end