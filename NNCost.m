function [cost, gradient] = NNCost(model, params)

    %% implementing batch here
    batch_size = 1;
    subset = randi(model.m_train, batch_size, 1);
    
    %% settings
    X = model.X_train(subset, :);
    Y = model.Y_train(subset, :);
    %X = model.X_train;
    %Y = model.Y_train;
    
    lambda = model.lambda;
    m = size(X, 1);
    n = model.n;
    n_f = model.num_filters;
    g = model.activation; % activation function
    cost = 0;
    
    b1_idx = n*n_f+1;
    w_idx = b1_idx+n*n_f;
    b2_idx = w_idx+n_f;
    
    filters = params(1:n*n_f);
    filters = reshape(filters, n_f, n); % n_f x n
    b_1 = params(b1_idx:b1_idx+n*n_f-1); 
    b_1 = reshape(b_1, n_f, n); % n_f x n
    w = params(w_idx:w_idx+n_f-1); % n_f x 1
    b_2 = params(b2_idx:b2_idx+n-1); % n x 1
    
    d_f = zeros(size(filters));
    d_b1 = zeros(size(b_1));
    d_w = zeros(size(w));
    d_b2 = zeros(size(b_2));
    
    for i = 1:m
        x = X(i, :)'; % n x 1
        y = Y(i, :)';
        a_2 = zeros(n_f, n); % n_f x n
        a_2_d = zeros(n_f, n);
        
        % conv layer
        for j = 1:n_f
            z_2 = circularConvolution(x, filters(j, :)') + b_1(j, :)'; % n x 1
            [a_2(j, :), a_2_d(j, :)] = g(z_2); % n x 1
        end
        
        % hidden layer
        z_3 = w' * a_2 + b_2'; % 1 x n
        [a_3, a_3_d] = g(z_3); % 1 x n
        h = a_3;
        % reg gradient is wrong
        % L2 loss
        cost = cost + sum((h' - y) .^ 2, 1);

        % compute gradient
        d_a4 = h - y'; % 1 x n
        
        delta_3 = d_a4 .* a_3_d; % 1 x n
        delta_2 = (w * delta_3) .* a_2_d; % n_f x n
        
        for j = 1:n_f
            d_f(j, :) = d_f(j, :) + circularConvolution(x, delta_2(j, :)')';
        end
        
        % accumulate gradient
        d_w = d_w + a_2 * delta_3';
        d_b2 = d_b2 + delta_3';
        d_b1 = d_b1 + delta_2;

    end
    
    % add regularization and scaling
    cost = (cost + lambda*sum(params .^ 2)) / (2 * m);
    
    d_f = (d_f + lambda*sum(filters, 'all')) / m;
    d_b1 = (d_b1 + lambda*sum(b_1, 'all')) / m;
    d_w = (d_w + lambda*sum(w, 'all')) / m;
    d_b2 = (d_b2 + lambda*sum(b_2, 'all')) / m;
    
    gradient = [d_f(:); d_b1(:); d_w(:); d_b2(:)];
end