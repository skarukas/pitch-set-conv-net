function displayConvergence(param_history, grad_history)
    %% perform PCA on param space

    % mean normalize
    mu = mean(param_history, 1);
    sigma2 = mean((param_history - mu), 1);
    param_history = (param_history - mu) ./ sigma2;

    [m_p, n_p] = size(param_history);

    covar = zeros(n_p);

    for i=1:m_p
        covar = covar + (param_history(i, :)' * param_history(i, :)) / m_p;
    end

    [U, S, V] = svd(covar);
    basis = U(:, 1:2); % n_p x 2
    mapping = param_history * basis; % m_p x 2

    %[dim1, dim2] = meshgrid(mapping(:, 1), mapping(:, 2));
    %% plot convergence
    figure;
    dim1 = mapping(:, 1);
    dim2 = mapping(:, 2);
    N = 5;
    for i=1:N:m_p
        plot3(dim1(i), dim2(i), grad_history(i), '.', 'Color', [1 - i/m_p 0 i/m_p]);
        hold on;
    end
    plot3(movmean(dim1, 10), movmean(dim2, 10), movmean(grad_history, 10), '-', 'LineWidth', 1, 'Color', [0.8 0.8 0.8]);
    plot3(movmean(dim1, 50), movmean(dim2, 50), movmean(grad_history, 50), '-', 'LineWidth', 1.5, 'Color', [0.4 0.4 0.4]);
    plot3(movmean(dim1, 200), movmean(dim2, 200), movmean(grad_history, 200), '-', 'LineWidth', 2, 'Color', 'k');
    grid on
    hold off
end