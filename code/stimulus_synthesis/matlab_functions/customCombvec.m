function allMotions = customCombvec(a, b)
    % This function replicates MATLAB's combvec function without requiring the Deep Learning Toolbox.
    % It returns all combinations of the elements of 'a' and 'b' as columns of a matrix.

    [A, B] = ndgrid(a, b);  % Create grids for all combinations
    allMotions = [A(:)'; B(:)'];  % Flatten grids and concatenate into a single matrix
end
