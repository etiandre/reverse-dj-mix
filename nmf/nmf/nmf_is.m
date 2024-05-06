function [W, H, cost] = nmf_is(V, n_iter, W, H)

% Unpenalized Itakura-Saito NMF with multiplicative updates
%
% [W, H, cost] = nmf_is(V, n_iter, W_ini, H_ini)
%
% Input:
%   - V: positive matrix data (F x N)
%   - n_iter: number of iterations
%   - W_ini: dictionary init. (F x K)
%   - H_ini: activation coefficients init. (K x N)
%
% Outputs :
%   - W and H such that
%
%               V \approx W * H
%
%   - cost : IS divergence though iterations
%
% Copyright (C) 2010 C. Fevotte
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.


[F,N] = size(V);

cost = zeros(1,n_iter);

V_ap = W*H;
cost(1) = sum(V(:)./V_ap(:) - log(V(:)./V_ap(:))) - F*N;

for iter = 2:n_iter
    
    % Update H
    H = H .* ((W'*(V.*V_ap.^-2))./(W'*V_ap.^-1));
    V_ap = W*H;
    
    % Update W
    W = W .* ((V.*V_ap.^-2)*H')./(V_ap.^-1*H');
    V_ap = W*H;
    
    % Normalize
    scale = sum(W,1);
    W = W .* repmat(scale.^-1,F,1);
    H = H .* repmat(scale',1,N);
    
    % Compute cost
    cost(iter) = sum(V(:)./V_ap(:) - log(V(:)./V_ap(:))) - F*N;
    
end
