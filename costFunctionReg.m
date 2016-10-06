function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
temp = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
hypo = sigmoid(X*theta);
sumj = -y'*log(hypo)-(1-y)'*log(1-hypo);

J = (1/m)*sum(sumj) + (lambda/(2*m))*sum(theta(2:size(theta,1)).^2);

%pas besoin de somme puisque on travaille sur des variables vectorisés !!! si on met une somme, on n'a plus de vectorisation
grad = ((1/m)*(hypo-y)'*X)';
temp = theta;
temp(1) = 0;

grad = grad + (lambda/m)*temp;





% =============================================================

end
