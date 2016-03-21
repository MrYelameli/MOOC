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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h=sigmoid(X*theta);
cost=-y.*log(h)-(1-y).*log(1-h);
thetaExcludingZero=[[0];theta([2:length(theta)])];
J=(1/m)*sum(cost)+(lambda/(2*m))*sum(thetaExcludingZero.^2);
grad=(1/m).*(X'*(h-y))+(lambda/m)*thetaExcludingZero;






% % J=(-((y'*log(sigmoid(X(:,2:end)*theta(2:end))))+((1-y)'*log(1-sigmoid(X(:,2:end)*theta(2:end)))))/m)+...
% %     ((lambda/2*m)*sum(theta(2:end).^2));
% 
% 
% reg=(lambda\2*m)*sum(theta(2:28))^2;
% 
% %reg1=sum((lambda/m)*theta(2:28));
% 
% 
% J=-((y'*log(sigmoid(X*theta)))+((1-y)'*log(1-sigmoid(X*theta))))/m...
%     +(lambda/2*m)*sum(theta(2:28).^2);
% 
% %J=(-((y'*log(sigmoid(X*theta)))+((1-y)'*log(1-sigmoid(X*theta))))./m)+reg;
% 
% grad(1)=((sigmoid(X*theta)-y)'*X(:,1))./m;
% grad(2:28)=((sigmoid(X *theta)-y)'*X(:,2:28))'./m+(lambda/m)*theta(2:28);
% 

% ===========================================================

end
