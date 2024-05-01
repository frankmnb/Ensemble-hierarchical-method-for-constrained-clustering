function [cl,CC] = constrained_spectral(data, ...
    ML, CL, PARAM, CC)

L = PARAM.L; % number of clusters

maxc = CC.MaxClusters; % maximum number of clusters

data = zscore(data);
my_var = var(data);
% The size of the data set
N = size(data,1);

% Compute the similarity matrix A using RBF kernel.
% Set diagonal entries to 0 for numerical consideration.
A = zeros(N,N);
for i = 1:N
    for j=(i+1):N
        A(i,j) = exp(-1*sum((data(i,:)-data(j,:)).^2./(2*my_var)));
        A(j,i) = A(i,j);
    end
end

% Compute the graph Laplacian.
D = diag(sum(A)); vol = sum(diag(D)); D_norm = D^(-1/2);
Lap = eye(N) - D_norm*A*D_norm;

% Constraints matrix
Q = zeros(N);
for i = 1:size(ML,1)
    Q(ML(i,1),ML(i,2)) = 1;
    Q(ML(i,2),ML(i,1)) = 1;
end
for i = 1:size(CL,1)
    Q(CL(i,1),CL(i,2)) = -1;
    Q(CL(i,2),CL(i,1)) = -1;
end

K = L;
U = csp_K (Lap, Q, D_norm, vol, K);
cl = kmeans(U,K);

CC.Ref = data;
CC.Lab = cl;
CC.MaxClusters = maxc;

end

% ----------------------------------------------------------------------
function U = csp_K (L, Q, D_norm, vol, K)

% @article{Wang_2012,
%    title={On constrained spectral clustering and its applications},
%    volume={28},
%    ISSN={1573-756X},
%    url={http://dx.doi.org/10.1007/s10618-012-0291-9},
%    DOI={10.1007/s10618-012-0291-9},
%    number={1},
%    journal={Data Mining and Knowledge Discovery},
%    publisher={Springer Science and Business Media LLC},
%    author={Wang, Xiang and Qian, Buyue and Davidson, Ian},
%    year={2012},
%    month=sep, pages={1–30} }

% CODE IS BY THE AUTHORS OF THE PAPER

% Constrained Spectral Clustering: The K-way version
%
% Input:
%   The normalized graph Laplacian, L;
%   The constraint matrix, Q;
%   The normalization matrix, D_norm = D^{-1/2};
%   The volume of the graph, vol;
%   The number of clusters, K;
% Ouput:
%   The relaxed cluster indicator vectors, U;



% number of nodes
N = size(L,1);

% set beta such that we have K feasible solutions
lam = svds(Q,2*K);
beta = (lam(K+1)+lam(K))/2-10^(-6);

Q1 = Q - beta*eye(N);

% solve the generalized eigenvalue problem
[vec,~] = eig(L,Q1);

% normalized the eigenvectors
for i = 1:N
    vec(:,i) = vec(:,i)/norm(vec(:,i));
end

% find feasible cuts
satisf = diag(vec'*Q1*vec);
I = find(satisf >= 0);

% sort the feasible cuts by their costs
cost = diag(vec(:,I)'*L*vec(:,I));
[~,ind] = sort(cost,'ascend');

% remove trivial cuts
i = 1;
while i <= numel(ind)
    if nnz(vec(:,I(ind(i)))>0)~=0 && nnz(vec(:,I(ind(i)))<0) ~= 0
        break;
    end
    i = i + 1;
end
ind(1:i-1) = [];

% output cluster indicators
ind = ind(1:min(length(ind),K-1));
cost = cost(ind);
U = vec(:,I(ind));
for i = 1:size(U,2)
    U(:,i) = D_norm * (U(:,i) * vol^(1/2)) * (1-cost(i));
end
end