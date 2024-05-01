function [cl,CC] = ccen(data, ML, CL, PARAM, CC)
% Constrained Clustering ENsemble (CCEN)

% Suppress warning
%#ok<*FNDSB>

% Online constrained single linkage
% The reference set is the previous window only

K = size(data,1); % window size
L = PARAM.L; % number of clusters
M = PARAM.EnsembleSize;

maxc = CC.MaxClusters; % maximum number of clusters

if ~isempty(ML)
    % To include all nodes in the graph, even those without
    % constraints, we need to add 1:K to the graph definition
    aux = 1:K;
    G = graph([aux';ML(:,1)], [aux';ML(:,2)]);
    tl = conncomp(G); % track labels in the window
else
    tl = 1:K;
end

NT = max(tl); % number of tracks in the window
% (conncomp in MATLAB returns labels 1, 2, 3, ...)

% Enforce ML
D = squareform(pdist(data)); % distance matrix
for i = 1:NT % for each track in w
    D(tl == i,tl == i) = 0; % obliterate distance within that cluster
end

% Enforce CL
for i = 1:size(CL,1)
    D(CL(i,1),CL(i,2)) = inf;
    D(CL(i,2),CL(i,1)) = inf;
end


% Cluster
Z = linkage(squareform(D),PARAM.BaseClusterer);

ensl = [];
for i = 1:M
    ensl = [ensl,cluster(Z,'MaxClust',L+i-1)];
end
cl = combine_cluster_labels(ensl);

if PARAM.vis
    cols = parula(maxc);
    plot_realtime(data,cl,cols)
    hold on
    axis equal
    legend off
    pause(0.001)
    hold off
end


CC.Ref = data;
CC.Lab = cl;
CC.MaxClusters = maxc;

end

function cl = combine_cluster_labels(B)

[N,k] = size(B);
A = zeros(N);
for i = 1:k
    p = B(:,i);
    M = bsxfun(@eq,p,p.');
    M(~p,~p) = 0;
    A = A + M;
end

% Convert back to cluster labels
cl = conncomp(graph(A>k/2))';
end