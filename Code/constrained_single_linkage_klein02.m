function [cl,CC] = constrained_single_linkage_klein02(data, ...
    ML, CL, PARAM, CC)

% Suppress warning
%#ok<*FNDSB>

% Online constrained single linkage
% The reference set is the previous window only

K = size(data,1); % window size
L = PARAM.L; % number of clusters

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
Z = linkage(squareform(D),"single");
cl = cluster(Z,'MaxClust',L);


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