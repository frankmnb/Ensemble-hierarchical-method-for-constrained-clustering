function [labels, centres, iter] = ...
    cop_kmeans_improved(data, ML, CL, PARAM, CC)

% COP-Kmeans implementation running step 2 of the algorithm in Table 1 [1]
% by point rather than as a batch.
%
% [1] Wagstaff01

maxiter = PARAM.maxiter;
initial_means = CC.Centres;

total_constraints = [ML;CL]; % all pairs of constraints

% 1. Let C_1...C_k be the initial cluster centres.-------------------------
me = initial_means;
% 1. ----------------------------------------------------------------------

number_of_clusters = size(me,1);

N = size(data,1);
old_labels = zeros(N,1);
new_labels = old_labels + 1;

iter = 0;

% 4. Iterate between (2) and (3) until convergence.------------------------

% 2. For each point d_i in D, assign it to the closest cluster C_j /such---
% that/ VIOLATE-CONSTRAINTS(d_i,C_j,ML,CL) /is false/. /If no such cluster-
% exists, fail (return {})./-----------------------------------------------
while any(old_labels~=new_labels) && iter<maxiter

    old_labels = new_labels;
    iter = iter + 1;

    e = pdist2(data,me); % distance for each index (r) to every cluster (c)
    [~,labels] = min(e,[],2); % labels (label of closest cluster centre)
    curr_labels = labels;
    old_me = me;

    used_points = zeros(size(data,1),1);

    rp = 1:N;
    for zz = 1:size(data,1)
        i = rp(zz);
        di = e(i,:); % distance from point (r) to CCs (c)
        [~,isorted] = sort(di); % distances to all means

        if find(total_constraints==i) % if constrained
            j = 1; % cluster index for loop
            not_done = true;

            while j <= number_of_clusters && not_done
                if ~violate_constraints(i,isorted(j),...
                        ML,CL,curr_labels,used_points)
                    % check constraints

                    curr_labels(i) = isorted(j);
                    not_done = false;
                end
                j = j + 1;
            end

        else % if no constraints assign closest mean
            curr_labels(i) = isorted(1);
        end

        used_points(i) = 1;

    end
    % 2. ------------------------------------------------------------------

    % 3. For each cluster C_i, update its center by averaging all of the---
    % points d_j that have been assigned to it.----------------------------
    nm = grpstats(data,curr_labels,"mean"); % new means
    uc = unique(curr_labels); % to avoid empty clusters - keep old means
    me(uc,:) = nm;
    % 3. ------------------------------------------------------------------

    if PARAM.vis
        cols = parula(size(me,1));
        plot_realtime(data,curr_labels,cols)
        hold on
        plot(old_me(:,1),old_me(:,2),'rx',MarkerSize=20)
        plot(me(:,1),me(:,2),'bo',MarkerSize=20)
        axis equal
        legend off
        pause(0.001)
        hold off
    end
    new_labels = curr_labels;
end


% 5. Return {C_1...C_k}.---------------------------------------------------
labels = curr_labels(:); centres = me;
% 5. ----------------------------------------------------------------------

end

function out = violate_constraints(point,cluster,ML,CL,...
    curr_labels, used_points)

used_points(point) = 1;
not_used = find(used_points == 0);
ML = ML(sum(ismember(ML,not_used),2)==0,:);
CL = CL(sum(ismember(CL,not_used),2)==0,:);

index1 = ML(:,1) == point; index2 = ML(:,2) == point;
index3 = CL(:,1) == point; index4 = CL(:,2) == point;

out = any(curr_labels(ML(index1,2)) ~= cluster) || ...
    any(curr_labels(ML(index2,1)) ~= cluster) || ...
    any(curr_labels(CL(index3,2)) == cluster) || ...
    any(curr_labels(CL(index4,1)) == cluster);

end