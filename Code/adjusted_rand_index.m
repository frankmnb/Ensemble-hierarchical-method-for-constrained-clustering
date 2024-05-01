function ar = adjusted_rand_index(A,B)
%=======================================================================
%jadjusted_rand_index.  Calculates the Adjusted Rand Index for matching 
%two label sets.
%
%   ar = adjusted_rand_index(A,B)
%
%   Input -----
%      'A': candidate labels (integers)
%      'B': true labels (integers)
%
%   Output -----
%      'ar': Adjusted Rand Index
%
%========================================================================

% (c) Lucy Kuncheva                                                 ^--^
% 20.07.2022 -----------------------------------------------------  \oo/
% -------------------------------------------------------------------\/-%

nA = numel(A);

ct = crosstab(A,B); % contingency table

N11 = sum(ct.*(ct - 1)/2,'all'); 
% number of pairs that are in the same cluster in both label sets

% Marginal sums
tA = tabulate(A);
msA = sum(tA(:,2).*(tA(:,2)-1)/2);
tB = tabulate(B);
msB = sum(tB(:,2).*(tB(:,2)-1)/2);
nP = nA * (nA - 1) / 2; % number of pairs

ar = (N11 - msA*msB/nP)/((msA + msB)/2 - msA*msB/nP);

