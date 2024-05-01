clear, clc, close all

%#ok<*SAGROW>
%WARNING OFF

% PREREQUISITES ==========================================================
FolderData = '..\Versions\'; % Data files location
FolderConstraints = '..\Constrains\'; % Constraint files location

NamesSynth = readtable('AllDataNames.xlsx', 'sheet','Synth',...
    "ReadVariableNames", false);
NamesReal = readtable('AllDataNames.xlsx', 'sheet','Real',...
    "ReadVariableNames", false);

Type = 'Synth'; % Select dataset collection

Methods = {'ccen'};

Ms = {'average', 'single','complete'}; % Base clusterers

propC = [0, 1, 2, 3, 4, 5, 10, 15, 20];

% ========================================================================

% Read the data table
NamesChosen = table2array(eval(['Names',Type]));

PARAM.vis = 0;
PARAM.maxiter = 20;

for gg = 1:3
    PARAM.BaseClusterer = Ms{gg};

    for EnsembleSize = 1:6
        PARAM.EnsembleSize = EnsembleSize;
        for da = 1:numel(NamesChosen)

            % READ THE DATA -----------------------------------------------
            dataset = NamesChosen{da}; % filename

            for rep = 0:4
                fprintf('Rep %i... ', rep)
                fn = [FolderData,dataset,'_',num2str(rep),'.csv'];
                z = readmatrix(fn);

                if strcmp(Type,'Video')
                    z = z(:,1:end-1);
                end

                d = z(:,1:end-1); % data

                % PCA
                [~,sco] = pca(d);
                d = sco(:,1:2);

                l = z(:,end); % labels

                PARAM.sigma_ini = max(pdist(d))/10; 
                % recommended in [Engel10]
                PARAM.L = max(l); 
                % Initialise with the true number of clusters
                L = PARAM.L;

                for pr = 1:numel(propC)
                    CC = [];
                    if pr == 1
                        CL = []; ML = []; % no constraints
                    else
                        % READ CONSTRAINTS --------------------------------
                        constr = readmatrix([FolderConstraints, ...
                            dataset,'_', ...
                            num2str(rep),'_',num2str(propC(pr)),'.csv']);
                        indexco = constr(:,3) == 1;
                        ML = constr(indexco,[1,2])+1;
                        CL = constr(~indexco,[1,2])+1;
                    end


                    % INITIALISE THE METHODS ------------------------------
                    % Find initial CC structure

                    CC.Centres = d(1:L,:);

                    for i = 1:PARAM.L
                        CC.Covs{i} = PARAM.sigma_ini^2 * eye(size(d,2));
                        CC.Priors(i) = 1/L;
                        CC.Sp(i) = 1;
                    end
                    CC.MaxClusters = L;

                    % RUN THE METHODS -------------------------------------
                    for me = 1:numel(Methods)
                        method = Methods{me};
                        tic
                        al = feval(method, d, ML, CL, PARAM, CC);
                        TIME(rep+1,da,me,pr) = toc;

                        % STORE THE INDEX VALUES --------------------------
                        NMI(rep+1,da,me,pr) = ...
                            normalised_mutual_information(al,l);
                        ARI(rep+1,da,me,pr) = adjusted_rand_index(al,l);
                    end
                end
            end
            fprintf('Dataset %2i %s done.\n',da,dataset)
        end

        % SAVE RESULTS ----------------------------------------------------
        dt = datetime;
        datetoday = date; %#ok<*DATE>
        fn = ['E_',Type, '_', PARAM.BaseClusterer, ...
            '_EnsSize', num2str(EnsembleSize),'.mat'];
        save(fn)
    end
end