% Note: this code used this github project as a reference: 
% https://github.com/neuromechanist/fastICA_EMG_decomp
% steady force: 27 MUs
% increasing force 1: 27MUs
% increasing force 2: 44MUs

clc, clear, close all;

emg = load("data/increasing force 2.mat");
emg = emg.out_mat';

% Preoprocessing Filtering
sr = 4000;
fs = 2000;
emg_new = [];
for i = 1:64
    emg_new = [emg_new, resample(emg(:,i), fs, sr)];
end

emg = emg_new;

fpass = [20, 500];
wo = 50/(fs/2);  
bw = wo/50; % Q=30

[b,a] = iirnotch(wo,bw);
emg_f = bandpass(emg_new,fpass,fs);
emg_f = filter(b, a, emg_f);

K = 50;
L = length(emg_f);
M = floor(L/K);
x  = emg(:,1);
y = emg_f(:,1);
period_xs = [];
period_ys = [];
for i = 1:K
   phi_xx = xcorr(x(1+(i-1)*M: i*M), "biased");
   phi_yy = xcorr(y(1+(i-1)*M: i*M), "biased");
   period_xs = [period_xs abs(fft(phi_xx))];
   period_ys = [period_ys abs(fft(phi_yy))];
end
%
period_x = mean(period_xs,2);
period_y = mean(period_ys,2);
% 
phh = log(period_y) - log(period_x);
df = fs/length(phh);
f = [0:df:df*length(phh)/2];
%
figure()
plot(f, phh(1:length(phh)/2+1))

title("H(w)^2")

% preprocessing process
R = 4; % Extension parameter
[emg_extend,W] = SimEMGProcessing(emg_f,R);  
% emg_extend: the extended matrix. size: [NumCh1*(R+1), N]


% source separation algorithm: fastICA
% assumptions: statistically independent; non-Gaussian
M = 100; % FastICA iteration number
[s,B,SpikeTraintemp] = FastICA(emg_extend,M);

% post-processing process
fs_emg=fs; % frequency of the emg signal
[SpikeTrain,GoodIndex] = MUReplicasRemoval(SpikeTraintemp,s,fs_emg); 
SpikeTrainGood = SpikeTrain(:,GoodIndex);
sGood=s(:,GoodIndex);

%% calculate the silhoutte value and plot
SIL = SILCal(sGood,fs_emg); 

plot_spikeTrain(SpikeTrainGood, fs_emg, SIL, 0);

%%
[~, N] = size(SpikeTrainGood);
colors = turbo(N);
figure()
for i = 1:N
    plot_instantaneous_frequency(SpikeTrainGood, fs, colors(i,:), i); hold on;
end
hold off;

xlabel("time (sec)");
ylabel("MU firing rates");
title("MU firing rates increasing force 2");
saveas(gcf, 'results/firing_increasing_2.png');


% Preprocessing: extension, centralization and whitening
function [EMGOutput,W] = SimEMGProcessing(EMGInput,R)
[N,NumCh1] = size(EMGInput);

% Extend the matrix
% Temporal Correlation Capture: EMG signals typically exhibit temporal 
% correlation among multiple components, indicating the presence of time-delay 
% relationships between different components. By duplicating and delaying the 
% original signal, the expanded matrix can better capture this temporal 
% correlation, enabling the decomposition algorithm to more accurately 
% separate different components.
EMGExtended = zeros(N,NumCh1*(R+1));
EMGExtended(:,1:NumCh1) = EMGInput;
if R~=0
    for i = 1:R
        EMGExtended(1+i:end,NumCh1*i+1:NumCh1*i+NumCh1) = EMGInput(1:end-i,:);
    end
end

% centralize the data
EMGExtendedSubMean = EMGExtended - ones(N,1)*mean(EMGExtended,1);

% withen the data
Rxx = EMGExtendedSubMean'*EMGExtendedSubMean;
[V,D] = eig(Rxx);
fudgefactor = 0;
W = real((V*diag(1./(diag(D)+fudgefactor).^(1/2))*V'))';
EMGOutput = W * EMGExtendedSubMean';
end


% FastICA algo: find the independent components from the EMG source
function [s,B,SpikeTrain] = FastICA(EMG,M)
% EMG (NumCh * N): Numch is the number of channels, N is the length of signal.
% M: number of total iteration
% s: separated source after decomposition
% B: separation vector
% SpikeTrain: motor unit spike trains after decomposition
[NumCh,N] = size(EMG);
Tolx = 10^-4;
s = zeros(N,M);
B = zeros(NumCh,1);
SpikeTrain = zeros(N,M);

for i = 1:M
    w = [];
    w(:,1) = randn(NumCh,1);
    w(:,2) = randn(NumCh,1);
    for n = 2:100
        if abs(w(:,n)'*w(:,n-1)-1)>Tolx
            A = mean(2*w(:,n)'*EMG);
            % using negentropy to update independent components
            w(:,n+1) = EMG*(((w(:,n)'*EMG)').^2)-A*w(:,n);
            w(:,n+1) = w(:,n+1) - B*B'*w(:,n+1);
            w(:,n+1) = w(:,n+1)/norm(w(:,n+1));
        else
            break;
        end
    end

    s(:,i) = w(:,n)'*EMG;
    
    [pks,loc] = findpeaks(s(:,i).^2);
    
    % cluster the data pks' into two groups, get the index of each data
    % kmeans++ has a better initial centroids selection than kmeans algo
    [idx,~] = kmeansplus(pks',2);
    
    if sum(idx==1)<=sum(idx==2)
        SpikeLoc = loc(idx==1);
    else
        SpikeLoc = loc(idx==2);
    end

    SpikeTrain(SpikeLoc,i) = 1;
    B(:,i) = w(:,end);
end

end

% Post-processing: remove infeasible and redundent spikes from the source
function [SpikeTrain,Goodindex] = MUReplicasRemoval(SpikeTrain,s1,Fs)
% Post-process the results of sEMG decomposition via physiological basis. 
% Goodindex returns the indices of motor units which are not noises or
% motion artifacts.

Timetemp = (1/Fs:1/Fs:length(SpikeTrain)/Fs)';

% Step 1 :select physiologically plausible sources
% The minimum of muscle firing is set to 4 Hz and the max should not be
% more than 35 Hz. This insight is form the physiological standpoint. For
% refrence look at Winter's biomechanics book chapter 9 and 10.
Firings = sum(SpikeTrain,1);
index1 = find(Firings>4*Timetemp(end));
index2 = find(Firings<35*Timetemp(end));
Goodindextemp = intersect(index1,index2);
NumGood = length(Goodindextemp);
sprintf('NumGood: %d', NumGood);

% Step 2 :select only one spike from twin spikes
% It so happens that the spikes in a good source with "playsible firgins"
% are very close to each other (say < 20ms, i.e., 50Hz spike rate), Such
% spike rates are not physiological, therefore, only one can exist. We only
% take the one with greater peak
Time = Timetemp*ones(1,NumGood);
FirT = cell(NumGood,1);

for k = 1:NumGood
    loc = find(SpikeTrain(:,Goodindextemp(k))==1);
    Diffloc = diff(loc);
    loc2 = Diffloc<Fs*0.02;
    for l = 1:length(loc2)
        if loc2(l) == 1
           peaktemp1 = s1(loc(l),k);
           peaktemp2 = s1(loc(l+1),k);
           if peaktemp1>=peaktemp2
           SpikeTrain(loc(l+1),Goodindextemp(k)) = 0;
           else
               SpikeTrain(loc(l),Goodindextemp(k)) = 0;
           end
        end
    end
    FirT{k} = Time(SpikeTrain(:,Goodindextemp(k))==1);
end

% Step 3 :finally, the duplicates motor units should be removed
% CSIndex is a fucntion to find the simialrity of source's (i.e., motor
% unit) spike trains. It works based on the overlapping histograms.
NumMU = length(FirT);
count = 1;
index = 1:NumMU;
NumRemoval = 0;

wrong=0;
while length(index)~=count
    indexRemovaltemp = [];
    for i = 1:length(index)-count %-1
        Logic = CSIndex(FirT{count}, FirT{(count+i)}, 0.01, 10);
        if Logic == 1
            indexRemovaltemp = [indexRemovaltemp count+i];
        end
    end
    FirT(indexRemovaltemp) = [];
    index(indexRemovaltemp) =[];
    NumRemoval = length(indexRemovaltemp);
    count = count+1;
     if(count>10000)
        wrong=1;
        break;
    end
end
Goodindex= Goodindextemp(index);

if(wrong==1)
    Goodindex=[];
    SpikeTrain=[];
end
end


% calculate the silhoutte value on each sample data
function SIL = SILCal(s,Fs)
% A higher silhouette value indicates that the samples are close to each 
% other within the same cluster and well separated from samples in 
% different clusters.
%[b,a] = butter(4,500/(Fs/2),'low');
[~,NumMU] = size(s);
SIL = zeros(1,NumMU);
    for i =1:NumMU
    
        %s(:,i) = filtfilt(b,a,s(:,i));
        [pks,~] = findpeaks(s(:,i).^2);
        
        [idx,~] = kmeansplus(pks',2);
        
        sil = silhouette(pks,idx);
        SIL(i) = (mean(sil(idx==1))+mean(sil(idx==2)))/2;
    end
end


function plot_spikeTrain(spike_train,frq,sil_score,minScore_toPlot)
%PLOT_SPIKETRAIN plots the spike trains in a sorted manner
% initialize
if ~exist("minScore_toPlot","var") || isempty(minScore_toPlot), minScore_toPlot = 0.7; 
end % default value for the recoding frequency 
selected_spikeTrain = spike_train(:,sil_score>minScore_toPlot);
[~,order] = sort(sum(selected_spikeTrain,1),"descend");
bar_height = 0.2;
figure("Renderer","painters","Name","Spike trains of the motor units");
sub_handle = subplot(1,1,1);
ylim(sub_handle,[0,10])
hold on
% plot as a rug plot
% plotting will be very simialr to the rug plots I used to have
% (https://github.com/neuromechanist/add_rug_plot). However, here the rugs
% are the plot.
rugVal = num2cell(selected_spikeTrain(:,order)',2);
rugCount = length(rugVal);
ylim(sub_handle,[0,rugCount*bar_height])
hold on
for r = 1: rugCount
    signRugVal = rugVal{r}; % only one row of rug plot at a time.
    signRugVal = abs(sign(signRugVal)); % rug plots only accepts 0 and 1.
    if exist('barColor', 'var') && size(barColor,1)==rugCount
        bCol = barColor(r,:);
    else
        barColor = turbo(rugCount); bCol = barColor(r,:);
    end
    if r == 1
        plot(sub_handle,[find(signRugVal);find(signRugVal)],[zeros(1,length(find(signRugVal)));...
            ones(1,length(find(signRugVal)))* bar_height*0.7],'Color',bCol,"LineWidth",0.01)
    elseif ~isempty(find(signRugVal)) %#ok<EFIND>
        plot(sub_handle,[find(signRugVal);find(signRugVal)],[ones(1,length(find(signRugVal)))*bar_height*(r-1);...
            ones(1,length(find(signRugVal)))*(bar_height*(r-1)+bar_height*0.7)],'Color',bCol,"LineWidth",0.01)
    end
end
xlim(sub_handle,[0,length(selected_spikeTrain)])
xticks(linspace(0,length(selected_spikeTrain),10));
xticklabels(round(linspace(0,length(selected_spikeTrain)/frq,10)));
xlabel("time (sec)");
ylabel("MUs");
title("MU spike trains increasing force 2");
end


function plot_instantaneous_frequency(spikeTrain, fs, color, unitNumber)
    % Compute inter-spike intervals (ISIs)
    spikeTrain = spikeTrain(:, unitNumber);
    [spikeTimes, ~] = find(spikeTrain);
    ISIs = diff(spikeTimes) / fs;

    % Compute instantaneous frequencies
    instantaneousFrequency = 1 ./ ISIs;
    instantaneousFrequency = filter(0.02*ones(1,50),1, instantaneousFrequency);
    % Plot the instantaneous frequency
    timeAxis = spikeTimes(1:end-1) / fs; % Use spike times as the time axis
    plot(timeAxis, instantaneousFrequency, 'Color', color, 'LineWidth', 1.5, 'DisplayName', ['Motor Unit ' num2str(unitNumber)]);
end