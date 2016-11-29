function sk_wordIter = bmm(K,iterations,sn)
% Bayesian Mixture of Multinomials applied to the KOS dataset

% ADVICE: consider doing clear, close all
rand('seed',sn);
load kos_doc_data.mat

W = max([A(:,2); B(:,2)]);  % number of unique words
D = max(A(:,1));            % number of documents in A
K = 20;                     % number of mixture components we will use

alpha = 10;                 % parameter of the Dirichlet over mixture components
gamma = 0.1;                % parameter of the Dirichlet over words

% Initialization: assign each document a mixture component at random
sd = ceil(K*rand(D,1));     % mixture component assignment of each document
swk = zeros(W,K);           % K multinomials over W unique words
sk_docs = zeros(K,1);            % number of documents assigned to each mixture component

% This populates count matrices swk, sk_docs and sk_words
for d = 1:D                % cycle through the documents
  w = A(A(:,1)==d,2);      % unique words in doc d
  c = A(A(:,1)==d,3);      % counts
  k = sd(d);               % doc d is in mixture k
  swk(w,k) = swk(w,k) + c; % num times word w is assigned to mixture component k
  sk_docs(k) = sk_docs(k) + 1;
end
sk_words = sum(swk,1)';    % num words assigned to mixture component k accross all docs

% This makes a number of Gibbs sampling sweeps through all docs and words
for iter = 1:iterations     % number of Gibbs sweeps
  for d = 1:D       % for each document iterate through all its words
    w = A(A(:,1)==d,2);    % unique words in doc d
    c = A(A(:,1)==d,3);    % counts
    swk(w,sd(d)) = swk(w,sd(d)) - c;  % remove doc d words from count table
    sk_docs(sd(d)) = sk_docs(sd(d)) - 1;        % remove document counts
    sk_words(sd(d)) = sk_words(sd(d)) - sum(c); % remove total word counts
    lb = zeros(1,K);    % log probability of doc d under mixture component k
    for k = 1:K
      ll = c'*( log(swk(w,k)+gamma) - log(sk_words(k) + gamma*W) );
      lb(k) = log(sk_docs(k) + alpha) + ll;
    end
    b = exp(lb-max(lb));  % exponentiation of log probability plus constant
    kk = sampDiscrete(b); % sample from unnormalized discrete distribution
    swk(w,kk) = swk(w,kk) + c;        % add back document word counts
    sk_docs(kk) = sk_docs(kk) + 1;              % add back document counts
    sk_words(kk) = sk_words(kk) + sum(c);       % add back document counts
    sd(d) = kk;
  end
  sk_wordIter(:,iter) = sk_words/sum(sk_words);
  [perplexity, lp] = computePerplexity(B,W,gamma,alpha,sk_words,sk_docs,K,swk);
  perpIter(iter) = perplexity;
  lpIter(iter) = lp;
end

lp = 0; nd = 0;
for d = unique(B(:,1))'
  w = B(B(:,1)==d,2);    % unique words in doc d
  c = B(B(:,1)==d,3);    % counts
  z = log(sk_docs(:) + alpha) - log(sum(sk_docs(:)+alpha));
  for k = 1:K
    b = (swk(:,k)+gamma)/(sk_words(k) + gamma*W);
    z(k) = z(k) + c'*log(b(w));    % probability, doc d
  end
  lp = lp + log(sum(exp(z-max(z))))+max(z);
  nd = nd + sum(c);             % number of words, doc d
end
perplexity = exp(-lp/nd);   % perplexity

% this code allows looking at top I words for each mixture component
I = 20;
for k=1:K, [i ii] = sort(-swk(:,k)); ZZ(k,:)=ii(1:I); end
for i=1:I, for k=1:K, fprintf('%-15s',V{ZZ(k,i)}); end; fprintf('\n'); end

figure;
hist(sk_words');
title(sn);
xlabel('iterations')
ylabel('mixing proportions')
figure;
plot(perpIter);
title(sn);
xlabel('iterations')
ylabel('perplexity')
figure;
plot(lpIter);
title(sn);
xlabel('iterations')
ylabel('logP')
