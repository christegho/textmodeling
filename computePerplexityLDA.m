function perplexity=computePerplexityLDA(B,Swd,K,swk,sk,alpha, gamma, W,iterations)
% compute the perplexity for all words in the test set B
% We need the new Skd matrix, derived from corpus B
lp = 0; nd = 0;
for d = unique(B(:,1))'  % loop over all documents in B
  % randomly assign topics to each word in test document d
  z = zeros(W,K);
  for w = B(B(:,1)==d,2)'   % w are the words in doc d
    for i=1:Swd(w,d)
      k = ceil(K*rand());
      z(w,k) = z(w,k) + 1;
    end
  end
  Skd = sum(z,1)';
  Sk = sk + Skd;  
  % perform some iterations of Gibbs sampling for test document d
  for iter = 1:2
    for w = B(B(:,1)==d,2)' % w are the words in doc d
      a = z(w,:); % number of times word w is assigned to each topic in doc d
      ka = find(a); % topics with non-zero counts for word d in document d
      for k = ka(randperm(length(ka)))
        for i = 1:a(k)
          z(w,k) = z(w,k) - 1;   % remove word from count matrix for doc d
          Skd(k) = Skd(k) - 1;
          b = (alpha + Skd) .* (gamma + swk(w,:)') ./ (W*gamma + sk);
          kk = sampDiscrete(b);
          z(w,kk) = z(w,kk) + 1; % add word with new topic to count matrix for doc d
          Skd(kk) = Skd(kk) + 1;
        end
      end
    end
  end
  b=(alpha+Skd')/sum(alpha+Skd)*bsxfun(@rdivide,gamma+swk',W*gamma+sk);  
  w=B(B(:,1)==d,2:3);
  lp = lp + log(b(w(:,1)))*w(:,2);   % log probability, doc d
  nd = nd + sum(w(:,2));             % number of words, doc d
end
perplexity = exp(-lp/nd)   % perplexity