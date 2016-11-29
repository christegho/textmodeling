function [perplexity, lp] = computePerplexity(B,W,gamma,alpha,sk_words,sk_docs,K,swk)
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