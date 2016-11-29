doc2001indeces = B(:,1) == 2001;
doc2001words = B(doc2001indeces,2);
logP = sum(B(doc2001indeces,3).*log(counts(doc2001words)))
%perplexity = sum( log(counts(doc2001words)) .* counts(doc2001words));
n2001 = sum(B(doc2001indeces,3))
perplexity = exp(-logP / n2001)

docBwords = B(:,2);
logPB = sum(B(:,3).*log(counts(docBwords)))
%perplexityB = sum( log(counts(docBwords)) .* counts(docBwords));
perplexityB = exp(- logPB / sum(B(:,3)))