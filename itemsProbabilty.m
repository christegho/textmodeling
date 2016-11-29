alpha = 1;
maxA = max(A);
counts = zeros(maxA(2),1);
for i=1:length(A)
    counts(A(i,2)) = counts(A(i,2)) + A(i,3);
end
counts = counts+alpha*ones(maxA(2),1);
counts = counts./ (sum(counts));
[kk, ii] = sort(counts, 'descend');
barh(kk(20:-1:1));
set(gca,'YTickLabel',V(ii(20:-1:1)),'YTick',1:20);
axis([0 .015 0.5 20+0.5]);
title('20 largest probability items')