PPK = zeros(10,1);
i=1;
for K=5:5:50
    
    PPk(i) = bmm(20, K,27);
    i = i+1;
end

PPilda = zeros(10,1);
i=1;
for K=5:5:50
    
    PPilda(i) = lda(20, K);
    i = i+1;
end

PPkldaA = zeros(10,1);
i=1;
for K=5:5:50
    
    PPkldaA(i) = lda(K, 10);
    i = i+1;
end

    PPk = bmm(20, 20,15);
    PPk = bmm(20, 20,27);
    PPk = bmm(20, 20,40)


