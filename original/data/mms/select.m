function j=select(popsize,sumfitness,popf)
partsum=0;
j=0;
r=rand*sumfitness;
while partsum<r & j<popsize;
  j=j+1;
  partsum=partsum+popf(j);
end;
