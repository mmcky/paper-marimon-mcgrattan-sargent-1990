function x=decode(popc,nparms,lparm,maxparm,minparm);
for j=1:nparms;
  s=sum(lparm(1:j-1));
  l=lparm(j);
  x=[x,minparm(j)+(maxparm(j)-minparm(j))*(popc(:,s+1:s+l)*((2).^  ..
    [0:l-1]')/(2^l-1))];
end;
