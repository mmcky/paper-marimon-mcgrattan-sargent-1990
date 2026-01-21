function [ind,CS]=create(CS,M,l2,new);
% CREATE takes the old classifier system that does not have a match for
% "new" and creates one by eliminating the most redundent string or
% if none are redundent, the weakest.
%
[row,col]=size(CS);
x=CS(:,1:l2);
lx=M;
most=1; 
info=[x(1,:)];
while lx>=1;
  ind=find(sum(abs((x-ones(lx,1)*x(1,:))')));
  li=lx-length(ind);
  if li>most; most=li; info=x(1,:); end;
  x=x(ind,:); 
  [lx,dum]=size(x);
end;
if li==1;
  win=find(CS(:,l2+2)<=min(CS(:,l2+2)));
  lw=length(win);
  if lw>1; win=win(ceil(rand*lw)); end;
else;
  ind=find(~sum(abs((CS(:,1:l2)-ones(M,1)*info)')))';
  c=[ind,CS(ind,l2+2)];
  win=find(c(:,2)<=min(c(:,2)));
  lw=length(win);
  win=c(win(ceil(rand*lw)),1);
end;
CS(win,:)=[new,round(rand),mean(CS(:,l2+2)),zeros(1,col-l2-2)];
ind=win;
