function mostsimilar=crowding(child,CS,crowdingfactor,crowdingsubpop,M,l)
if crowdingfactor<1; crowdingfactor=1; end;
matchmax=-1; mostsimilar=0;
for j=1:crowdingfactor;
  worst=1+floor(rand(crowdingsubpop,1)*M);
  worst=[worst,CS(worst,l+2)];
  worststr=find(worst(:,2)<=min(worst(:,2)));
  popmember=worst(worststr(1),1);
  match=sum([child(1:l)==CS(popmember,1:l),child(l+1)~=CS(popmember,l+1)] );
  if match>matchmax;
    matchmax=match;
    mostsimilar=popmember;
  end;
end;
