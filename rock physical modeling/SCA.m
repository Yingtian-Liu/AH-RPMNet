function [kbr,mubr]=SCA(k,mu,asp,x)

% SCA - Effective elastic moduli for multi-component composite
% using SCA method. 
%	k,mu:      Bulk and shear moduli of the N constituent
%		       phases (k,mu, vectors of length N)
%	asp:       Aspect ratio for the inclusions of the N phases
%     x:       Fraction of each phase. Sum(x) should be 1.
%	kbr,mubr:  	Effective bulk and shear moduli 

kbr=[]; mubr=[]; 

k=k(:); mu=mu(:); asp=asp(:); x=x(:);
indx=find(asp==1); asp(indx)=0.99*ones(size(indx));
theta=zeros(size(asp)); fn=zeros(size(asp));  


obdx=find(asp<1);
theta(obdx)=(asp(obdx)./((1-asp(obdx).^2).^(3/2))).*...
             (acos(asp(obdx)) -asp(obdx).*sqrt(1-asp(obdx).^2));
fn(obdx)=(asp(obdx).^2./(1-asp(obdx).^2)).*(3.*theta(obdx) -2);

prdx=find(asp>1);
theta(prdx)=(asp(prdx)./((asp(prdx).^2-1).^(3/2))).*...
             (asp(prdx).*sqrt(asp(prdx).^2-1)-acosh(asp(prdx)));
fn(prdx)=(asp(prdx).^2./(asp(prdx).^2-1)).*(2-3.*theta(prdx));

ksc= sum(k.*x);
musc= sum(mu.*x);
knew= 0.;
munew= 0.;

tol=1e-6*k(1);
del=abs(ksc-knew);
niter=0;

while( (del > abs(tol)) && (niter<3000) )
	nusc=(3*ksc-2*musc)/(2*(3*ksc+musc));
	a=mu./musc -1; 
	b=(1/3)*(k./ksc -mu./musc); 
	r=(1-2*nusc)/(2*(1-nusc));

	f1=1+a.*((3/2).*(fn+theta)-r.*((3/2).*fn+(5/2).*theta-(4/3)));
	f2=1+a.*(1+(3/2).*(fn+theta)-(r/2).*(3.*fn+5.*theta))+b.*(3-4*r);
	f2=f2+(a/2).*(a+3.*b).*(3-4.*r).*(fn+theta-r.*(fn-theta+2.*theta.^2));
	f3=1+a.*(1-(fn+(3/2).*theta)+r.*(fn+theta));
	f4=1+(a./4).*(fn+3.*theta-r.*(fn-theta));
	f5=a.*(-fn+r.*(fn+theta-(4/3))) + b.*theta.*(3-4*r);
	f6=1+a.*(1+fn-r.*(fn+theta))+b.*(1-theta).*(3-4.*r);
	f7=2+(a./4).*(3.*fn+9.*theta-r.*(3.*fn+5.*theta)) + b.*theta.*(3-4.*r);
    f8=a.*(1-2.*r+(fn./2).*(r-1)+(theta./2).*(5.*r-3))+b.*(1-theta).*(3-4.*r);
	f9=a.*((r-1).*fn-r.*theta) + b.*theta.*(3-4.*r);
	p=3*f1./f2; 
	q=(2./f3) + (1./f4) +((f4.*f5 + f6.*f7 - f8.*f9)./(f2.*f4));
	p=p./3; 
	q=q./5; 

	knew= sum(x.*k.*p)/sum(x.*p);
	munew= sum(x.*mu.*q)/sum(x.*q);
	
	del=abs(ksc-knew);
	ksc=knew;
	musc=munew;
	niter=niter+1;
end		
kbr=ksc; mubr=musc;
end

