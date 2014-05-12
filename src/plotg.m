clc
clear
close all

mu=[2.50374 0.98681]; 
Sigma=[0.0394932 0.000301976 ;0.000301976 0.0547735];
[X,Y]=meshgrid(-2:0.1:4,-2:0.1:4);
p=mvnpdf([X(:) Y(:)],mu,Sigma);

mu=[0.79469 1.50773];
Sigma=[0.0325006 0.000385762 ;0.000385762 0.0475773];
temp=mvnpdf([X(:) Y(:)],mu,Sigma);
p=p+temp;

mu=[1.49907 -5.09279e-005];
Sigma=[0.072716 -0.00492702 ;-0.00492702 0.0749338];
temp=mvnpdf([X(:) Y(:)],mu,Sigma);
p=p+temp;

mu=[-0.503759 -0.014981];
Sigma=[0.067924 -0.00593991; -0.00593991 0.0691033];
temp=mvnpdf([X(:) Y(:)],mu,Sigma);
p=p+temp;

p=p/4;

p=reshape(p,size(X));%将Z值对应到相应的坐标上

figure
set(gcf,'Position',get(gcf,'Position').*[1 1 1.3 1])

subplot(2,3,[1 2 4 5])
surf(X,Y,p),axis tight,title('GMM')
subplot(2,3,3)
surf(X,Y,p),view(2),axis tight,title('Projection on XY')
subplot(2,3,6)
surf(X,Y,p),view([0 0]),axis tight,title('Projection on XZ')
