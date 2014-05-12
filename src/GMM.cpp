#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
//*******************declare********************
void train(int argc,char *argv[]);
void dev(int argc,char *argv[]);
void test(int argc,char *argv[]);
void train_GMM(int comp_num,int label);
void ini_par(int comp_num,int label);
/*******************data**********************/
string	help="";
const int MAX_DATA=100000;
const int MAX_COMP=20;
const double PI=3.141592653;
const double e=2.71828182845904523536028;
double data[2][MAX_DATA][2]={{{0}}};
int counter[2]={0};
double weight[MAX_COMP]={0};
double mean[MAX_COMP][2]={{0}};
double covar[MAX_COMP][2][2]={{{0}}};
int comp_num=0;

double weight2[MAX_COMP]={0};
double mean2[MAX_COMP][2]={{0}};
double covar2[MAX_COMP][2][2]={{{0}}};
int comp_num2=0;
//********
double prob[MAX_DATA][MAX_COMP]={{0}};
double sum_prob[MAX_COMP]={0};
double MAX_X[2],MAX_Y[2],MIN_X[2],MIN_Y[2];

/*******************tools*********************/
int str2num(string s)
{
	int num=0;
	for(int i=0;i<s.length();++i)
	{
		num=num*10+s[i]-'0';
	}
	return num;
}
void read_train_file(string datafile)
{
	ifstream fin(datafile.c_str());
	if(fin==NULL)
	{
		cout<<"The train file is not exist"<<endl;
		exit(1);
	}
	double x=0,y=0;
	int label=0;
	MAX_X[0]=1,MAX_Y[0]=1,MIN_X[0]=0,MIN_Y[0]=0;
	MAX_X[1]=1,MAX_Y[1]=1,MIN_X[1]=0,MIN_Y[1]=0;	
	while(fin>>x>>y>>label)
	{
		if(label==1)
		{
			if(x>MAX_X[0])
				MAX_X[0]=x;
			if(y>MAX_Y[0])
				MAX_Y[0]=y;
			if(x<MIN_X[0])
				MIN_X[0]=x;
			if(y<MIN_Y[0])
				MIN_Y[0]=y;
			data[0][counter[0]][0]=x;
			data[0][counter[0]][1]=y;
			counter[0]++;	
		}
		else
		{
			if(x>MAX_X[1])
				MAX_X[1]=x;
			if(y>MAX_Y[1])
				MAX_Y[1]=y;
			if(x<MIN_X[1])
				MIN_X[1]=x;
			if(y<MIN_Y[1])
				MIN_Y[1]=y;
			data[1][counter[1]][0]=x;
			data[1][counter[1]][1]=y;
			counter[1]++;			
		}
	}
	fin.close();
}
void read_model(string modelfile)
{
	ifstream fin(modelfile.c_str());
	if(fin==NULL)
	{
		cout<<"The train file is not exist"<<endl;
		exit(1);
	}
	fin>>counter[0];
	fin>>comp_num;
	for(int i=0;i<comp_num;++i)
	{
		fin>>weight[i]>>mean[i][0]>>mean[i][1]
			>>covar[i][0][0]>>covar[i][0][1]
			>>covar[i][1][0]>>covar[i][1][1]; 			
	}
	fin>>counter[1];
	fin>>comp_num2;
	for(int i=0;i<comp_num2;++i)
	{
		fin>>weight2[i]>>mean2[i][0]>>mean2[i][1]
			>>covar2[i][0][0]>>covar2[i][0][1]
			>>covar2[i][1][0]>>covar2[i][1][1]; 			
	}	
}
void wirte_model(string modelfile,int label,string way)
{
	label=label-1;
	ofstream fout;
	if(way=="new")
		fout.open(modelfile.c_str());
	if(way=="app")
		fout.open(modelfile.c_str(),ios::app);
	fout<<counter[label]<<endl;
	fout<<comp_num<<endl;
	for(int i=0;i<comp_num;++i)
	{
		fout<<weight[i]<<" "<<mean[i][0]<<" "<<mean[i][1]
			<<" "<<covar[i][0][0]<<" "<<covar[i][0][1]
			<<" "<<covar[i][1][0]<<" "<<covar[i][1][1]<<endl; 
	}
	fout.close();
}
void sub(double a[2],double b[2],double res[2])
{
	res[0]=a[0]-b[0];
	res[1]=a[1]-b[1];
}
double detMat(double a[2][2])
{
	return (a[0][0]*a[1][1]-a[0][1]*a[1][0]);
}
void invMat(double a[2][2],double inva[2][2])
{
	double det=detMat(a);
	inva[0][0]=1/det*a[1][1];
	inva[0][1]=-1/det*a[0][1];
	inva[1][0]=-1/det*a[1][0];
	inva[1][1]=1/det*a[0][0];
}
double quadric(double a[2],double mat[2][2],double b[2])
{
	double temp[2]={0};	
	temp[0]=a[0]*mat[0][0]+a[1]*mat[1][0];
	temp[1]=a[0]*mat[0][1]+a[1]*mat[1][1];
	return temp[0]*b[0]+temp[1]*b[1];
}
/*********************************************/
int main(int argc,char *argv[])
{
	string	help="";
			help.append("1. in: -train -numcomp -infile -outfile\n");
			help.append("   out: comp_num, weight, mean, cov\n");
			help.append("2. in: -dev -modelfile -devfile -analysisfile\n");
			help.append("   out: outputlabel, accuracy\n");
			help.append("3  in: -test -modelfile -testfile -labelfile\n");
			help.append("   out: outlabel\n");
	if(argc<=1)
	{
		cout<<help;
		exit(1);	
	}
	string oper(argv[1]);
	++argv;
	if(oper=="-train")
		train(argc-2,++argv);
	else if(oper=="-dev")
		dev(argc-2,++argv);
	else if(oper=="-test")
		test(argc-2,++argv);
	else{
		cout<<help;
		exit(1);		
	}  

}
void train(int argc,char *argv[])
{
	comp_num=5;
	string datafile;
	string modelfile;
	if(argc>3||argc<2)
	{
		cout<<help<<endl;
		exit(1);	
	}	
	if(argc==3)
	{
		string t=string(++argv[0]);
		comp_num=str2num(t);
		datafile=string(++argv[1]);
		modelfile=string(++argv[2]);
	//	cout<<comp_num<<" "<<datafile<<" "<<modelfile<<endl;
	}
	if(argc==2)
	{
		datafile=string(++argv[0]);
		modelfile=string(++argv[1]);		
	}
	read_train_file(datafile);
	cout<<"start train"<<endl;
	train_GMM(comp_num,1);
	wirte_model(modelfile,1,"new");	
	train_GMM(comp_num,2);
	wirte_model(modelfile,2,"app");		
	cout<<"over train"<<endl;
}
void train_GMM(int comp_num,int label)
{
	ini_par(comp_num,label);
	label=label-1;

for(int c=0;c<10000;++c)
{
	//initial 
	if(c%200==0)
		cout<<c<<" iter of "<<10000<<endl;
	double maxchval=0;
	for(int i=0;i<comp_num;++i)
	{
		sum_prob[i]=0;
	}
	//cal probability
	for(int i=0;i<counter[label];++i)
	{
		double sum=0;
		double inv[2][2]={{0}};
		double dif[2]={0};
		for(int j=0;j<comp_num;++j)
		{
			invMat(covar[j],inv);
			sub(data[label][i],mean[j],dif);
			sum+=weight[j]*(1/(2*PI)/sqrt(abs(detMat(covar[j])))*
				 pow(e,-0.5*quadric(dif,inv,dif)));
		}
		for(int j=0;j<comp_num;++j)
		{
			invMat(covar[j],inv);
			sub(data[label][i],mean[j],dif);
			prob[i][j]=weight[j]*(1/(2*PI)/sqrt(abs(detMat(covar[j])))*
				pow(e,-0.5*quadric(dif,inv,dif)))/sum;
			sum_prob[j]+=prob[i][j];		
		}
	}
	//update weight
	for(int i=0;i<comp_num;++i)
	{
		double mid=sum_prob[i]/counter[label];
		if(abs(weight[i]-mid)>maxchval)
			maxchval=abs(weight[i]-mid);
		weight[i]=mid; 
	} 
	//update mean
	for(int i=0;i<comp_num;++i)
	{
		double sum[2]={0};
		for(int j=0;j<counter[label];++j)
		{
			sum[0]+=data[label][j][0]*prob[j][i];
			sum[1]+=data[label][j][1]*prob[j][i];
		}
		if(abs(sum[0]/sum_prob[i]-mean[i][0])>maxchval)
			maxchval=abs(sum[0]/sum_prob[i]-mean[i][0]);
		if(abs(sum[1]/sum_prob[i]-mean[i][1])>maxchval)
			maxchval=abs(sum[1]/sum_prob[i]-mean[i][1]);			
		mean[i][0]=sum[0]/sum_prob[i];
		mean[i][1]=sum[1]/sum_prob[i];
	} 
	//update covariance
	for(int i=0;i<comp_num;++i)
	{
		double newmat[2][2]={{0}};
		for(int j=0;j<counter[label];++j)
		{
			newmat[0][0]+=prob[j][i]*(data[label][j][0]-mean[i][0])*(data[label][j][0]-mean[i][0]);
			newmat[0][1]+=prob[j][i]*(data[label][j][0]-mean[i][0])*(data[label][j][1]-mean[i][1]);
			newmat[1][0]+=prob[j][i]*(data[label][j][1]-mean[i][1])*(data[label][j][0]-mean[i][0]);
			newmat[1][1]+=prob[j][i]*(data[label][j][1]-mean[i][1])*(data[label][j][1]-mean[i][1]);	
		}
		if(abs(covar[i][0][0]-newmat[0][0]/sum_prob[i])>maxchval)
			maxchval=abs(covar[i][0][0]-newmat[0][0]/sum_prob[i]);
		if(abs(covar[i][0][1]-newmat[0][1]/sum_prob[i])>maxchval)
			maxchval=abs(covar[i][0][1]-newmat[0][1]/sum_prob[i]);
		if(abs(covar[i][1][0]-newmat[1][0]/sum_prob[i])>maxchval)
			maxchval=abs(covar[i][1][0]-newmat[1][0]/sum_prob[i]);
		if(abs(covar[i][1][1]-newmat[1][1]/sum_prob[i])>maxchval)
			maxchval=abs(covar[i][1][1]-newmat[1][1]/sum_prob[i]);
		covar[i][0][0]=newmat[0][0]/sum_prob[i];	
		covar[i][0][1]=newmat[0][1]/sum_prob[i];
		covar[i][1][0]=newmat[1][0]/sum_prob[i];
		covar[i][1][1]=newmat[1][1]/sum_prob[i];
	}

	if(maxchval<0.00000000001)
	{
		cout<<"Converge!!! Total iteration number is"<<c<<endl;
		break;
	}
}
}
void ini_par(int comp_num,int label)
{
	label=label-1;
	for(int i=0;i<comp_num;++i)
	{
		weight[i]=(double)1/comp_num;
	}
	for(int i=0;i<comp_num;++i)
	{
		mean[i][0]=(MAX_X[label]-MIN_X[label])*(double)rand()/RAND_MAX+MIN_X[label];
		mean[i][1]=(MAX_Y[label]-MIN_Y[label])*(double)rand()/RAND_MAX+MIN_Y[label];
	}
	for(int i=0;i<comp_num;++i)
	{
		covar[i][0][0]=1;
		covar[i][1][1]=1;
		covar[i][0][1]=0;
		covar[i][1][0]=0;			
	}
}
void dev(int argc,char *argv[])
{
	string modelfile;
	string devfile;
	string outfile;
	if(argc!=3)
	{
		cout<<help<<endl;
		exit(1);	
	}	
	modelfile=string(++argv[0]);
	devfile=string(++argv[1]);
	outfile=string(++argv[2]);
	read_model(modelfile);
	

	ifstream fin(devfile.c_str());
	ofstream fout(outfile.c_str());
	double point[2]={0};
	int label=0;
	int c_c=0;
	int c_t=0;
	while(fin>>point[0]>>point[1]>>label)
	{
		double p1=0;
		double inv[2][2]={{0}};
		double dif[2]={0};
		for(int i=0;i<comp_num;++i)
		{
			invMat(covar[i],inv);
			sub(point,mean[i],dif);
			p1+=weight[i]*(1/(2*PI)/sqrt(abs(detMat(covar[i])))*
				 pow(e,-0.5*quadric(dif,inv,dif)));
		}	
		p1*=(double)counter[0]/(counter[0]+counter[1]);
		double p2=0;
		//cout<<comp_num2<<endl;
		for(int i=0;i<comp_num2;++i)
		{
			invMat(covar2[i],inv);
			sub(point,mean2[i],dif);
			p2+=weight2[i]*(1/(2*PI)/sqrt(abs(detMat(covar2[i])))*
				 pow(e,-0.5*quadric(dif,inv,dif)));
				//cout<<weight2[i]<<endl;
		}	
		p2*=(double)counter[1]/(counter[0]+counter[1]);
		int t_lab=(p1>p2)?1:2;
		if(t_lab==label)
		{
			c_c+=1;
		}
		c_t+=1;
		fout<<label<<"\t"<<t_lab<<"\t"<<(t_lab==label)<<"\t"<<point[0]<<"\t"<<point[1]<<"\t"<<p1<<"\t"<<p2<<endl;
	}
	fout<<"Accuracy"<<(double)c_c/c_t<<endl; 
	cout<<"Accuracy"<<(double)c_c/c_t<<endl; 
	fin.close();
	fout.close(); 
}
void test(int argc,char *argv[])
{
	string modelfile;
	string testfile;
	string outfile;

	if(argc!=3)
	{
		cout<<help<<endl;
		exit(1);	
	}	
	modelfile=string(++argv[0]);
	testfile=string(++argv[1]);
	outfile=string(++argv[2]);
	read_model(modelfile);
	ifstream fin(testfile.c_str());
	ofstream fout(outfile.c_str());
	double point[2]={0};
	int label;
	while(fin>>point[0]>>point[1])
	{
		double p1=0;
		double inv[2][2]={{0}};
		double dif[2]={0};
		for(int i=0;i<comp_num;++i)
		{
			invMat(covar[i],inv);
			sub(point,mean[i],dif);
			p1+=weight[i]*(1/(2*PI)/sqrt(abs(detMat(covar[i])))*
				 pow(e,-0.5*quadric(dif,inv,dif)));
		}	
		p1*=(double)counter[0]/(counter[0]+counter[1]);
		
		double p2=0;
		for(int i=0;i<comp_num2;++i)
		{
			invMat(covar2[i],inv);
			sub(point,mean2[i],dif);
			p2+=weight2[i]*(1/(2*PI)/sqrt(abs(detMat(covar2[i])))*
				 pow(e,-0.5*quadric(dif,inv,dif)));
		}	
		p2*=(double)counter[1]/(counter[0]+counter[1]);
		
		int t_lab=(p1>p2)?1:2;
		fout.setf(ios::fixed, ios::floatfield);
		fout.precision(6);
		fout<<point[0]<<" "<<point[1]<<"  "<<t_lab<<endl;
	} 
	fin.close();
	fout.close(); 	
}

