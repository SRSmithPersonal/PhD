#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include "Fresnel.h"

# include "asa0472.h"

double SamL = 0.00499905160831;
double * HPE;
double * HME;
int Zone;
double * freq2;
double * phaseHold;
double * ampHold;
double complex * nCalc;
int currentSpace;
int ScaleUp;
double complex n0 = 1.00027 + 0j;
double Delta;
double MaxS;
double MaxR;
double MaxT;

double Absu(double a){
	if(a<0.0){
		return -a;
	}
	else{
		return a;
	}
}

double * TIME(int len, double start, double end){
	double dist = end - start;
	double step = dist/(double)len;
	static double *t;
	t = (double *)malloc(len*sizeof(double));
	int j;
	for(j=0;j<len;j++){
		t[j] = start + step*(double)j;
	}
	return t;
}

double * FREQS(int len, double step){
	static double *f;
	f = (double *)malloc(len*sizeof(double));
	int j;
	for(j=0;j<len;j++){
		f[j] = (1.0/step)*((double)j/(double)len);
	}
	return f;
}

double * FREQShifted(int len, double step){
	static double *f;
	f = (double *)malloc(len*sizeof(double));
	int j;
	for(j=0;j<len;j++){
		f[j] = (1.0/step)*((double)(j-len/2)/(double)len);
	}
	return f;
}

void plotE(char *Oname){
	FILE * gnuplotPipe = popen("gnuplot -persistent","w");
	fprintf(gnuplotPipe, "set ylabel \"Amplitude (a.u.)\"\n set xlabel \"Time (ps)\"\n plot \"%s\" using 1:2 title \'Electric field in time\' with lines\n quit\n",Oname);
	fclose(gnuplotPipe);
}

void plotNcomp(char *Oname){
	FILE * gnuplotPipe = popen("gnuplot -persistent","w");
	fprintf(gnuplotPipe, "set ylabel \"Refractive Index\"\n set xlabel \"Frequency (THz)\"\n set xrange [0.0:10.0]\n plot \"%s\" using 1:2 title \'Refractive Index Out\' with lines, \"%s\" using 1:3 title \'Refractive Index In\' with lines\n quit\n",Oname,Oname);
	fclose(gnuplotPipe);
}

void plotKcomp(char *Oname){
	FILE * gnuplotPipe = popen("gnuplot -persistent","w");
	fprintf(gnuplotPipe, "set ylabel \"Extinction Coefficient\"\n set xlabel \"Frequency (THz)\"\n set xrange [0.0:10.0]\n plot \"%s\" using 1:2 title \'Extinction Coefficient Out\' with lines, \"%s\" using 1:3 title \'Extinction Coefficient In\' with lines\n quit\n",Oname,Oname);
	fclose(gnuplotPipe);
}
void plotNcomp2(char *Oname){
	FILE * gnuplotPipe = popen("gnuplot -persistent","w");
	fprintf(gnuplotPipe, "set ylabel \"Refractive Index\"\n set xlabel \"Frequency (THz)\"\n set xrange [0.3:0.7]\n set yrange [1.44:1.46]\n plot \"%s\" using 1:2 title \'Refractive Index Out\' with lines, \"%s\" using 1:3 title \'Refractive Index In\' with lines\n quit\n",Oname,Oname);
	fclose(gnuplotPipe);
}

void plotKcomp2(char *Oname){
	FILE * gnuplotPipe = popen("gnuplot -persistent","w");
	fprintf(gnuplotPipe, "set ylabel \"Extinction Coefficient\"\n set xlabel \"Frequency (THz)\"\n set xrange [0.3:0.7]\n plot \"%s\" using 1:2 title \'Extinction Coefficient Out\' with lines, \"%s\" using 1:3 title \'Extinction Coefficient In\' with lines\n quit\n",Oname,Oname);
	fclose(gnuplotPipe);
}

int * Appendint(int len, int * y, int x){
	static int *temp;
	temp = (int*)malloc((len + 1)*sizeof(int));
	int j;
	int standin;
	for(j = 0; j<len; j++){
		standin = y[j]+0;
		temp[j] = standin;
	}
	temp[len] = x;
	return temp;
}
int findStep(int len,double * E, double * time, double d){
	double step = time[1] - time[0];
	int j;
	int Max;
	Max = 0;
	for(j=0;j<len;j++){
		if(fabs(E[j])>fabs(E[Max])){
			Max = j;
		}
	}
	int shift = (int)((d/c) / step);
	int Max2;
	Max2 = Max + shift/2;
	for(j=Max2;j< len;j++){
		if(fabs(E[j])>fabs(E[Max2])){
			Max2 = j;
		}
	}
	return Max2 - Max;
}
int * peakFindS(int len,double * E, int shift){
	int j;
	int Max;
	Max = 0;
	for(j=0;j<len;j++){
		if(fabs(E[j])>fabs(E[Max])){
			Max = j;
		}
	}
	static int *Peaks;
	int numPeaks = 0;
	int Max2;
	while(1==1){
		Peaks = Appendint(numPeaks,Peaks,Max);
		numPeaks++;
		if(Max + 3*shift > len){break;}
		Max2 = Max + shift;
		for(j=Max2;j< (Max + 3*shift);j++){
			if(fabs(E[j]>fabs(E[Max2]))){
				Max2 = j;
			}
		}
		if(0.01*fabs(E[Peaks[0]])>fabs(E[Max2])){break;}
		Max = Max2;
	}

	return Peaks;
}

double * Cut(int len, double * E, int shift, int peak){
	static double * New;
	New = (double*)malloc((len)*sizeof(double));
	int j;
	int start = peak-shift/2;
	int stop = peak+shift/2;
	for(j=0;j<len;j++){
		if(j>=start && j<stop){
			New[j] = E[j];
		}
		else{
			New[j] = 0.0;
		}
	}
	return New;
}

double * EPadder(int oldLen, int newLen, double * E){
	static double *New;
	New = (double *)malloc(newLen*sizeof(double));
	int i =0;
	int start = (int)((float)newLen/2.0 - (float)oldLen/2.0);
	int end = (int)((float)newLen/2.0 + (float)oldLen/2.0);
	for(i=0;i<newLen;i++){
		if(i>=start && i< end){
			New[i] = E[i-start];
		}
		else{
			New[i] = 0.0;
		}
	}
	return New;
}

double complex * ChopShopE(int oldLen,int newLen,int Start, double complex *E){
	static double complex *New;
	New = (double complex *)malloc(newLen*sizeof(double complex));
	int i;
	for(i=0;i<newLen;i++){
		New[i] = E[i+Start];
	}
	return New;
}

double * ChopShopF(int oldLen,int newLen,int Start, double *F){
	static double *New;
	New = (double *)malloc(newLen*sizeof(double));
	int i;
	for(i=0;i<newLen;i++){
		New[i] = F[i+Start];
	}
	return New;
}

int SearchStart(int len, double StartF, double *Freq){
	int S = 0;
	int i;
	for(i=0;i<len;i++){
		if(Freq[i]>=StartF){
			S = i;
			break;
		}
	}
	return S;
}

int NewLenScan(int len,int start,double EndF, double *Freq){
	int count = 0;
	int i;
	for(i=start;i<len;i++){
		if(Freq[i]<=EndF){
			count++;
		}
		else{
			break;
		}
	}
	return count;
}

double complex *Hexp(int len,double complex * Eref,double complex * Esam){
	static double complex *H;
	H = (double complex *)malloc(len*sizeof(double complex));
	int i;
	for(i=0;i<len;i++){
		H[i] = Esam[i]/Eref[i];
	}
	return H;
}

int findAMP(int len, double * E){
	int i;
	double Max = 0.0;
	int pos;
	for(i=0;i<len;i++){
		if(Absu(E[i])>Max){
			pos=i;
			Max=Absu(E[i]);
		}
	}
	return pos;
}

double complex T12(double complex n1, double complex n2){
    return 2*n1/(n2+n1);
}

double nInitgen(){
	return c*Delta/SamL+n0;
}

double kInitgen(double f){
	/*double complex nT = nInitgen();
	double AT = creal(T12(n0,nT)*T12(nT,n0));
	double kk = -(1.0/SamL)*c/(2.0*pi*f)*log(MaxS/AT*MaxR);*/
	double kk = -(1.0/SamL)*c/(2.0*pi*f)*log(MaxS/MaxR);
	return kk;
}

double complex R12(double complex n1,double complex n2){
    return (n2-n1)/(n2+n1);
}

double complex P(double complex n,double f){
    return cexp(-I*2*pi*f*n*SamL/c);
}

int Sig(double complex n1){
    double Si = MaxT*c/(2*creal(n1)*SamL) - 0.5;
    return (int)Si;
}

double complex Htheory(double f,double complex n1){
    double complex PA = P((n1-n0),f);
    double complex P1 = P(n1, f);
    double complex R = R12(n0,n1);
    double complex T0 = T12(n0,n1);
    double complex T1 = T12(n1, n0);
    double complex A = PA*T0*T1;
    double complex H = A;
    int i;
    int D = Sig(n1);
    double complex A2;
    for(i=1;i<D+1;i++){
        A2 = cpow((cpow(R,2.0) * cpow(P1,2.0)),(double)i);
        H += A*A2;
    }
    return H;
}

double ERR(double nA1[2]){
    double Err = 0.0;
    double complex * Ht;
    Ht = (double complex *)malloc(Zone*sizeof(double complex));
    int i;
    nCalc[currentSpace] = nA1[0]-I*nA1[1]*pow(10,ScaleUp);
    for(i=0;i<Zone;i++){
        Ht[i] = Htheory(freq2[i],nCalc[i]);
    }
	
    double * HtM;
    double * HtP;
	
    HtM = Amplitude(Zone,Ht);
    HtP = Phase(Zone,Ht);
    Unwrap(Zone,HtP);
    double M;
    double A;
    for(i=0;i<Zone;i++){
        M = HME[i]-HtM[i];
        A = HPE[i]-HtP[i];
	/*printf("%d, %f, %f\n",i,M,A);*/
        Err += Absu(M) + Absu(A);
	/*if(currentSpace==1){
		printf("%d, %f, %f, %f\n",i,A,Absu(M),Err);
	}*/
    }
    return Err;
}

/*double Smoothness(double Thick[1]){
	SamL = Thick[0];
	double * nCalc;
	nCalc = (double *)malloc(2*Zone*sizeof(double));
	
	int i;
	for(i=0;i<Zone;i++){
		nCalc[i] = nInitgen();
		nCalc[i+Zone] = kInitgen(freq2[i]);
	}
	
	double reqmin = 1.0E-04;
	int konvge = 10;
	int kcount = 100000;
	double ynewlo = ERR(nCalc);
	double nCalc2[754];
	double StepN[754];
	int icount;
	int ifault;
	int numres;
	for(i=0;i<2*Zone;i++){
		StepN[i] = 1.0;
	}
	nelmin(ERR,754,nCalc,nCalc2,&ynewlo,reqmin,StepN,konvge,kcount,&icount,&numres,&ifault);
	
	double T = 0.0;
	for(i=1;i<Zone;i++){
		T += abs(nCalc2[i]-nCalc2[i-1]) + abs(nCalc2[i+Zone]-nCalc2[i+Zone-1]);
	}
	printf("%f\n",T);
	return T;
}*/	

int main(){
	char Name1[10] = "pulse.txt";
	int Length = numLines(Name1);
	char Name2[10] = "sample.txt";
	double timestep = TS(Name1) * pow(10,-12);
	MaxT = timestep* (double)(Length-1);
	double *ER;
	double *ES;
	int NFFT = pow(2,nextPow2(Length));
	ER = readE(Length,Name1);
	ES = readE(Length,Name2);
	double complex *ES2;
	double complex *ER2;

	ER2 = toComp(NFFT,EPadder(Length,NFFT,ER));
	ES2 = toComp(NFFT,EPadder(Length,NFFT,ES));

	double complex * ERF;
	double complex * ESF;

	ERF = FFTShift(NFFT,FFT(NFFT,FFTShift(NFFT,ER2)));
	ESF = FFTShift(NFFT,FFT(NFFT,FFTShift(NFFT,ES2)));
	
	double *freq;
	freq = FREQShifted(NFFT,timestep);
	
	int ZoneStart = SearchStart(NFFT,0.3*pow(10,12),freq);
	Zone = NewLenScan(NFFT,ZoneStart,0.7*pow(10,12),freq);

	double complex * ERF2;
	double complex * ESF2;

	freq2 = ChopShopF(NFFT,Zone,ZoneStart,freq);
	ERF2 = ChopShopE(NFFT,Zone,ZoneStart,ERF);
	ESF2 = ChopShopE(NFFT,Zone,ZoneStart,ESF);
	
	double complex *Hex;
	Hex = Hexp(Zone,ERF2,ESF2);	

	HME = Amplitude(Zone,Hex);
	HPE = Phase(Zone,Hex);
	Unwrap(Zone,HPE);
	int PosS = findAMP(Length,ES);
	int PosR = findAMP(Length,ER);
	
	Delta = timestep*(double)(PosS-PosR);
	MaxR = abs(ER[PosR]);
	MaxS = abs(ES[PosS]);
	
	/*double lNew[1];
	double keeper[1];
	keeper[0] = SamL;

	double reqmin2 = 1.0E-08;
	int konvge2 = 1;
	int kcount2 = 1;
	double ynewlo2 = Smoothness(keeper);
	double StepN2[1];
	StepN2[0] = 0.001;
	int icount2;
	int ifault2;
	int numres2;
	nelmin(Smoothness,1,keeper,lNew,&ynewlo2,reqmin2,StepN2,konvge2,kcount2,&icount2,&numres2,&ifault2);
	SamL = lNew[0];*/
	/*Test Case*/
	nCalc = (double complex *)malloc(Zone*sizeof(double complex));
	currentSpace = 0;
	int i;
	printf("Start2\n");
	for(i=0;i<Zone;i++){
		nCalc[i] = nInitgen() - I*kInitgen(freq2[i]);
	}
	printf("Step1\n");
	double reqmin = 1.0E-04;
	int konvge = 5;
	int kcount = 400;
	double ynewlo; /*= ERR(nCalc);*/
	double nMin[2];
	double nTemp[2];
	double StepN[2];
	int icount;
	int ifault;
	int numres;
	for(i=0;i<2;i++){
		StepN[i] = 0.05;
	}
	printf("Step2\n");
	for(i=0;i<Zone;i++){
		nTemp[0] = creal(nCalc[i]);
		ScaleUp = log10(-cimag(nCalc[i]));
		nTemp[1] = -cimag(nCalc[i]*pow(10,-ScaleUp));
		ynewlo = ERR(nTemp);
		nelmin(ERR,2,nTemp,nMin,&ynewlo,reqmin,StepN,konvge,kcount,&icount,&numres,&ifault);
		nCalc[i] = nMin[0] - I*nMin[i]*pow(10,ScaleUp);
		printf("%f -I%f\n",creal(nCalc[i]),-cimag(nCalc[i]));
		currentSpace++;
	}
	/**/
	/*TODO: Nelder-Mead for n here */
	
	/*TODO: OutPut data and plot*/
	FILE *fp;
	int j;
	fp = fopen("NewTest1.txt","w+");
	for(j=0;j<Zone;j++){
		fprintf(fp,"%f\t%f\t%f\n",freq2[j]*pow(10.0,-12),creal(nCalc[j]),creal(nCalc[j]));
	}
	fclose(fp);
	plotNcomp2("NewTest1.txt");
	fp = fopen("NewTest2.txt","w+");
	for(j=0;j<Zone;j++){
		fprintf(fp,"%f\t%f\t%f\n",freq2[j]*pow(10.0,-12),-cimag(nCalc[j]),-cimag(nCalc[j]));
		/*printf("%f\t%f\n",freq2[j]*pow(10.0,-12),kInitgen(freq2[i]));*/
		/*fprintf(fp,"%f\t%f\t%f\n",freq2[j]*pow(10.0,-12),kInitgen(freq2[j]),-cimag(nCalc[j]));*/
	}
	fclose(fp);
	plotKcomp2("NewTest2.txt");
	/*printf("%d %f %f\n",Zone,nCalc[0],Delta*pow(10,12));*/
	printf("%f\n",SamL);
	
	return 0;
}