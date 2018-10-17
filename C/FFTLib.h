#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

double e = M_E;
double pi = M_PI;

int nextPow2(int a){
	double b = log2(a);
	double c = (double)((int)b);
	if(b==c){
		return (int)b;
	}
	else{
		return (int)b+1;
	}
}

double complex * DFT(int len, double complex *y){
	static double complex *r;
	r = (double complex *)malloc(len*sizeof(double complex));
	int j;
	int k;
	double complex read;
	for (j=0;j<len;j++){
		for(k=0;k<len;k++){
			if(y[k] != y[k]){
				read = 0.0;
			}
			else{
				read = y[k];
			}
			r[j] += read*cexp(-2.0*pi*I*(double)j*(double)k/(double)len);
		}
	}
	return r;
}

double complex * FFT(int len, double complex *y){
	if(len<=32){
		return DFT(len,y);
	}
	else{
		int len2 = (int)((double)len/2.0);
		double complex *Even;
        	Even = (double complex *)malloc((len2)*sizeof(double complex));
		double complex *Odd;
        	Odd = (double complex *)malloc((len2)*sizeof(double complex));
		double complex *even;
		double complex *odd;
		int j;
		int count=0;
		for(j=0;j<len;j+=2){
			Even[count] = y[j];
			Odd[count] = y[j+1];
			count++;
		}
		even = FFT(len2,Even);
		odd = FFT(len2,Odd);
		static double complex *r;
	        r = (double complex *)malloc(len*sizeof(double complex));
		for(j=0;j<len2;j++){
			r[j] = even[j] + cexp(-2*I*pi*((double)j)/((double)len))*odd[j];
			r[j+len2] =even[j] + cexp(-2*I*pi*((double)(j+len2))/((double)len))*odd[j];
		}
		return r;

	}
}

double cAngle(double complex Y){
	return cimag(clog(Y/cabs(Y)));
}

void Unwrap(int len, double *Ang){
	int j;
	int k;
	for(j=1;j<len;j++){
		if(Ang[j]-Ang[j-1] > pi){
			for(k=j;k<len;k++){
				Ang[k] -= 2*pi;
			}
		}
		else if(Ang[j]-Ang[j-1] < -pi){
			for(k=j;k<len;k++){
				Ang[k] += 2*pi;
			}
		}
	}
}

double complex * iDFT(int len, double complex *y){
        static double complex *r;
        r = (double complex *)malloc(len*sizeof(double complex));
        int j;
        int k;
        for (j=0;j<len;j++){
                for(k=0;k<len;k++){
                        r[j] += (1.0/((double)len))*y[k]*cexp(2.0*pi*I*(double)j*(double)k/((double)len));
                }
        }
	return r;
}

int numLines(char * Iname){
	FILE *fp;
	fp = fopen(Iname,"r+");
	int lines =0;
	int c;
	if(fp){
		while((c=getc(fp)) != EOF){
			if((char)c == '\n'){
				lines++;
			}
		}
		fclose(fp);
	}
	lines++;
	return lines;
}

double * readE(int Lines,char * Iname){
	FILE *fp;
	fp = fopen(Iname,"r+");
	static double *E;
	int count=0;
	int NFFT = pow(2,nextPow2(Lines-1));
        E = (double *)malloc((NFFT)*sizeof(double));
	int start = (int)(((double)(NFFT))/2.0-((double)(Lines-1))/2.0);
	for(count=0;count<NFFT;count++){
		E[count] = 0.0;
	}
	int c;
	count=0;
	int place=0;
	int lineNum = 0;
	char temp[10];
	for(place=0;place<10;place++){
		temp[place] = '0';
	}
	if(fp){
                while((c=getc(fp)) != EOF){
                        if((char)c == '\t'){
                                if(count==0){
					for(place=0;place<8;place++){
               					temp[place] = '0';
				        }
					place=0;
					count++;
				}
				else if(count==1){
					if(lineNum != 0){
					E[lineNum-1+start] = (double)atof(temp);}
                                        for(place=0;place<10;place++){
                                                temp[place] = '0';
                                        }
                                        count++;
                                        place=0;

				}
                        }
			else if((char)c=='\n'){
				lineNum++;
				count = 0;
                                for(place=0;place<10;place++){
                                	temp[place] = '0';
                                }
				place=0;
			}
			else{
				temp[place] = (char)c;
				place++;
			}
                }
                fclose(fp);
        }
	return E;
}

double TS(char * Iname){
	FILE *fp;
	fp = fopen(Iname,"r+");
	double Time1;
	double Time0;
	int c;
	char temp[10];
	int count=0;
	int place=0;
	int lineNum = 0;
	for(place=0;place<10;place++){
		temp[place] = '0';
	}
	place = 0;
	if(fp){
                while((c=getc(fp)) != EOF){

                        if((char)c == '\t'){
                                if(count==0){
					if(lineNum == 1){
						Time0 = (double)atof(temp);
					}
					else if(lineNum == 2){
						Time1 = (double)atof(temp);
						break;
					}
					for(place=0;place<10;place++){
               					temp[place] = '0';
				        }
					count++;
					place=0;

				}
				else if(count==1){
                                        for(place=0;place<10;place++){
                                                temp[place] = '0';
                                        }
                                        count++;
                                        place=0;

				}
                        }
			else if((char)c=='\n'){
				lineNum++;
				count = 0;
                                for(place=0;place<10;place++){
                                	temp[place] = '0';
                                }
				place=0;
			}
			else{
				temp[place] = (char)c;
				place++;
			}
                }
                fclose(fp);
        }
	return (Time1-Time0);
}

double complex * iFFT(int len, double complex *y){
	double complex *A;
	double complex *Temp;
	Temp = (double complex *)malloc(len*sizeof(double complex));
	int j;
	for(j=0;j<len;j++){
		Temp[j] = conj(y[j]);
	}
	A = FFT(len,Temp);
	for(j=0;j<len;j++){
		A[j] = conj(A[j])/((double)len);
	}
	return A;
}

double complex * toComp(int len, double *y){
	int j;
	static double complex *A;
	A = (double complex *)malloc(len*sizeof(double complex));
	for(j=0;j<len;j++){
		A[j] = y[j] + 0*I;
	}
	return A;
}

double complex * FFTShift(int len, double complex *y){
	int j;
	static double complex *A;
	int len2 = len/2;
	A = (double complex *)malloc(len*sizeof(double complex));
	for(j=0;j<len2;j++){
		A[j+len2] = y[j];
		A[j] = y[j+len2];
	}
	return A;
}

void plotAmp(char *Oname){
	FILE * gnuplotPipe = popen("gnuplot -persistent","w");
	fprintf(gnuplotPipe, "set ylabel \"Amplitude (a.u.)\"\n set xlabel \"Frequency (THz)\"\n set logscale y\n set xrange [0.0:10.0]\n plot \"%s\" using 1:2 title \'Amplitude\' with lines\n quit\n",Oname);
	fclose(gnuplotPipe);
}

void plotPhase(char *Oname){
	FILE * gnuplotPipe = popen("gnuplot -persistent","w");
	fprintf(gnuplotPipe, "set ylabel \"Phase Angle (rads)\"\n set xlabel \"Frequency (THz)\"\n set xrange [0.0:10.0]\n plot \"%s\" using 1:3 title \'Phase\' with lines\n quit\n",Oname);
	fclose(gnuplotPipe);
}

double * Amplitude(int len, double complex *y){

	int j;
	static double *A;
	A = (double *)malloc(len*sizeof(double));
	for(j=0;j<len;j++){
		A[j] = cabs(y[j]);
	}
	return A;

}

double * Phase(int len, double complex *y){

	int j;
	static double *A;
	A = (double *)malloc(len*sizeof(double));
	for(j=0;j<len;j++){
		A[j] = cAngle(y[j]);
	}
	return A;

}