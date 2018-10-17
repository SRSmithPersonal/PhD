#include <stdio.h>
#include <complex.h>
#include <math.h>
#include "FFTLib.h"

double complex theta = 60.0*M_PI/180.0 +0.0*I;
double c = 299792458.0;
double complex n_i = 1.0+0.0*I;

double complex thetaI(double complex n2){
	return casin(n_i*csin(theta)/n2);
}

double complex rp(double complex n1,double complex n2){
	double complex thetaIn = thetaI(n1);
	double complex thetaUit = thetaI(n2);
	return (n2*ccos(thetaIn)-n1*ccos(thetaUit))/(n2*ccos(thetaIn)+n1*ccos(thetaUit));
}

double complex rs(double complex n1, double complex n2){
	double complex thetaIn = thetaI(n1);
	double complex thetaUit = thetaI(n2);
    	return (n1*ccos(thetaIn)-n2*ccos(thetaUit))/(n1*ccos(thetaIn)+n2*ccos(thetaUit));
}

double complex tp(double complex n1, double complex n2){
	double complex thetaIn = thetaI(n1);
	double complex thetaUit = thetaI(n2);
    	return (2*n1*ccos(thetaIn))/(n2*ccos(thetaIn)+n1*ccos(thetaUit));
}

double complex ts(double complex n1, double complex n2){
	double complex thetaIn = thetaI(n1);
	double complex thetaUit = thetaI(n2);
    	return (2*n1*ccos(thetaIn))/(n1*ccos(thetaIn)+n2*ccos(thetaUit));
}

void setTheta(double complex nT){
	theta = nT;
}

void setN_i(double complex nT){
	n_i = nT;
}

double abso(double k, double freq){
    return 2.0*pi*freq*k/(c*100.0);
}

double kgen(double alpha, double freq){
    return 100.0*alpha*c/(2.0*pi*freq);
}