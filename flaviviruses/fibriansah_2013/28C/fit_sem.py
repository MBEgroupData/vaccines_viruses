import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from scipy.optimize import minimize
import sys
import matplotlib.pyplot as plt
import scipy.special
from os import walk
import math
import cv2
from scipy.signal import square

curve_types = ["Gaussian","Loretenzian","Skewed Gaussian"]

def preview_hist(imageFile):
    im = cv2.imread(imageFile)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])

    x1 = np.arange(0,256,1)
    y1 = np.array(hist).reshape(1,256)[0]

    plt.plot(x1,y1)
    plt.show()
    plt.close()

def preview_hist_csv(csvFile):
    file = open(csvFile,'r')
    lines = file.readlines()
    x1=[]
    y1=[]
    for line in lines[1:]:
        parts = line.split(",")
        x1.append(float(parts[0]))
        y1.append(float(parts[1]))


    x1 = np.array(x1)
    y1 = np.array(y1)

    plt.plot(x1,y1)
    plt.show()
    plt.close()


def readParms(parmsFile):
    file = open(parmsFile)
    lines = file.readlines()

    numCurves = int(lines[0].split(":")[1])
    types=[]
    parms=[]
    bnds_high=[]
    bnds_low=[]
    range_list=[]
    adder=0
    for i in range(0,numCurves):
        start = 15*i+2+adder
        type=lines[start].split(":")[1][:-1]
        types.append(type)
        amp = float(lines[start+2].split(":")[1])
        cent = float(lines[start+6].split(":")[1])
        parms.extend([amp,cent])

        amp_low = float(lines[start+3].split(":")[1])
        amp_high = float(lines[start+4].split(":")[1])
        cent_low = float(lines[start+7].split(":")[1])
        cent_high = float(lines[start+8].split(":")[1])

        bnds_low.extend([amp_low,cent_low])
        bnds_high.extend([amp_high,cent_high])

        if type!=curve_types[2]:
            sig =  float(lines[start+10].split(":")[1])
            parms.append(sig)
            sig_low = float(lines[start+11].split(":")[1])
            sig_high = float(lines[start+12].split(":")[1])
            bnds_low.append(sig_low)
            bnds_high.append(sig_high)
            adder+=0

        else:
            sig1 = float(lines[start+10].split(":")[1])
            sig2 = float(lines[start+14].split(":")[1])
            parms.extend([sig1,sig2])
            sig_low = float(lines[start+11].split(":")[1])
            sig_high = float(lines[start+12].split(":")[1])
            sig2_low = float(lines[start+15].split(":")[1])
            sig2_high = float(lines[start+16].split(":")[1])
            bnds_low.extend([sig_low,sig2_low])
            bnds_high.extend([sig_high,sig2_high])
            adder+=4

    c  = float(lines[-3].split(":")[1])
    low_c  = float(lines[-2].split(":")[1])
    high_c  = float(lines[-1].split(":")[1])
    parms.append(c)
    bnds_low.append(low_c)
    bnds_high.append(high_c)

    range_list.extend([int(lines[-6].split(":")[1]),int(lines[-5].split(":")[1])])

    file.close()

    bnds = [bnds_low,bnds_high]
    return types,parms,bnds,range_list

def step(amp,dutyCycle,freq,x):
	return amp*square(2*np.pi*freq*x,dutyCycle)

def skewNorm(a,mean,sd1,sd2,x):
	  norm = []
	  middle=mean
	  for i in range(len(x)):
		  if x[i]>=middle:
			  norm += [np.abs(a)*np.exp(-(x[i] - mean)**2/(2*sd2**2))]
		  elif x[i] < middle:
			  norm  += [np.abs(a)*np.exp(-(x[i] - mean)**2/(2*sd1**2))]
	  return np.array(norm)

def skewNormNoArray(a,mean,sd1,sd2,x):
		middle=mean
		if x>=middle:
		    return  np.abs(a)*np.exp(-(x - mean)**2/(2*sd2**2))
		elif x < middle:
			return np.abs(a)*np.exp(-(x - mean)**2/(2*sd1**2))

def normNoArray(a,mean,sd,x):
	return a*np.exp(-(x - mean)**2/(2*sd**2))

def loretenNoArray(a,p0,w,x):
	return np.abs(a)*1.0/(1.0+((p0-x)/(w/2.0))**2)

def norm(a,mean,sd,x):
  norm = []
  for i in range(len(x)):
    norm += [np.abs(a)*np.exp(-(x[i] - mean)**2/(2*sd**2))]
  return np.array(norm)

def loreten(a,p0,w,x):
	lor=[]
	for i in range(len(x)):
		lor+=[np.abs(a)*1.0/(1.0+((p0-x[i])/(w/2.0))**2)]
	return np.array(lor)

def fitting_func(p,x,y,types):
    res = np.zeros(len(x))
    adder=0
    for i in range(0,len(types)):
        if types[i] == curve_types[0]:
            res = np.add(res,norm(p[3*i+adder],p[3*i+1+adder],p[3*i+2+adder],x))
        elif types[i] == curve_types[1]:
            res = np.add(res,loreten(p[3*i+adder],p[3*i+1+adder],p[3*i+2+adder],x))
        elif types[i] == curve_types[2]:
            res = np.add(res,skewNorm(p[3*i+adder],p[3*i+1+adder],p[3*i+2+adder],p[3*i+3+adder],x))
            adder+=1
        res=np.add(res,p[-1])

    return res


def get_threshold(p,types,cross_point):
    adder =0
    if types[0] == "Skewed Gaussian":
        adder+=1

    threshold =p[3+1+adder]
    sigma = p[3+2+adder]

    numSigma = (threshold-cross_point)/sigma

    if numSigma > 1.0:
        threshold = threshold-np.floor(numSigma)*sigma

    return threshold


def get_curves(p,x,types):
    res1 = np.zeros(len(x))
    res2 = np.zeros(len(x))
    adder=0
    if types[0] == curve_types[0]:
        res1 = np.add(res1,norm(p[adder],p[1+adder],p[2+adder],x))
    elif types[0] == curve_types[1]:
        res1 = np.add(res1,loreten(p[adder],p[1+adder],p[2+adder],x))
    elif types[0] == curve_types[2]:
        res1 = np.add(res1,skewNorm(p[adder],p[1+adder],p[2+adder],p[3+adder],x))
        adder+=1

    if types[1] == curve_types[0]:
        res2 = np.add(res2,norm(p[3+adder],p[3+1+adder],p[3+2+adder],x))
    elif types[1] == curve_types[1]:
        res2 = np.add(res2,loreten(p[3+adder],p[3+1+adder],p[3+2+adder],x))
    elif types[1] == curve_types[2]:
        res2 = np.add(res2,skewNorm(p[3+adder],p[3+1+adder],p[3+2+adder],p[3+3+adder],x))
        adder+=1

    return res1,res2


def print_curves_to_files(fitParms,x,y1,y_fit1,types):
    y_c1,y_c2 = get_curves(fitParms,x,types)
    file_1 = open("curve_1_fit.dat",'w')
    file_2 = open("curve_2_fit.dat",'w')
    file_3 = open("curve_total_fit.dat",'w')
    file_4 = open("pixel_intensity_hist.csv","w")

    for i in range(0,len(x)):
        file_1.write(str(x[i])+","+str(y_c1[i])+"\n")
        file_2.write(str(x[i])+","+str(y_c2[i])+"\n")
        file_3.write(str(x[i])+","+str(y_fit1[i])+"\n")
        file_4.write(str(x[i])+","+str(y1[i])+"\n")

    file_1.close()
    file_2.close()
    file_3.close()
    file_4.close()


def print_curves_to_files_csv(fitParms,x,y1,y_fit1,types):
    y_c1,y_c2 = get_curves(fitParms,x,types)
    file_1 = open("curve_1_fit.dat",'w')
    file_2 = open("curve_2_fit.dat",'w')
    file_3 = open("curve_total_fit.dat",'w')

    for i in range(0,len(x)):
        file_1.write(str(x[i])+","+str(y_c1[i])+"\n")
        file_2.write(str(x[i])+","+str(y_c2[i])+"\n")
        file_3.write(str(x[i])+","+str(y_fit1[i])+"\n")

    file_1.close()
    file_2.close()
    file_3.close()



def print_parms_to_file(fitParms,types,thresh,s2):
    print(s2)
    file = open("summary.md",'r')
    lines = file.readlines()
    file.close()
    adder=0
    lines[0] = "## Pixel Intensity Histogram Fit\n"
    lines[1] = "\n"
    lines[2] = "#### Curve 1 (Disordered Peak)\n"
    lines[3] = "**Type**: "+types[0]+"\\\n"
    lines[4] = "**amplitude:** {:.1f}\\\n".format(np.abs(fitParms[0]))
    lines[5] = "**center** {:.1f}\\\n".format(np.abs(fitParms[1]))
    lines[6] = "**σ** {:.1f}\n".format(np.abs(fitParms[2]))
    if types[0] == curve_types[2]:
        lines[7] = "**σ2** {:.1f}\n".format(np.abs(fitParms[3]))
        adder+=1
    lines[7+adder] = "\n\n"
    lines[8+adder] = "#### Curve 2 (Ordered Peak)\n"
    lines[9+adder] = "**Type**:"+types[1]+"\\\n"
    lines[10+adder] = "**amplitude** {:.1f}\\\n".format(np.abs(fitParms[3+adder]))
    lines[11+adder] = "**center** {:.1f}\\\n".format(np.abs(fitParms[4+adder]))
    lines[12+adder] = "**σ** {:.1f}\\\n".format(np.abs(fitParms[5+adder]))
    if types[1] == curve_types[2]:
        lines[13+adder] = "**σ2** {:.1f}\n".format(np.abs(fitParms[6+adder]))
        adder+=1
    lines[13+adder] = "\n\n"

    lines[14+adder] = "## S<sup>2</sup>\n"

    lines[15+adder]="**Threshold Value:** {:d}\\\n".format(int(np.round(thresh)))
    lines[16+adder]="**Ordered Area Percentage:** {:2.2f}%\\\n".format(s2*100)
    lines[17+adder]="**S<sup>2</sup>:** {:0.4f}\n".format(s2)
    lines[18+adder] = "\n"
    lines[19+adder] = "\n"
    lines[20+adder] = "\n"

    file = open("summary.md",'w')
    file.writelines(lines)
    file.close()


def res(p, y, x):
	y_fit =fitting_func(p,x,y,types)
	err = y - y_fit
	return err


def gausIntersect(fitParms,types,debug=True):
    intersect = 0

    if types[0] != curve_types[2] and types[1] != curve_types[2]:
        a = fitParms[5]**2 - fitParms[2]**2
        b = 2*(fitParms[4]*fitParms[2]**2 - fitParms[1]*fitParms[5]**2)
        c = -fitParms[4]**2*fitParms[2]**2 + fitParms[1]**2*fitParms[5]**2 - 2*np.log(np.abs(fitParms[0])/np.abs(fitParms[3]))*fitParms[2]**2*fitParms[5]**2

        res1 = (-b + np.sqrt(b**2-4*a*c))/(2*a)
        res2 = (-b - np.sqrt(b**2-4*a*c))/(2*a)

        if debug:
            print("- root:"+str(res1))
            print("+ root:"+str(res2))
            print("\n")

        intersect = res1

    elif types[0] == curve_types[2] and types[1] != curve_types[2]:
        a = fitParms[6]**2 - fitParms[3]**2
        b = 2*(fitParms[5]*fitParms[3]**2 - fitParms[1]*fitParms[6]**2)
        c = -fitParms[5]**2*fitParms[3]**2 + fitParms[1]**2*fitParms[6]**2 - 2*np.log(np.abs(fitParms[0])/np.abs(fitParms[4]))*fitParms[3]**2*fitParms[6]**2

        res1 = (-b + np.sqrt(b**2-4*a*c))/(2*a)
        res2 = (-b - np.sqrt(b**2-4*a*c))/(2*a)

        if debug:
            print("- root:"+str(res1))
            print("+ root:"+str(res2))
            print("\n")

        intersect = res2

    elif types[0] == curve_types[2] and types[1] == curve_types[2]:
        a = fitParms[6]**2 - fitParms[3]**2
        b = 2*(fitParms[5]*fitParms[3]**2 - fitParms[1]*fitParms[6]**2)
        c = -fitParms[5]**2*fitParms[3]**2 + fitParms[1]**2*fitParms[6]**2 - 2*np.log(np.abs(fitParms[0])/np.abs(fitParms[4]))*fitParms[3]**2*fitParms[6]**2

        res1 = (-b + np.sqrt(b**2-4*a*c))/(2*a)
        res2 = (-b - np.sqrt(b**2-4*a*c))/(2*a)

        if debug:
            print("- root:"+str(res1))
            print("+ root:"+str(res2))
            print("\n")

        intersect = res1

    return intersect


def printFits(fitRes,types):
    adder=0
    for i in range(0,len(types)):
        print("Curve: "+types[i])
        if types[i] != curve_types[2]:
            print("Amplitude: {:.0f}".format(fitRes[3*i+adder]))
            print("Center: {:.6f}".format(fitRes[3*i+1+adder]))
            print("Sigma: {:.6f}".format(fitRes[3*i+2+adder]))
            print("\n")

        elif types[i] == curve_types[2]:
            print("Amplitude: {:.0f}".format(fitRes[3*i+adder]))
            print("Center: {:.6f}".format(fitRes[3*i+1+adder]))
            print("Sigma: {:.6f}".format(fitRes[3*i+2+adder]))
            print("Sigma2: {:.6f}".format(fitRes[3*i+3+adder]))
            print("\n")
            adder+=1


def fit_pixel_hist(imageFile,types,paramsToFit,bnds,range_list,debug=True):
    im = cv2.imread(imageFile)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    x1 = np.arange(0,256,1)
    y1 = np.array(hist).reshape(1,256)[0]

    x1 = x1[range_list[0]:range_list[1]]
    y1 = y1[range_list[0]:range_list[1]]

    fit1 = least_squares(res,paramsToFit,args=(y1,x1),bounds=bnds)

    y_fit1 = fitting_func(fit1.x,x1,y1,types)

    crossPoint = gausIntersect(fit1.x,types,debug)
    threshold_value = get_threshold(fit1.x,types,crossPoint)


    ret,thresh1 = cv2.threshold(gray,threshold_value,255,cv2.THRESH_BINARY)
    nzCount = cv2.countNonZero(thresh1)
    imgSize = thresh1.size
    s2 = nzCount/float(imgSize)

    if debug:
        printFits(fit1.x,types)
        print("\n")
        print("Intersection Point; {:.6f}".format(crossPoint))
        print("Threshold: "+str(threshold_value))
        print("\n")
        print("S^2: "+str(s2)+"\n\n")

        y_c1,y_c2 = get_curves(fit1.x,x1,types)
        plt.plot(x1,y1)
        plt.plot(x1,y_c1,'b')
        plt.plot(x1,y_c2,'b')
        plt.plot(x1,y_fit1)
        plt.show()
        plt.close()
    else:
        print_curves_to_files(fit1.x,x1,y1,y_fit1,types)
        print_parms_to_file(fit1.x,types,threshold_value,s2)
        cv2.imwrite("sem_thresholded.png",thresh1)




def fit_pixel_hist_csv(csvFile,types,paramsToFit,bnds,range_list,debug=True):
    file = open(csvFile,'r')
    lines = file.readlines()
    x1=[]
    y1=[]
    for line in lines[1:]:
        parts = line.split(",")
        x1.append(float(parts[0]))
        y1.append(float(parts[1]))


    x1 = np.array(x1)
    y1 = np.array(y1)

    x1 = x1[range_list[0]:range_list[1]]
    y1 = y1[range_list[0]:range_list[1]]

    fit1 = least_squares(res,paramsToFit,args=(y1,x1),bounds=bnds)

    y_fit1 = fitting_func(fit1.x,x1,y1,types)

    crossPoint = gausIntersect(fit1.x,types,debug)
    threshold_value = get_threshold(fit1.x,types,crossPoint)

    totals = np.sum(y1)
    bright = np.sum(y1[int(np.round(threshold_value)):])
    s2 = bright/totals

    if debug:
        printFits(fit1.x,types)
        print("\n")
        print("Intersection Point; {:.6f}".format(crossPoint))
        print("Threshold: "+str(threshold_value))
        print("\n")
        print("S^2: "+str(s2)+"\n\n")

        y_c1,y_c2 = get_curves(fit1.x,x1,types)
        plt.plot(x1,y1)
        plt.plot(x1,y_c1,'b')
        plt.plot(x1,y_c2,'b')
        plt.plot(x1,y_fit1)
        plt.show()
        plt.close()
    else:
        print_curves_to_files_csv(fit1.x,x1,y1,y_fit1,types)
        print_parms_to_file(fit1.x,types,threshold_value,s2)



if __name__ == "__main__":
    switch = sys.argv[1]
    if switch == "init":
        preview_hist_csv("pixel_intensity_hist.csv")
    elif switch == "test":
        types,parms,bnds,range_list=readParms("parms.dat")
        fit_pixel_hist_csv("pixel_intensity_hist.csv",types,parms,bnds,range_list)
    elif switch == "final":
        types,parms,bnds,range_list=readParms("parms.dat")
        fit_pixel_hist_csv("pixel_intensity_hist.csv",types,parms,bnds,range_list,debug=False)
    else:
        print("No option provided!")
