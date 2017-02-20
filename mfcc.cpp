#include <iomanip>  //For setprecision
#include <fstream>
#include <math.h>
#include "mfcc.h"

using namespace std;

const double Mfcc::PI  = 3.14159265358979323846;
const double Mfcc::PI2 = 6.28318530717958647692;
const double Mfcc::PI4 = 12.56637061435917295384;

Mfcc::Mfcc() : Mfcc(16000, 16, 8, WindowMethod::Hamming, 24, 12)
{
    //m_FrameSize must be a power of 2 !!
    // or modify mfcc::fft()
}

Mfcc::Mfcc(int freq, int size, int shift, WindowMethod method, int filterNum, int mfccDim)
{
    m_Frequence = freq;
    m_FrameSize = freq*size/1000;
    m_FrameShift= freq*shift/1000;
    m_FilterNumber=filterNum;
    m_MFCCDim   = mfccDim;

    m_FrameCount  = 0;
    m_CurrentFrame= 0;

    setWindowMethod(method);
    setFilterBank();
    setDCTCoeff();
    setLiftCoeff();
}

Mfcc::~Mfcc()
{
    m_WindowCoefs.clear();
    m_FilterBank.clear();
    m_DCTCoeff.clear();
    m_CepLifter.clear();
    m_MFCCData.clear();
    m_RestData.clear();
}

size_t Mfcc::Analyse(const short int data[], size_t sizeData)
{
    vector<vector<double>> postData;

    ///*** Initialisation
    m_MFCCData.clear();
    m_CurrentFrame = 0;
    m_FrameCount = (sizeData-m_FrameSize+m_FrameShift)/m_FrameShift;
    m_MFCCData.resize(m_FrameCount);
    postData.resize(m_FrameCount);

    ///*** Apply the window coefficients
    for(size_t i=0; i<m_FrameCount; i++)
    {
        for(int j=0; j<m_FrameSize; j++)
        {
            postData[i].push_back(data[i*m_FrameShift+j]*m_WindowCoefs[j]);
            postData[i].push_back(0);
        }
    }

    ///*** Analyse
    internalAnalyse(postData, m_FrameCount, 0);
    postData.clear();

    return m_FrameCount;
}

void Mfcc::internalAnalyse(vector<vector<double>>& postData, size_t frameCount, size_t currrentFrame)
{
    vector<vector<double>> spectraP;
    vector<vector<double>> melSpectraP;


    ///*** FFT matrix
    for(size_t i=0; i<frameCount; i++)
        fft(postData[i],m_FrameSize,1);

    ///*** Energy matrix
    spectraP.resize(frameCount);

    for(size_t i=0; i<frameCount; i++)
        for(int j=0; j<m_FrameSize/2+1; j++)
            spectraP[i].push_back(postData[i][j<<1]*postData[i][j<<1]+postData[i][(j<<1)+1]*postData[i][(j<<1)+1]);

    melSpectraP.resize(m_FilterNumber);

    ///*** Apply filter bank
    for(int i=0; i<m_FilterNumber; i++)
    {
        for(size_t k=0; k<frameCount; k++)
        {
            melSpectraP[i].push_back(0);
            for(int j=0; j<m_FrameSize/2+1; j++)
                melSpectraP[i][k]+=m_FilterBank[i][j]*spectraP[k][j];

            melSpectraP[i][k]=log(melSpectraP[i][k]);
        }
    }

    spectraP.clear();

    ///*** Mfcc matrix
    for(size_t k=0; k<frameCount; k++)
    {
        for(int i=0; i<m_MFCCDim; i++)
        {
            m_MFCCData[currrentFrame+k].push_back(0);
            for(int j=0; j<m_FilterNumber; j++)
                m_MFCCData[currrentFrame+k][i]+=m_DCTCoeff[i][j]*melSpectraP[j][k];
        }
    }
    melSpectraP.clear();

    ///*** Ceplift
    for(size_t i=0; i<frameCount; i++)
        for(int j=0; j< m_MFCCDim; j++)
            m_MFCCData[currrentFrame+i][j]*=m_CepLifter[j];

    return;
}

bool Mfcc::Save(const string& filePath)
{
	size_t frameCount;
    ofstream outFile(filePath);
    if(!outFile.is_open())
        return false;

    outFile << fixed << std::setprecision(6);

    if(m_CurrentFrame > 0 )
        frameCount = m_CurrentFrame;
    else
        frameCount = m_FrameCount;

    for(size_t i=0; i<frameCount; i++)
    {
        for(int j=0; j<m_MFCCDim; j++)
            outFile << m_MFCCData[i][j] << " ";
        outFile << endl;
    }

    outFile.close();
    return true;
}

const std::vector<std::vector<double>>& Mfcc::GetMFCCData()
{
    return m_MFCCData;
}

void Mfcc::StartAnalyse(size_t maxSize)
{
    m_CurrentFrame = 0;
    m_FrameCount = (maxSize-m_FrameSize+m_FrameShift)/m_FrameShift;
    m_MFCCData.clear();
    m_MFCCData.resize(m_FrameCount);

    m_RestData.clear();
}

bool Mfcc::AddBuffer(const short int data[], size_t sizeData)
{
    vector<vector<double>> postData;
    size_t restSize = m_RestData.size();
    size_t frameCount;
	size_t i, k;
    int j;


    ///*** Initialisation
    if(m_FrameCount==0) return false;
    frameCount = (sizeData+restSize-m_FrameSize+m_FrameShift)/m_FrameShift;
    if(m_CurrentFrame+frameCount>m_FrameCount)
    {
        frameCount = m_FrameCount-m_CurrentFrame;
        if(frameCount==0) return false;
    }
    postData.resize(frameCount);

    ///*** Apply the window coefficients
    for(i=0; i<frameCount; i++)
    {
        for(j=0; j<m_FrameSize; j++)
        {
            k = i*m_FrameShift+j;
            if(k<restSize)
                postData[i].push_back(m_RestData[k]*m_WindowCoefs[j]);
            else
                postData[i].push_back(data[i*m_FrameShift+j-restSize]*m_WindowCoefs[j]);

            postData[i].push_back(0);
        }
    }
    m_RestData.clear();

    ///*** Analyse
    internalAnalyse(postData, frameCount, m_CurrentFrame);
    postData.clear();
    m_CurrentFrame += frameCount;
    if(m_CurrentFrame>=m_FrameCount) return false;

    ///*** Memorize the remainder
    restSize = (sizeData+restSize)-frameCount*m_FrameShift;
    if(restSize>0)
    {
        for(i=sizeData-restSize; i<sizeData; i++)
            m_RestData.push_back(data[i]);
    }

    return true;
}

size_t Mfcc::GetFrameCount()
{
    return m_CurrentFrame;
}

void Mfcc::fft(vector<double>& data, int nn, int isign)
{
	int i,j,m,n,mmax,istep;
	double wtemp,wr,wpr,wpi,wi,theta;
	double tempr,tempi;


	n=nn<<1;
	j=1;
	for(i=1;i<n;i+=2)
	{
		if(j>i)
		{
		    swap(data[j-1], data[i-1]);
			swap(data[j], data[i]);
		}
		m=nn;

		while(m>=2&&j>m)
		{
            j-=m;
			m>>=1;
        }
		j+=m;
	}

	mmax=2;
	while(n>mmax)
	{
		istep=mmax<<1;
		theta=isign*(PI2/mmax);
		wtemp=sin(0.5*theta);
		wpr=-2.0*wtemp*wtemp;
		wpi=sin(theta);
		wr=1.0;
		wi=0.0;
		for(m=1;m<mmax;m+=2)
		{
			for(i=m;i<=n;i+=istep)
			{
			    if(i>=n) continue;      //Buffer overflow if nn is not a power of 2
   			    j=i+mmax;
   			    if(j>=n) continue;      //Buffer overflow if nn is not a power of 2
			    tempr=wr*data[j-1]-wi*data[j];
			    tempi=wr*data[j]+wi*data[j-1];
                data[j-1]=data[i-1]-tempr;
			    data[j]=data[i]-tempi;
			    data[i-1]+=tempr;
			    data[i]+=tempi;
			}
			wr=(wtemp=wr)*wpr-wi*wpi+wr;
			wi=wi*wpr+wtemp*wpi+wi;
		}
		mmax=istep;
	}
}

void Mfcc::setWindowMethod(WindowMethod method)
{
    m_WindowCoefs.clear();

    switch(method)
    {
        case WindowMethod::Hamming :
            for(int i=0;i<m_FrameSize;i++)
                m_WindowCoefs.push_back(0.54-0.46*(cos(PI2*(double)i/(m_FrameSize))));
            break;

        case WindowMethod::Hann :
            for(int i=0;i<m_FrameSize;i++)
                m_WindowCoefs.push_back(0.5-0.5*(cos(PI2*(double)i/(m_FrameSize-1))));
            break;

        case WindowMethod::Blackman :
            for(int i=0;i<m_FrameSize;i++)
                m_WindowCoefs.push_back(0.42-0.5*(cos(PI2*(double(i)/(m_FrameSize-1))))+0.08*(cos(PI4*(double(i)/(m_FrameSize-1)))));
            break;

        case WindowMethod::None :
            for(int i=0;i<m_FrameSize;i++)
                m_WindowCoefs.push_back(1);
            break;
    }
}

double Mfcc::freq2mel(double freq)
{
	return 1125 * log10(1 + freq / 700);
}

double Mfcc::mel2freq(double mel)
{
	return (pow(10, mel / 1125) - 1) * 700;
}

void Mfcc::setFilterBank()
{
    //http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
	double maxMelf, deltaMelf;
	double lowFreq, mediumFreq, highFreq, currentFreq;
	int filterSize = m_FrameSize/2+1;

	maxMelf = freq2mel(m_Frequence/4);
	deltaMelf = maxMelf / (m_FilterNumber + 1);

	m_FilterBank.resize(m_FilterNumber);
    lowFreq = mel2freq(0);
    mediumFreq = mel2freq(deltaMelf);
	for(int i = 0; i < m_FilterNumber; i++)
    {
		highFreq = mel2freq(deltaMelf*(i+2));

		for(int j = 0; j < filterSize; j++)
		{
			currentFreq = (j*1.0 / (filterSize - 1) * (m_Frequence / 4));

			if((currentFreq >= lowFreq)&&(currentFreq <= mediumFreq))
				m_FilterBank[i].push_back(2*(currentFreq - lowFreq) / (mediumFreq - lowFreq));
			else if((currentFreq >= mediumFreq)&&(currentFreq <= highFreq))
				m_FilterBank[i].push_back(2*(highFreq - currentFreq) / (highFreq - mediumFreq));
			else
				m_FilterBank[i].push_back(0);
		}

		lowFreq = mediumFreq;
		mediumFreq = highFreq;
	}
}

void Mfcc::setDCTCoeff()
{
    m_DCTCoeff.resize(m_MFCCDim);
	for(int i = 0; i < m_MFCCDim; i++)
		for(int j = 0; j < m_FilterNumber; j++)
			m_DCTCoeff[i].push_back(2*cos((PI*(i+1)*(2*j + 1)) / (2 * m_FilterNumber)));
}

void Mfcc::setLiftCoeff()
{
	for(int i = 0; i < m_MFCCDim; i++)
        m_CepLifter.push_back((1.0+0.5*m_MFCCDim*sin(PI*(i+1)/(m_MFCCDim)))/((double)1.0+0.5*m_MFCCDim));
}
