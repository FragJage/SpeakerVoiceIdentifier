#include <algorithm> //For max_element
#include <iomanip>  //For setprecision
#include <fstream>
#include <math.h>
#include "gmm.h"

using namespace std;

const double Gmm::PI2 = 6.28318530717958647692;

Gmm::Gmm()
{
    m_MixDim = 16;
    m_MfccDim= 12;

    m_Threshold = 0.001;
    m_MinCov = 0.01;

    m_Model = newModel();
}

Gmm::~Gmm()
{
    delModel(m_Model);

    map<string, GmmModel>::const_iterator it = m_Models.begin();
    while(it != m_Models.end())
    {
        delModel(it->second);
        ++it;
    }
    m_Models.clear();
}

int Gmm::Trainning(const vector<vector<double>>& melCepData, size_t frameCount)
{
	int step;
	int loop=0;
	double newProb;
	double recProb=0.0;

    vector<double> sumProb(m_MixDim);
    vector<double> mixedProb(frameCount);
    vector<vector<double>> normProb(frameCount, vector<double>(m_MixDim));
    vector<vector<double>> tempProb(m_MfccDim, vector<double>(m_MixDim));
    vector<vector<double>> squareCep(frameCount, vector<double>(m_MfccDim));


	//*** Initialization
	for(int i=0; i<m_MixDim; i++)
		m_Model.MixCoeff[i]=1.0/m_MixDim;

    step=(int)floor(frameCount/m_MixDim);
	for(int j=0; j<m_MixDim; j++)
	    for(int i=0; i<m_MfccDim; i++)
	     	m_Model.Mean[i][j]=melCepData[step*(j+1)-1][i];

	for(int i=0; i<m_MixDim; i++)
		for(int j=0; j<m_MfccDim; j++)
			m_Model.CovMatrix[i][j]=1.0;

    for(size_t i=0; i<frameCount; i++)
        for(int j=0; j<m_MfccDim; j++)
            squareCep[i][j]=melCepData[i][j]*melCepData[i][j];


    //*** Iterative processing
	while(1)
	{
	    completeModel(m_Model);
	    newProb = internalProbability(melCepData, frameCount, m_Model, normProb, mixedProb);

	    if( ((newProb-recProb)<(m_Threshold*frameCount)) && (loop>0) )
			break;

        //*** E process, a probability matrix, n*m_MixDim
	   	for(size_t i=0; i<frameCount; i++)
		    for(int j=0; j<m_MixDim; j++)
			   	normProb[i][j]*=m_Model.MixCoeff[j]/mixedProb[i];

        //*** M process
	    for(int i=0; i<m_MixDim; i++)
		{
		   	sumProb[i]=0.0;
		   	for(size_t j=0; j<frameCount; j++)
			   	sumProb[i]+=normProb[j][i];
		}

        //*** renew mixcoef
	   	for(int i=0; i<m_MixDim; i++)
		   	m_Model.MixCoeff[i]=sumProb[i]/frameCount;

        //*** renew mean
		for(int i=0; i<m_MfccDim; i++)
			for(int j=0; j<m_MixDim; j++)
			{
				tempProb[i][j]=0.0;
				for(size_t k=0; k<frameCount; k++)
			    	tempProb[i][j]+=melCepData[k][i]*normProb[k][j];

				m_Model.Mean[i][j]=tempProb[i][j]/sumProb[j];
			}

        //*** renew covmatrix
	   	for(int i=0; i<m_MixDim; i++)
		   	for(int j=0; j<m_MfccDim; j++)
				{
			    	tempProb[j][i]=0.0;
			    	for(size_t k=0; k<frameCount; k++)
				    	tempProb[j][i]+=squareCep[k][j]*normProb[k][i];
					tempProb[j][i]=tempProb[j][i]/sumProb[i];
				}

	   	for(int i=0; i<m_MixDim; i++)
	     	for(int j=0; j<m_MfccDim; j++)
			{
		   		m_Model.CovMatrix[i][j]=tempProb[j][i]-m_Model.Mean[j][i]*m_Model.Mean[j][i];
			   	if( m_Model.CovMatrix[i][j]<=m_MinCov )
			       	m_Model.CovMatrix[i][j]=m_MinCov;
			}

        //*** prepare for next iteration
	  	recProb = newProb;
        loop++;
        if(loop > 200)  break;
	}

	tempProb.clear();
	normProb.clear();
	sumProb.clear();
	mixedProb.clear();

    return loop;
}

string Gmm::Recogniser(const vector<vector<double>>& melCepData, size_t frameCount)
{
    double prob;
    double probMax = 0;
    string name;
    bool first = true;

    vector<double> mixedProb;
    vector<vector<double>> normProb;

    map<string, GmmModel>::const_iterator it = m_Models.begin();
    map<string, GmmModel>::const_iterator itEnd = m_Models.end();

    mixedProb.resize(frameCount);
    normProb.resize(frameCount);
    for(size_t i=0; i<frameCount; i++)
        normProb[i].resize(m_MixDim);


    while(it != itEnd)
    {
        prob=internalProbability(melCepData, frameCount, it->second, normProb, mixedProb);

        if((first==true)||(probMax<=prob))
        {
            probMax = prob;
            name = it->first;
            first = false;
        }
        ++it;
    }

    normProb.clear();
	mixedProb.clear();

    return name;
}

double Gmm::Probability(const vector<vector<double>>& melCepData, size_t frameCount)
{
    double prob;
    vector<double> mixedProb;
    vector<vector<double>> normProb;


    mixedProb.resize(frameCount);
    normProb.resize(frameCount);
    for(size_t i=0; i<frameCount; i++)
        normProb[i].resize(m_MixDim);

    prob = internalProbability(melCepData, frameCount, m_Model, normProb, mixedProb);

    normProb.clear();
	mixedProb.clear();

    return prob;
}

bool Gmm::SaveModel(const string& filePath)
{
    ofstream outFile(filePath);
    if(!outFile.is_open())
        return false;

    outFile << fixed << std::setprecision(6);

    outFile << "mixcoef:" << endl;
    for(int i=0; i<m_MixDim; i++)
        outFile << m_Model.MixCoeff[i] << " ";

    outFile << endl << "mean:" << endl;
    for(int i=0; i<m_MfccDim; i++)
    {
        for(int j=0; j<m_MixDim; j++)
            outFile << m_Model.Mean[i][j] << " ";
        outFile << endl;
    }

    outFile << "covmatrix:" << endl;
    for(int i=0; i<m_MixDim; i++)
    {
        for(int j=0; j<m_MfccDim; j++)
            outFile << m_Model.CovMatrix[i][j] << " ";
        outFile << endl;
    }

    outFile.close();
    return true;
}

bool Gmm::LoadModel(const string& filePath)
{
    string title;

    ifstream inFile(filePath, ifstream::in);
    if(!inFile.is_open())
        return false;


    inFile >> title;
    for(int i=0; i<m_MixDim; i++)
        inFile >> m_Model.MixCoeff[i];

    inFile >> title;
    for(int i=0; i<m_MfccDim; i++)
    {
        for(int j=0; j<m_MixDim; j++)
            inFile >> m_Model.Mean[i][j];
    }

    inFile >> title;
    for(int i=0; i<m_MixDim; i++)
    {
        for(int j=0; j<m_MfccDim; j++)
            inFile >> m_Model.CovMatrix[i][j];
    }

    inFile.close();
    completeModel(m_Model);
    return true;
}

bool Gmm::AddModel(const string& person)
{
    GmmModel model;


    model = newModel();

    for(int i=0; i<m_MixDim; i++)
    {
        model.MixCoeff[i]=m_Model.MixCoeff[i];
        model.ExpCoeff[i]=m_Model.ExpCoeff[i];
    }

    for(int i=0; i<m_MfccDim; i++)
    {
        for(int j=0; j<m_MixDim; j++)
        {
            model.Mean[i][j] = m_Model.Mean[i][j];
            model.CovMatrix[j][i] = m_Model.CovMatrix[j][i];
            model.CovInv[j][i] = m_Model.CovInv[j][i];
        }
    }

    m_Models[person] = model;

    return true;
}

bool Gmm::AddModel(const string& filePath, const string& person)
{
    if(!LoadModel(filePath))
        return false;

    AddModel(person);

    return true;
}

double Gmm::internalProbability(const vector<vector<double>>& melCepData, size_t frameCount, GmmModel model, vector<vector<double>>& normProb, vector<double>& mixedProb)
{
	double prob=0.0;
    vector<double> maxMatrix(frameCount);
    vector<vector<double>> expMatrix(frameCount, vector<double>(m_MixDim, 0));


    for(size_t i=0; i<frameCount; i++ )
    {
        for(int j=0; j<m_MixDim; j++)
        {
            for(int k=0; k<m_MfccDim; k++)
                expMatrix[i][j]+=(pow((melCepData[i][k]-model.Mean[k][j]),2 ))*model.CovInv[j][k];
        }
    }

    for(size_t i=0; i<frameCount; i++)
    {
        auto it = max_element(expMatrix[i].begin(), expMatrix[i].end());
        maxMatrix[i] = *it;
    }

    for(size_t i=0; i<frameCount; i++)
    {
        mixedProb[i]=0.0;
        for(int j=0; j<m_MixDim; j++)
        {
            expMatrix[i][j]=exp(expMatrix[i][j]-maxMatrix[i]);
            normProb[i][j]=expMatrix[i][j]*model.ExpCoeff[j];
            mixedProb[i]=mixedProb[i]+normProb[i][j]*model.MixCoeff[j];
        }
        prob+=log(mixedProb[i])+maxMatrix[i];
    }

	expMatrix.clear();
	maxMatrix.clear();

    return prob;
}

GmmModel Gmm::newModel()
{
    GmmModel model;

    model.MixCoeff.resize(m_MixDim);

    model.Mean.resize(m_MfccDim);
    for(int i=0; i<m_MfccDim; i++)
        model.Mean[i].resize(m_MixDim);

    model.CovMatrix.resize(m_MixDim);
    model.CovInv.resize(m_MixDim);
    for(int i=0; i<m_MixDim; i++)
    {
        model.CovMatrix[i].resize(m_MfccDim);
        model.CovInv[i].resize(m_MfccDim);
    }

    model.ExpCoeff.resize(m_MixDim);

    return model;
}

void Gmm::delModel(GmmModel model)
{
    model.MixCoeff.clear();
    model.Mean.clear();
    model.CovMatrix.clear();
    model.CovInv.clear();
    model.ExpCoeff.clear();
}

void Gmm::completeModel(GmmModel& model)
{
    double x = pow(PI2,(-m_MfccDim/2));

    for(int i=0; i<m_MixDim; i++)
    {
        model.ExpCoeff[i] = 1.0;
        for(int j=0; j<m_MfccDim; j++)
            model.ExpCoeff[i]*=1.0/model.CovMatrix[i][j];
        model.ExpCoeff[i]=x*sqrt(model.ExpCoeff[i]);
    }

    for(int i=0; i<m_MixDim; i++)
        for(int j=0; j<m_MfccDim; j++)
            model.CovInv[i][j]=(-0.5)/model.CovMatrix[i][j];

}
