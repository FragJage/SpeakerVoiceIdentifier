/// Gaussian mixture model Class
/// Un mod�le de m�lange gaussien (usuellement abr�g� par l'acronyme anglais GMM pour Gaussian Mixture Model)
/// est un mod�le statistique exprim� selon une densit� m�lange. Il sert usuellement � estimer param�triquement la
/// distribution de variables al�atoires en les mod�lisant comme une somme de plusieurs gaussiennes (appel�es noyaux).
/// Il s'agit alors de d�terminer la variance, la moyenne et l'amplitude de chaque gaussienne. Ces param�tres sont
/// optimis�s selon un crit�re de maximum de vraisemblance pour approcher le plus possible la distribution recherch�e.
/// Cette proc�dure se fait le plus souvent it�rativement via l'algorithme esp�rance-maximisation (EM).
#ifndef GMM_H
#define GMM_H

#include<string>
#include<vector>
#include<map>

struct GmmModel
{
    std::vector<double> MixCoeff;
    std::vector<std::vector<double>> Mean;
    std::vector<std::vector<double>> CovMatrix;
    std::vector<std::vector<double>> CovInv;
    std::vector<double> ExpCoeff;
};

class Gmm
{
    public:
        Gmm();
        virtual ~Gmm();

        int Trainning(const std::vector<std::vector<double>>& melCepData, size_t frameCount);
        std::string Recogniser(const std::vector<std::vector<double>>& melCepData, size_t frameCount);
        double Probability(const std::vector<std::vector<double>>& melCepData, size_t frameCount);
        bool LoadModel(const std::string& filePath);
        bool SaveModel(const std::string& filePath);
        bool AddModel(const std::string& name);
        bool AddModel(const std::string& filePath, const std::string& name);

    private:
        GmmModel newModel();
        void delModel(GmmModel model);
        void completeModel(GmmModel& model);
        double internalProbability(const std::vector<std::vector<double>>& melCepData, size_t frameCount, GmmModel model, std::vector<std::vector<double>>& normProb, std::vector<double>& mixedProb);

        int m_MixDim;
        int m_MfccDim;
        double m_Threshold;
        double m_MinCov;
        GmmModel m_Model;
        std::map<std::string, GmmModel> m_Models;

        static const double PI2;
};
#endif //GMM_H
