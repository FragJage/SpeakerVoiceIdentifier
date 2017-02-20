/// Mel-frequency cepstral coefficients Class
/// MFCCs are commonly used as features in speech recognition.
/// MFCCs are commonly derived as follows
///   - Take a Fast Fourier Transform of a windowed of a signal.
///   - Map the powers of the spectrum obtained above onto the mel scale
///   - Take the logs of the powers at each of the mel frequencies
///   - Take the discrete cosine transform of the list of mel log powers
///   - The MFCCs are the amplitudes of the resulting spectrum

#ifndef MFCC_H
#define MFCC_H

#include <vector>
#include <string>

class Mfcc
{
    public:
        enum WindowMethod {Hamming, Hann, Blackman, None};
        Mfcc();
        Mfcc(int freq, int size, int shift, WindowMethod method, int filterNum, int mfccDim);
        virtual ~Mfcc();

		size_t Analyse(const short int data[], size_t sizeData);
        bool Save(const std::string& filePath);
        const std::vector<std::vector<double>>& GetMFCCData();

        void StartAnalyse(size_t maxSize);
        bool AddBuffer(const short int data[], size_t sizeData);
		size_t GetFrameCount();

    private:
        void fft(std::vector<double>& data, int nn, int isign);
        void setWindowMethod(WindowMethod method);
        void setFilterBank();
        void setDCTCoeff();
        void setLiftCoeff();
        void internalAnalyse(std::vector<std::vector<double>>& postData, size_t frameCount, size_t currrentFrame);

        double freq2mel(double freq);
        double mel2freq(double mel);

        //Settings
        int m_Frequence;
        int m_FrameSize;
        int m_FrameShift;
        int m_FilterNumber;
        int m_MFCCDim;

        //Internal
        size_t m_FrameCount;
        std::vector<double> m_WindowCoefs;
        std::vector<std::vector<double>> m_FilterBank;
        std::vector<std::vector<double>> m_DCTCoeff;
        std::vector<double> m_CepLifter;
        std::vector<std::vector<double>> m_MFCCData;

		size_t m_CurrentFrame;
        std::vector<short int> m_RestData;

        //Constantes
        static const double PI;
        static const double PI2;
        static const double PI4;
};

#endif //MFCC_H
