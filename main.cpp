#include <iostream>
#include <chrono>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "mfcc.h"
#include "gmm.h"

#define FREQ 16000              //You can try 48000 to use 48000Hz wav files, but it's more slow.
#define TRAINSIZE FREQ*4        //4 secondes of voice for trainning
                                // --- you can increase this value to improve the recognition rate
#define RECOGSIZE FREQ*1        //1 seconde of voice for recognition
                                // --- you can increase this value to improve the recognition rate

using namespace std;

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds Milliseconds;


string GetPersonName(int personId)
{
    ostringstream oss;

    oss << "Person" << personId;
    return oss.str();
}

string GetFilePath(int person, int num, int mode, const string& extention)
{
    ostringstream oss;

    switch(mode)
    {
        case 0 :
            oss << "recog/F0" << person << "_" << num << "-" << FREQ << "." << extention;
            break;
        case 1 :
            oss << "train/F0" << person << "-" << FREQ << "." << extention;
            break;
        case 2 :
            oss << "model/" << person << "." << extention;
            break;
    }

    return oss.str();
}

void CheckWavHeader(char *header)
{
	int sr;

	if (header[20] != 0x1)
		cout << endl << "Input audio file has compression [" << header[20] << "] and not required PCM" << endl;

	sr = ((header[24] & 0xFF) | ((header[25] & 0xFF) << 8) | ((header[26] & 0xFF) << 16) | ((header[27] & 0xFF) << 24));
    cout << " " << (int)header[34] << " bits, " << (int)header[22] << " channels, " << sr << " Hz";
}

size_t ReadWav(const string& filePath, short int voiceData[], size_t sizeData, size_t seek)
{
    ifstream inFile(filePath, ifstream::in|ifstream::binary);
    size_t ret;

    if(!inFile.is_open())
    {
        cout << endl << "Can not open the WAV file !!" << endl;
        return -1;
    }

    char waveheader[44];
    inFile.read(waveheader, 44);
    if(seek==0) CheckWavHeader(waveheader);

    if(seek!=0) inFile.seekg (seek*sizeof(short int), ifstream::cur);

    inFile.read(reinterpret_cast<char *>(voiceData), sizeof(short int)*sizeData);
    ret = (size_t)inFile.gcount()/sizeof(short int);

    inFile.close();
    return ret;
}

int main()
{
   	Clock::time_point trainStart, trainEnd, recogScoreStart, recogScoreEnd, recogPercentStart, recogPercentEnd;
    Milliseconds ms;

	string filePath;
	size_t realSize;
    short int bigVoiceBuffer[TRAINSIZE];
    short int littleVoiceBuffer[2000];
	size_t frameCount;
    vector<vector<double>> melCepData;
    int loop;
    string name;
    int goodScoreRecog = 0;
    int goodPercentRecog = 0;
    map<string, int> recogHits;


    Mfcc mfcc(16000, 16, 8, Mfcc::Hamming, 24, 12);
    Gmm gmm;


    cout << "*** TRAINNING ***" << endl;
    trainStart = Clock::now();
    for(int personId=0; personId<=9; personId++)
    {
        //** Load wav file
        filePath = GetFilePath(personId, 0, 1, "wav");
        cout << filePath;
        realSize = ReadWav(filePath, bigVoiceBuffer, TRAINSIZE, 0);
        if(realSize<1) continue;

        //** Mfcc analyse WITH BIG BUFFER
        frameCount = mfcc.Analyse(bigVoiceBuffer,realSize);
        melCepData = mfcc.GetMFCCData();

        //** Gmm trainning
        loop = gmm.Trainning(melCepData, frameCount);
        filePath = GetFilePath(personId, 0, 2, "gmm");
        gmm.SaveModel(filePath);
        cout << " : " << loop << " trainning loops" << endl;
    }
    trainEnd = Clock::now();

    //** Reload saved models
    for(int personId=0; personId<=9; personId++)
    {
        filePath = GetFilePath(personId, 0, 2, "gmm");
        gmm.AddModel(filePath, GetPersonName(personId));
    }

    cout << endl << "*** RECOGNITION best score ***" << endl;
    recogScoreStart = Clock::now();
    for(int personId=0; personId<=9; personId++)
    {
        for(int num=1; num<=3; num++)
        {
            //** Mfcc analyse WITH BIG BUFFER
            filePath = GetFilePath(personId, num, 0, "wav");
            cout << filePath;
            realSize = ReadWav(filePath, bigVoiceBuffer, RECOGSIZE, 0);
            if(realSize<1) continue;

            frameCount = mfcc.Analyse(bigVoiceBuffer,realSize);
            melCepData = mfcc.GetMFCCData();
            name = gmm.Recogniser(melCepData, frameCount);

            if(name == GetPersonName(personId))
            {
                cout << " recognize correctly :)" << endl;
                goodScoreRecog++;
            }
            else
            {
                cout << " recognize wrong " << name << " !!!" << endl;
            }
        }
    }
    recogScoreEnd = Clock::now();
    cout << endl;

    cout << endl << "*** RECOGNITION best percentage ***" << endl;
    recogPercentStart = Clock::now();
    for(int personId=0; personId<=9; personId++)
    {
        for(int num=1; num<=3; num++)
        {
            //** Mfcc analyse WITH LITTLE BUFFER
            bool bContinue = true;
			size_t position = 0;
            mfcc.StartAnalyse(RECOGSIZE);
            filePath = GetFilePath(personId, num, 0, "wav");
            cout << filePath;
            recogHits.clear();
            do
            {
                realSize = ReadWav(filePath, littleVoiceBuffer, 2000, position);
                bContinue = mfcc.AddBuffer(littleVoiceBuffer, realSize);
                position += realSize;
                if(realSize!=2000) bContinue = false;
                if(position>8000)
                {
                    name = gmm.Recogniser(mfcc.GetMFCCData(), mfcc.GetFrameCount());
                    recogHits[name]++;
                }
            } while(bContinue);


            auto it1 = max_element(recogHits.cbegin(), recogHits.cend(), [](const pair<string, int>& p1, const pair<string, int>& p2) { return p1.second < p2.second; });
            name = it1->first;
            int score = it1->second;
            recogHits[name] = 0; //To find the second
            it1 = max_element(recogHits.cbegin(), recogHits.cend(), [](const pair<string, int>& p1, const pair<string, int>& p2) { return p1.second < p2.second; });

            cout << " " << name << " " << score*100/(score+it1->second) << "%";

            if(name == GetPersonName(personId))
            {
                cout << " correctly :)" << endl;
                goodPercentRecog++;
            }
            else
            {
                cout << " wrong !!!" << endl;
            }
        }
    }
    recogPercentEnd = Clock::now();
    cout << endl;

    cout << "Recognition by score rate " << goodScoreRecog*100/30 << "%" << endl;
    cout << "Recognition by % rate " << goodPercentRecog*100/30 << "%" << endl;
    ms = std::chrono::duration_cast<Milliseconds>(trainEnd - trainStart);
   	cout << "Training time " << ms.count() << " ms (" << ms.count()/8 << " by train)" << endl;
    ms = std::chrono::duration_cast<Milliseconds>(recogScoreEnd - recogScoreStart);
	cout << "Recognition by score time " << ms.count() << " ms (" << ms.count()/24 << " by recog)" << endl;
    ms = std::chrono::duration_cast<Milliseconds>(recogPercentEnd - recogPercentStart);
	cout << "Recognition by % time " << ms.count() << " ms (" << ms.count()/24 << " by recog)" << endl;

	//cin >> name;

    return 0;
}
