SpeakerVoiceIdentifier
======================
SpeakerVoiceIdentifier recognize the voice of a speaker.

Introduction
============
SpeakerVoiceIdentifier can recognize the voice of a speaker by learning.

Features
========
 - Use MFCC (Mel-frequency cepstral coefficients) to analyse the voice
 - Use GMM classifier (Gaussian mixture model) to modelize and recognize the voice

Timing
======
Recognition of one voice betwwen ten :
 - 5 ms on Core i7 3.4Ghz
 - 10 ms on Celeron G540 2.5Ghz
 - 154 ms on Raspberry PI model A 
 
Portability
===========
 - Compatible with x86, x64 and ARM architecture
 - Compatible with windows and Linux OS.
 - So it's compatible for Raspberry Pi
 - No dependency

Builds
======
You can build SpeakerVoiceIdentifier with Code::Blocks project (SpeakerIdCpp.cbp) or VS2015 solution (SpeakerIdCpp.sln) or CMake (CMakeLists.txt).

Licence
=======
SpeakerVoiceIdentifier is free software : you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

SpeakerVoiceIdentifier is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with SpeakerVoiceIdentifier. If not, see http://www.gnu.org/licenses/.
