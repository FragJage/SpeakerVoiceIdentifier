#Activer le support c++11 
if(CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
else ()
  set(CMAKE_CXX_STANDARD 11)
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wshadow -Wunused-variable -Wunused-parameter -Wunused-function -Wunused -Wno-system-headers -Wno-deprecated -Woverloaded-virtual") # various warning flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s -O2") # no debug, active optimisation

#Déclaration de l'exécutable
set(EXECUTABLE_OUTPUT_PATH ./)
add_executable(
  SpeakerID
  gmm.cpp
  mfcc.cpp
  main.cpp
)