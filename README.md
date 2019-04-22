# Manhattan Verifier with python

## Project Requirements

You are required to implement Manhattan verifier and report **false accept (impostor pass) and false reject rates** on a publicly available keystroke biometric dataset. You may use any programming language, as long as it can be compiled. 
In addition, I will ask you to demonstrate and explain your program.

***Dataset:*** The data consist of keystroke-timing information from 51 subjects (typists), each typing a password (.tie5Roanl) 400 times. (http://www.cs.cmu.edu/~keystroke/)

***Verification Task:*** For each user, (a) compute the template using mean key hold and key interval features calculated on the first N typing samples; (b) compute the genuine and im-postor scores using Manhattan distance; and (c) calculate and report false accept (impostor pass) and false reject rates at a given threshold T.

***Program Input:*** (1) N is the number of samples to be used for building the template (e.g., if N = 200, use the first 200 samples of each user to compute the average vector and the remaining 200 for testing; if N = 100, use the first 100 samples for the template and the re-maining 300 for testing); and (2) T is the verification threshold.
Program Output: Clearly display false accept (impostor pass) and false reject rates at a given threshold T.

## Environment
- Python 3.6
- Pandas
- Numpy
- Scipy
- Sklearn
