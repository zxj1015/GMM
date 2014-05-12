rm -rf GMM
rm -rf model.txt
rm -rf dev_out.txt
rm -rf test_out.txt
g++ GMM.cpp -o GMM
./GMM -train -4 -train.txt -model.txt
./GMM -dev -model.txt -dev.txt -dev_out.txt
./GMM -test -model.txt -test.txt -test_out.txt

