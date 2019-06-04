pip3 install -U pip
pip3 install scikit-learn==0.17
pip3 install Cython

cd DYNETDIR/dynet
mkdir build
cd build
pip3 install -U pip
pip3 install Cython
cmake .. -DEIGEN3_INCLUDE_DIR=/app/home/DYNETDIR/eigen -DPYTHON='/usr/bin/python3' -DBACKEND=cuda
make -j 40
cd python
python3 ../../setup.py build --build-dir=.. --skip-build install
export DYNET=/app/home/dynet/build/python
export PYTHONPATH=$DYNET:$PYTHONPATH

