This code base provides functionality for deep generative models for NLP.


# Install

We develop in `python 3.5` and `tensorflow r1.4`


* Creating a virtual environment based on python3: this you only need to do the first time.
```sh
virtualenv -p python3 ~/envs/dgm4nlp
```

* Sourcing it: this you need to do whenever you want to run (or develop) the code.
```sh
source ~/envs/dgm4nlp/bin/activate
```

* In case you use `PyCharm` you will need to configure it to use your environment:        
```sh
# navigate to
PyCharm/Preferences/Project/Project Interpreter
# point your interpreter to
~/envs/dgm4nlp/bin/python
```

* Requirements
```sh
# yolk3k is not a requirement, but it is helpful to list packages in our virtual environment
pip install yolk3k
pip install numpy
pip install scipy
pip install tabulate
pip install dill
# this may vary depending on your OS check tf docs
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0rc1-cp35-cp35m-linux_x86_64.whl
```
      
* Clone the code
```sh
git clone https://github.com/uva-slpl/dgm4nlp.git
```
        
* Build
```sh
cd dgm4nlp
python setup.py develop
```
        


