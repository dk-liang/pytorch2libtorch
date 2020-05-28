# pytorch2libtorch
 A demo for using C++ to inference the pytorch model

## Introduction
Althought pytorch is efficient and convenient for developer, which still having many hinderer to deploy on many devices. Here, we demonstrate how to use c++ with libtorch to inference pytorch model.

## Environment
python >= 3.6 <br />
pytorch >= 1.3 <br />
OpenCV >= 4.0.0 (It means C++ version instead of python.)

## Pytorch-python


* ```cd ./pytorch_code```  and put the pretrain model into ```./model```
* Run ```python pth2lib.py``` to generate traced script model.  The generated model can be found in ```./model```,  which is named 'model_transfer.pt'.

## LibTorch-C++
Now, we use C++ to inference the transfered model.
* ``` cp ./pytorch_code/model/model_transfer.pt ./C++_inference/model/```

* ```cd ./C++_inference```

* Download LibTorch from [here](https://pytorch.org/).  

* Unzip the libTorch and rename as  ``` libtorch```

* ``` mv libtorch ./lib/```

* Mkdir build directory. ```mkdir build```

* Compile the project.  ``` cd ./build ```  , run ```cmake..``` and  ``` make ```

* Test. ```./main  ./data/test.jpg```

  



.

