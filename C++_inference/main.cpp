#include <torch/script.h> // One-stop header.
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <time.h>
#include <cstring>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>

using namespace std;
using namespace cv;
int main(int argc, const char **argv) {


    //读取模型
    const std::string model_path= "../model/model_transfer.pt";//模型路径
    torch::jit::script::Module module = torch::jit::load(model_path);

    //定义变量
    Mat image, image_transfomed;
    double t, fps;
    int people_num;

    if (argc != 2)
    {
        cout << "请输入图片路径 " << endl;
        waitKey();
        return -1;
    }

    while(1)
    {
        t = (double)cv::getTickCount();

        //读取数据
        const std::string mat_path= argv[1];
        image = imread(mat_path);

        //把图像resize到固定大小，模型只能接收固定大小的图像，默认设定720P
        resize(image, image_transfomed,Size(1280,720), 0, 0, INTER_LINEAR);
        cvtColor(image_transfomed, image_transfomed, COLOR_BGR2RGB);

        //转化为pytorch的数据格式
        torch::Tensor tensor_image = torch::from_blob(image_transfomed.data, {image_transfomed.rows, image_transfomed.cols,3},torch::kByte).cuda();
        tensor_image = tensor_image.permute({2,0,1});
        tensor_image = tensor_image.toType(torch::kFloat);
        tensor_image = tensor_image.div(255);
        tensor_image[0] = (tensor_image[0] - 0.485) / 0.229;
        tensor_image[1] = (tensor_image[1] - 0.456) / 0.224;
        tensor_image[2] = (tensor_image[2] - 0.406) / 0.225;
        tensor_image = tensor_image.unsqueeze(0);

        //模型推理
        auto outputs = module.forward({tensor_image});
        people_num = int(outputs.toTensor().item<double>());

        //计算帧数
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        fps = 1.0 / t;

        cout<< "预测人数：" <<  people_num << "   fps: " << fps << "\n";

        waitKey(1);

    };

}
