//
// Created by Travis A. Ebesu on 2019-02-08.
//

#ifndef TORCHTEST_FILENAMEDATASET_H
#define TORCHTEST_FILENAMEDATASET_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "torch/torch.h"
#include <vector>
#include <dirent.h>

bool file_exists(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

class FilenameDataset : public  torch::data::Dataset<FilenameDataset> {
public:
    explicit FilenameDataset(const std::string directory){
        DIR* dirp = opendir(directory.c_str());
        struct dirent * dp;
        while ((dp = readdir(dirp)) != NULL) {
            std::string fname = dp->d_name;
            if (hasEnding(fname, "jpg") | hasEnding(fname, "png") | hasEnding(fname, "jpeg")){
                fname = directory + "/" + dp->d_name;
                if(file_exists(fname.c_str()))
                    filenames.emplace_back(fname);
            }
        }
        closedir(dirp);
        std::cout << "Loaded Images: " << filenames.size() << std::endl;
    }

    bool hasEnding (std::string const &fullString, std::string const &ending) {
        if (fullString.length() >= ending.length()) {
            return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
        } else {
            return false;
        }
    }

    torch::optional<size_t> size() const override{
        return filenames.size();
    }

    torch::data::Example<> get(size_t index) override{
        /**
         * See: https://github.com/jainshobhit/pytorch-cpp-examples/blob/master/libtorch_inference.cpp#L39
         */
        // Second is a dummy placeholder
        using namespace cv;
        Mat image_bgr = imread(filenames[index], CV_LOAD_IMAGE_COLOR);
	       Mat image;

        cvtColor(image_bgr, image, COLOR_BGR2RGB);
	       resize(image, image, Size(224, 224));
        // The channel dimension is the last dimension in OpenCV
        torch::Tensor tensor_image = torch::from_blob(image.data, {image.rows, image.cols, 3}, at::kByte);
        tensor_image = tensor_image.to(at::kFloat) / 255.0f;  // Cast to float and normalize
	       // H W C => [0, 1] => W H C
        // Transpose the image for [channels, rows, columns] format of pytorch tensor
        tensor_image = at::transpose(tensor_image, 0, 1);
        // W H C => C H W
        tensor_image = at::transpose(tensor_image, 0, 2);

        // TODO: How to return a Single Example without second Tensor?
        return {tensor_image, torch::empty(1)};
    }

    std::vector<std::string> filenames;
};


#endif //TORCHTEST_FILENAMEDATASET_H
