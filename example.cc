// Copyright 2018 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <sys/time.h>    
#include <unistd.h>   

#include "/usr/local/include/opencv2/opencv.hpp"


#include "include/gflags.h"
#include "include/env.h"
#include "include/linux_env.h"
#include "include/file_system.h"
#include "include/mace.h"
#include "include/logging.h"
#include "include/string_util.h"
// if convert model to code.

#define MODEL_GRAPH_FORMAT_CODE
#ifdef MODEL_GRAPH_FORMAT_CODE
#include "include/mace_engine_factory.h"
#endif


#include "include/mobilenet_v2.h"

#include "ctime"

#ifdef MACE_ENABLE_OPENCL
namespace mace {
const unsigned char *LoadOpenCLBinary();
size_t OpenCLBinarySize();
const unsigned char *LoadOpenCLParameter();
size_t OpenCLParameterSize();
}  // namespace mace
#endif


namespace mace {
namespace examples {

std::vector<cv::Mat> preprocess(const std::string filePath, int data_count, const std::vector<std::string> &input_data)
{
    std::vector<unsigned char> *Data;
  
    std::vector<cv::Mat> Data_mat;
  
    for(int i=0;i<data_count;i++){
        Data_mat.push_back(cv::imread(filePath+input_data[i]+".jpg"));
    }
    
   
    return Data_mat;
}

void ParseShape(const std::string &str, std::vector<int64_t> *shape) {
  std::string tmp = str;
  while (!tmp.empty()) {
    int dim = atoi(tmp.data());
    shape->push_back(dim);
    size_t next_offset = tmp.find(",");
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
}

std::string FormatName(const std::string input) {
  std::string res = input;
  for (size_t i = 0; i < input.size(); ++i) {
    if (!isalnum(res[i])) res[i] = '_';
  }
  return res;
}

DeviceType ParseDeviceType(const std::string &device_str) {
  if (device_str.compare("CPU") == 0) {
    return DeviceType::CPU;
  } else if (device_str.compare("GPU") == 0) {
    return DeviceType::GPU;
  } else if (device_str.compare("HEXAGON") == 0) {
    return DeviceType::HEXAGON;
  } else {
    return DeviceType::CPU;
  }
}

DataFormat ParseDataFormat(const std::string &data_format_str) {
  if (data_format_str == "NHWC") {
    return DataFormat::NHWC;
  } else if (data_format_str == "NCHW") {
    return DataFormat::NCHW;
  } else {
    return DataFormat::DF_NONE;
  }
}

DEFINE_string(model_name,
              "mobilenet-v2",
              "model name in model deployment file");
DEFINE_string(input_node,
              "input",
              "input nodes, separated by comma");
DEFINE_string(input_shape,
              "10,224,224,3",
              "input shapes, separated by colon and comma");
DEFINE_string(output_node,
              "MobilenetV2/Predictions/Reshape_1",
              "output nodes, separated by comma");
DEFINE_string(output_shape,
              "10,1001",
              "output shapes, separated by colon and comma");
DEFINE_string(input_data_format,
              "NHWC",
              "input data formats, NONE|NHWC|NCHW");
DEFINE_string(output_data_format,
              "NHWC",
              "output data formats, NONE|NHWC|NCHW");
DEFINE_string(input_file,
              "000001,000006",
              "input file name | input file prefix for multiple inputs.");
DEFINE_string(output_file,
              "output1,output2",
              "output file name | output file prefix for multiple outputs");
DEFINE_string(input_dir,
              std::string(std::getenv("HOME"))+"/VOCdevkit/VOC2007/input",
              "input directory name");
DEFINE_string(output_dir,
              std::string(std::getenv("HOME"))+"/mace",
              "output directory name");
DEFINE_string(opencl_binary_file,
              "",
              "compiled opencl binary file path");
DEFINE_string(opencl_parameter_file,
              "",
              "tuned OpenCL parameter file path");
DEFINE_string(model_data_file,
              std::string(std::getenv("HOME"))+"/mace_pro/mobilenet/mobilenet_v2.data",
              "model data file name, used when model_data_format == file");
DEFINE_string(model_file,
              std::string(std::getenv("HOME"))+"/mace_pro/mobilenet/mobilenet_v2.pb",
              "model file name, used when load mace model in pb");
DEFINE_string(device, "CPU", "CPU/GPU/HEXAGON");
DEFINE_int32(round, 1, "round");
DEFINE_int32(restart_round, 1, "restart round");
DEFINE_int32(malloc_check_cycle, -1, "malloc debug check cycle, -1 to disable");
DEFINE_int32(gpu_perf_hint, 0, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(gpu_priority_hint, 0, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(omp_num_threads, 11, "num of openmp threads");
DEFINE_int32(cpu_affinity_policy, 1,
             "0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY");

bool RunModel(const std::vector<std::string> &input_names,
              const std::vector<std::vector<int64_t>> &input_shapes,
              const std::vector<DataFormat> &input_data_formats,
              const std::vector<std::string> &output_names,
              const std::vector<std::vector<int64_t>> &output_shapes,
              const std::vector<DataFormat> &output_data_formats) {
  
 // struct timeval timeStart, timeEnd; 
 // double runTime=0; 
  //gettimeofday(&timeStart, NULL );

  // load model
  DeviceType device_type = ParseDeviceType(FLAGS_device);
  // configuration
  // Detailed information please see mace.h
  MaceStatus status;
  MaceEngineConfig config(device_type);

  status = config.SetCPUThreadPolicy(
      FLAGS_omp_num_threads,
      static_cast<CPUAffinityPolicy>(FLAGS_cpu_affinity_policy));
;
  if (status != MaceStatus::MACE_SUCCESS) {
    std::cerr << "Set openmp or cpu affinity failed." << std::endl;
  }
#ifdef MACE_ENABLE_OPENCL
  std::shared_ptr<GPUContext> gpu_context;
  if (device_type == DeviceType::GPU) {
    // DO NOT USE tmp directory.
    // Please use APP's own directory and make sure the directory exists.
    const char *storage_path_ptr = getenv("MACE_INTERNAL_STORAGE_PATH");
    const std::string storage_path =
        std::string(storage_path_ptr == nullptr ?
                    "/data/local/tmp/mace_run/interior" : storage_path_ptr);
    std::vector<std::string> opencl_binary_paths = {FLAGS_opencl_binary_file};

    gpu_context = GPUContextBuilder()
        .SetStoragePath(storage_path)
        .SetOpenCLBinaryPaths(opencl_binary_paths)
        .SetOpenCLBinary(LoadOpenCLBinary(), OpenCLBinarySize())
        .SetOpenCLParameterPath(FLAGS_opencl_parameter_file)
        .SetOpenCLParameter(LoadOpenCLParameter(), OpenCLParameterSize())
        .Finalize();

    config.SetGPUContext(gpu_context);
    config.SetGPUHints(
        static_cast<GPUPerfHint>(FLAGS_gpu_perf_hint),
        static_cast<GPUPriorityHint>(FLAGS_gpu_priority_hint));
  }
#endif  // MACE_ENABLE_OPENCL

  // Create Engine
  std::shared_ptr<mace::MaceEngine> engine;
  MaceStatus create_engine_status;

  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_graph_data;
  if (FLAGS_model_file != "") {
    auto fs = GetFileSystem();
    auto status = fs->NewReadOnlyMemoryRegionFromFile(FLAGS_model_file.c_str(),
        &model_graph_data);
    if (status != MaceStatus::MACE_SUCCESS) {
      LOG(FATAL) << "Failed to read file: " << FLAGS_model_file;
    }
  }

  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_weights_data;
  if (FLAGS_model_data_file != "") {
    auto fs = GetFileSystem();
    auto status = fs->
    NewReadOnlyMemoryRegionFromFile(
        FLAGS_model_data_file.c_str(),
        &model_weights_data);
    if (status != MaceStatus::MACE_SUCCESS) {
      LOG(FATAL) << "Failed to read file: " << FLAGS_model_data_file;
    }
    MACE_CHECK(model_weights_data->length() > 0);
  }

  // Only choose one of the two type based on the `model_graph_format`
  // in model deployment file(.yml).
#ifdef MODEL_GRAPH_FORMAT_CODE
  // if model_data_format == code, just pass an empty string("")
  // to model_data_file parameter.
  std::cout<<model_weights_data->length()<<std::endl;
  std::cout<<model_weights_data->data()<<std::endl;

  create_engine_status = CreateMaceEngineFromCode(
      FLAGS_model_name,
      reinterpret_cast<const unsigned char *>(model_weights_data->data()),
      model_weights_data->length(),
      input_names,
      output_names,
      config,
      &engine);

#else
  create_engine_status = CreateMaceEngineFromProto(
      reinterpret_cast<const unsigned char *>(model_graph_data->data()),
      model_graph_data->length(),
      reinterpret_cast<const unsigned char *>(model_weights_data->data()),
      model_weights_data->length(),
      input_names,
      output_names,
      config,
      &engine);
#endif

  if (create_engine_status != MaceStatus::MACE_SUCCESS) {
    std::cerr << "Create engine error, please check the arguments first, "
              << "if correct, the device may not run the model, "
              << "please fall back to other strategy."
              << std::endl;
    exit(1);
  }
  
  const size_t input_count = input_names.size();///////////////////////
  const size_t output_count = output_names.size();////////////////////////

  std::map<std::string, mace::MaceTensor> inputs;
  std::map<std::string, mace::MaceTensor> outputs;
  std::map<std::string, int64_t> inputs_size;
  std::vector<std::string> inputs_data={"000001","000002","000003","000004","000011","000006","000013","000008","000014","000010"};
  //,"000002","000003","000004","000011","000006","000013","000008","000014","000010"
  int data_count=10;
  std::vector<cv::Mat> img=preprocess("/home/de/VOCdevkit/VOC2007/input/",data_count,inputs_data);



 /* for (size_t i = 0; i < input_count; ++i) {
    int64_t input_size =
        std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 1,
                        std::multiplies<int64_t>());
    inputs_size[input_names[i]] = input_size;
    auto buffer_in = std::shared_ptr<float>(new float[input_size],
                                            std::default_delete<float[]>());
    inputs[input_names[i]] = mace::MaceTensor(input_shapes[i], buffer_in,
        input_data_formats[i]);
  }*/

  /*for (size_t i = 0; i < output_count; ++i) {
    int64_t output_size =
        std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 1,
                        std::multiplies<int64_t>());
    auto buffer_out = std::shared_ptr<float>(new float[output_size],
                                             std::default_delete<float[]>());
    outputs[output_names[i]] = mace::MaceTensor(output_shapes[i], buffer_out,
        output_data_formats[i]);
  }
  std::cout<<"normal!"<<std::endl;*/
  

  if (!FLAGS_input_dir.empty()) {

    DIR *dir_parent;
    struct dirent *entry;
    dir_parent = opendir(FLAGS_input_dir.c_str());
    if (dir_parent) {
           
          std::string suffix = ".jpg"; //file_name.substr(prefix.size());
        
          for (size_t i = 0; i < input_count; ++i) {
            /*file_name = FLAGS_input_dir + "/" + FormatName(input_names[i])
                + suffix;
            std::ifstream in_file(file_name, std::ios::in | std::ios::binary);
            std::cout << "Read " << file_name << std::endl;
            if (in_file.is_open()) {
              in_file.read(reinterpret_cast<char *>(
                               inputs[input_names[i]].data().get()),
                           inputs_size[input_names[i]] * sizeof(float));
              in_file.close();
            } else {
              std::cerr << "Open input file failed" << std::endl;
              return -1;
            }*/
             

            int64_t input_size = std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 1,
                        std::multiplies<int64_t>());
            inputs_size[input_names[i]] = input_size;
            auto buffer_in = std::shared_ptr<float>(new float[input_size],std::default_delete<float[]>());
          for(int n=0; n < input_shapes[i][0]; ++n)
            for (int h=0; h < input_shapes[i][1]; ++h)
              for (int w=0; w < input_shapes[i][2]; ++w)
                for (int c=0; c < input_shapes[i][3]; ++c)
	                {
     	              int index_tmp = n*input_shapes[i][1]*input_shapes[i][3]*input_shapes[i][2]+h*input_shapes[i][3]*input_shapes[i][2] + w*input_shapes[i][3] + c;
                    int a= img[n].at<cv::Vec3b>(h,w)[c];
      	            buffer_in.get()[index_tmp] = a;
                  }
            inputs[input_names[i]] = mace::MaceTensor(input_shapes[i], buffer_in);
          }
          for (size_t i = 0; i < output_count; ++i) {
            int64_t output_size = std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 1,
                        std::multiplies<int64_t>());
            auto buffer_out = std::shared_ptr<float>(new float[output_size],std::default_delete<float[]>());
            outputs[output_names[i]] = mace::MaceTensor(output_shapes[i], buffer_out);
                  }

         /* if (!FLAGS_output_dir.empty()) {
            for (size_t i = 0; i < output_count; ++i) {
              std::string output_name =
                  FLAGS_output_dir + "/" + FormatName(output_names[i]) + ".md";
              std::ofstream out_file(output_name, std::ios::binary);
              if (out_file.is_open()) {
                out_file.write(
                    reinterpret_cast<char *>(
                        outputs[output_names[i]].data().get()),
                    output_size * sizeof(float));
                out_file.flush();
                out_file.close();
              } else {
                std::cerr << "Open output file failed" << std::endl;
                return -1;
              }
            }
          }*/
          struct timeval timeStart, timeEnd; 
          double runTime=0; 
          gettimeofday(&timeStart, NULL );
          std::cout<<"time starts"<<std::endl;
          engine->Run(inputs, &outputs);
          gettimeofday( &timeEnd, NULL ); 
          runTime = (timeEnd.tv_sec - timeStart.tv_sec ) + (double)(timeEnd.tv_usec -timeStart.tv_usec)/1000000; 
          std::cout << "The run time is: " <<runTime << "s" << std::endl;
          float sum=0;
          //for(int i=0;i<1001;i++){
        //  std::cout<<"the result in "<<i+1<<" is "<<outputs[output_names[0]].data().get()[i]<<std::endl;
         // sum+=outputs[output_names[0]].data().get()[i];}
         // std::cout<<sum<<std::endl;
        }
        
      }

     
  std::cout << "Finished" << std::endl;
 // gettimeofday( &timeEnd, NULL ); 
 // runTime = (timeEnd.tv_sec - timeStart.tv_sec ) + (double)(timeEnd.tv_usec -timeStart.tv_usec)/1000000; 
 // std::cout << "The run time is: " <<runTime << "s" << std::endl;
  return true;
}

int Main(int argc, char **argv) {
  std::string usage = "example run\nusage: " + std::string(argv[0])
      + " [flags]";
  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::cout << "mace version: " << MaceVersion() << std::endl;
  std::cout << "input node: " << FLAGS_input_node << std::endl;
  std::cout << "input shape: " << FLAGS_input_shape << std::endl;
  std::cout << "output node: " << FLAGS_output_node << std::endl;
  std::cout << "output shape: " << FLAGS_output_shape << std::endl;
  std::cout << "input_file: " << FLAGS_input_file << std::endl;
  std::cout << "output_file: " << FLAGS_output_file << std::endl;
  std::cout << "input_dir: " << FLAGS_input_dir << std::endl;
  std::cout << "output dir: " << FLAGS_output_dir << std::endl;
  std::cout << "model_data_file: " << FLAGS_model_data_file << std::endl;
  std::cout << "model_file: " << FLAGS_model_file << std::endl;
  std::cout << "device: " << FLAGS_device << std::endl;
  std::cout << "round: " << FLAGS_round << std::endl;
  std::cout << "restart_round: " << FLAGS_restart_round << std::endl;
  std::cout << "gpu_perf_hint: " << FLAGS_gpu_perf_hint << std::endl;
  std::cout << "gpu_priority_hint: " << FLAGS_gpu_priority_hint << std::endl;
  std::cout << "omp_num_threads: " << FLAGS_omp_num_threads << std::endl;
  std::cout << "cpu_affinity_policy: "
            << FLAGS_cpu_affinity_policy
            << std::endl;

  std::vector<std::string> input_names = Split(FLAGS_input_node, ',');
  std::vector<std::string> output_names = Split(FLAGS_output_node, ',');
  std::vector<std::string> input_shapes = Split(FLAGS_input_shape, ':');
  std::vector<std::string> output_shapes = Split(FLAGS_output_shape, ':');

  const size_t input_count = input_shapes.size();
  const size_t output_count = output_shapes.size();
  std::vector<std::vector<int64_t>> input_shape_vec(input_count);
  std::vector<std::vector<int64_t>> output_shape_vec(output_count);
  for (size_t i = 0; i < input_count; ++i) {
    ParseShape(input_shapes[i], &input_shape_vec[i]);
  }
  for (size_t i = 0; i < output_count; ++i) {
    ParseShape(output_shapes[i], &output_shape_vec[i]);
  }

  std::vector<std::string> raw_input_data_formats =
    Split(FLAGS_input_data_format, ',');
  std::vector<std::string> raw_output_data_formats =
    Split(FLAGS_output_data_format, ',');
  std::vector<DataFormat> input_data_formats(input_count);
  std::vector<DataFormat> output_data_formats(output_count);
  for (size_t i = 0; i < input_count; ++i) {
    input_data_formats[i] = ParseDataFormat(raw_input_data_formats[i]);
  }
  for (size_t i = 0; i < output_count; ++i) {
    output_data_formats[i] = ParseDataFormat(raw_output_data_formats[i]);
  }

  bool ret = false;
  for (int i = 0; i < FLAGS_restart_round; ++i) {
    std::cout << "restart round " << i << std::endl;
    ret =
        RunModel(input_names, input_shape_vec, input_data_formats,
                 output_names, output_shape_vec, output_data_formats);
  }
  if (ret) {
    return 0;
  } else {
    return -1;
  }
}

}  // namespace examples
}  // namespace mace

int main(int argc, char **argv) { mace::examples::Main(argc, argv); }
