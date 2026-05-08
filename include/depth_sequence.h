#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "types.h"

namespace fs = std::filesystem;

/**
 * Structure to hold a single frame of data
 */
struct DepthFrame
{
  cv::Mat depth_m;    // Depth map in meters (float)
  cv::Mat color_gray; // Grayscale version of the color image
  double timestamp = 0.0;
  int index = 0;
};

class DepthSequence
{
public:
  enum class Format
  {
    TUM,
    ICL,
    RAW_PNG,
    RAW_EXR
  };

  struct Config
  {
    std::string path;
    Format format = Format::TUM;
    float depth_scale = 0.001f; // Conversion factor to meters
    int start_frame = 0;
    int max_frames = -1;
    CameraIntrinsics cam;
  };

  explicit DepthSequence(const Config &cfg) : cfg_(cfg)
  {
    switch (cfg.format)
    {
    case Format::TUM:
      load_tum();
      break;
    case Format::ICL:
      load_icl();
      break;
    case Format::RAW_PNG:
    case Format::RAW_EXR:
      load_raw();
      break;
    }
  }

  bool has_next() const { return current_idx_ < (int)frames_.size(); }

  DepthFrame next()
  {
    if (!has_next())
      return {};
    return frames_[current_idx_++];
  }

  int total() const { return (int)frames_.size(); }
  const CameraIntrinsics &camera() const { return cfg_.cam; }

  /**
   * Fallback utility: creates a grayscale visualization from depth if color is missing
   */
  static void depth_to_gray(const cv::Mat &depth_m, cv::Mat &gray)
  {
    double min_v = 0.0, max_v = 0.0;
    cv::Mat valid = depth_m > 0.01f;
    cv::minMaxLoc(depth_m, &min_v, &max_v, nullptr, nullptr, valid);
    if (max_v <= min_v)
    {
      gray = cv::Mat(depth_m.size(), CV_8UC1, cv::Scalar(0));
      return;
    }
    cv::Mat norm;
    depth_m.convertTo(norm, CV_8UC1, -255.0 / (max_v - min_v),
                      255.0 * max_v / (max_v - min_v));
    norm.setTo(0, ~valid);
    gray = norm;
  }

private:
  Config cfg_;
  std::vector<DepthFrame> frames_;
  int current_idx_ = 0;

  /**
   * Logic for TUM datasets. Handles both standard associations.txt
   * and direct 'depth'/'color' folder structures.
   */
  void load_tum()
  {
    std::string assoc_path = cfg_.path + "/associations.txt";
    std::ifstream assoc(assoc_path);

    // CASE A: No association file. Load directly from subfolders.
    if (!assoc.is_open())
    {
      std::cout << "[DepthSequence] No associations.txt found. Loading from /depth and /color subfolders.\n";

      std::string depth_dir = cfg_.path + "/depth";
      std::string color_dir = cfg_.path + "/color";

      if (!fs::exists(depth_dir) || !fs::exists(color_dir))
      {
        std::cerr << "[Error] Ensure both /depth and /color folders exist.\n";
        return;
      }

      // Collect all image paths
      std::vector<std::string> d_paths, c_paths;
      for (const auto &e : fs::directory_iterator(depth_dir))
        if (e.path().extension() == ".png")
          d_paths.push_back(e.path().string());
      for (const auto &e : fs::directory_iterator(color_dir))
        if (e.path().extension() == ".png")
          c_paths.push_back(e.path().string());

      // Sort alphabetically so that frame 001.png matches in both folders
      std::sort(d_paths.begin(), d_paths.end());
      std::sort(c_paths.begin(), c_paths.end());

      size_t n = std::min(d_paths.size(), c_paths.size());
      int added_count = 0;

      for (size_t i = 0; i < n; ++i)
      {
        if (added_count < cfg_.start_frame)
        {
          added_count++;
          continue;
        }
        if (cfg_.max_frames > 0 && (int)frames_.size() >= cfg_.max_frames)
          break;

        cv::Mat raw = cv::imread(d_paths[i], cv::IMREAD_ANYDEPTH);
        cv::Mat color = cv::imread(c_paths[i], cv::IMREAD_GRAYSCALE);

        if (raw.empty() || color.empty())
          continue;

        cv::Mat depth_m;
        raw.convertTo(depth_m, CV_32F, cfg_.depth_scale);

        // Ensure color image matches depth resolution
        if (color.size() != depth_m.size())
          cv::resize(color, color, depth_m.size(), 0, 0, cv::INTER_LINEAR);

        frames_.push_back({depth_m, color, (double)i, (int)i});
      }
      return;
    }

    // CASE B: Standard TUM logic using associations.txt
    std::string line;
    int count = 0;
    while (std::getline(assoc, line))
    {
      if (line.empty() || line[0] == '#')
        continue;
      if (count < cfg_.start_frame)
      {
        count++;
        continue;
      }
      if (cfg_.max_frames > 0 && (int)frames_.size() >= cfg_.max_frames)
        break;

      std::istringstream ss(line);
      double ts_rgb, ts_depth;
      std::string rgb_file, depth_file;
      ss >> ts_rgb >> rgb_file >> ts_depth >> depth_file;

      cv::Mat raw = cv::imread(cfg_.path + "/" + depth_file, cv::IMREAD_ANYDEPTH);
      cv::Mat color = cv::imread(cfg_.path + "/" + rgb_file, cv::IMREAD_GRAYSCALE);

      if (raw.empty())
      {
        count++;
        continue;
      }

      cv::Mat depth_m;
      raw.convertTo(depth_m, CV_32F, cfg_.depth_scale);

      if (!color.empty() && (color.size() != depth_m.size()))
        cv::resize(color, color, depth_m.size(), 0, 0, cv::INTER_LINEAR);

      frames_.push_back({depth_m, color, ts_depth, count});
      count++;
    }
  }

  void load_icl()
  {
    load_raw_dir(cfg_.path + "/depth", ".png");
  }

  void load_raw()
  {
    std::string ext = (cfg_.format == Format::RAW_EXR) ? ".exr" : ".png";
    load_raw_dir(cfg_.path, ext);
  }

  /**
   * Generic loader for simple directories (generates color from depth)
   */
  void load_raw_dir(const std::string &dir, const std::string &ext)
  {
    if (!fs::exists(dir))
      return;

    std::vector<std::string> paths;
    for (const auto &entry : fs::directory_iterator(dir))
      if (entry.path().extension() == ext)
        paths.push_back(entry.path().string());

    std::sort(paths.begin(), paths.end());

    int count = 0;
    for (const auto &p : paths)
    {
      if (count < cfg_.start_frame)
      {
        count++;
        continue;
      }
      if (cfg_.max_frames > 0 && (int)frames_.size() >= cfg_.max_frames)
        break;

      cv::Mat raw = cv::imread(p, cv::IMREAD_ANYDEPTH);
      if (raw.empty())
        raw = cv::imread(p, cv::IMREAD_UNCHANGED);
      if (raw.empty())
      {
        count++;
        continue;
      }

      cv::Mat depth_m;
      if (raw.type() == CV_32F)
        depth_m = raw;
      else
        raw.convertTo(depth_m, CV_32F, cfg_.depth_scale);

      // Basic filtering
      cv::threshold(depth_m, depth_m, 0.1f, 0, cv::THRESH_TOZERO);
      cv::threshold(depth_m, depth_m, 6.0f, 0, cv::THRESH_TOZERO_INV);

      cv::Mat color;
      depth_to_gray(depth_m, color); // Create fake grayscale from depth
      frames_.push_back({depth_m, color, (double)count, count});
      count++;
    }
  }
};