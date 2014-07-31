/** Copyright (c) 2014, Pal Robotics
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Pal Robotics nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Pal Robotics BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

@author Jeremie Deray

*/

#include <vocabulary_tree/cv_tree.h>
#include <stdexcept>
#include <boost/format.hpp>

namespace vt {

CVTree::CVTree()
{
}

CVTree::CVTree(const std::string& file)
{
  load(file);
}

/// @todo Currently assuming float. Really need more info in the save format.
// Kinda sucks that we need to wrap these functions.
void CVTree::save(const std::string& file) const
{
  assert( initialized() );

  uint32_t sizetype = centers_[0].elemSize();
  uint32_t type = centers_[0].type();
  uint32_t cols = centers_[0].cols;

  std::ofstream out(file.c_str(), std::ios_base::binary);

  out.write((char*)(&k_), sizeof(uint32_t)); //branchnig factor
  out.write((char*)(&levels_), sizeof(uint32_t)); //levels of the tree

  uint32_t size = centers_.size();
  out.write((char*)(&size), sizeof(uint32_t)); //nb of feat

  out.write((char*)(&cols), sizeof(uint32_t)); //cols of one feat

  out.write((char*)(&type), sizeof(uint32_t)); //opencv int defining type

  // This is pretty hacky! write feature by feature from data block
  BOOST_FOREACH(Feature f, centers_)
  {
      assert(f.isContinuous());
      out.write((const char*)(f.data), sizetype*cols);
  }

  out.write((const char*)(&valid_centers_[0]), valid_centers_.size());
}

void CVTree::load(const std::string& file)
{
  clear();

  std::ifstream in;
  in.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);

  uint32_t size;
  uint32_t cols;
  uint32_t type;

  try {
    in.open(file.c_str(), std::ios_base::binary);

    in.read((char*)(&k_), sizeof(uint32_t)); // Read branching factor
    in.read((char*)(&levels_), sizeof(uint32_t)); // Read tree levels
    in.read((char*)(&size), sizeof(uint32_t)); // Read nb of feature
    in.read((char*)(&cols), sizeof(uint32_t)); // Read cols
    in.read((char*)(&type), sizeof(uint32_t)); // Opencv int feature type

    // Read in centers as one big cv::Mat to preserve data locality.
    cv::Mat all(size, cols, type);
    assert(all.isContinuous());
    in.read((char*)all.data, size * cols * all.elemSize());

    // Now add cv::Mat centers that point into the big block of data.
    centers_.reserve(size);
    for (int i = 0; i < all.rows; ++i)
      centers_.push_back(all.row(i));

    // Read in valid centers as usual
    valid_centers_.resize(size);
    in.read((char*)(&valid_centers_[0]), valid_centers_.size());
    assert(in.tellg() == cols);
  }
  catch (std::ifstream::failure& e) {
    throw std::runtime_error( (boost::format("Failed to load vocabulary tree file '%s'") % file).str() );
  }

  setNodeCounts();
  assert(size == num_words_ + word_start_);
}

size_t CVTree::dimension() const
{
  assert( initialized() );
  return centers_[0].cols;
}

CVSimpleKmeans::squared_distance_type CVSimpleKmeans::clusterPointers(const std::vector<Feature>& features, size_t k,
                                      std::vector<Feature>& centers,
                                      std::vector<unsigned int>& membership) const
{

    std::vector<Feature> new_centers;

    for(size_t t=0;t<k;t++)
    {
        new_centers.push_back(zero_.clone());
    }

    std::vector<unsigned int> new_membership(features.size(),k+2);

    squared_distance_type least_sse = std::numeric_limits<squared_distance_type>::max();
    for (size_t starts = 0; starts < restarts_; ++starts) {
      choose_centers_(features, k, new_centers);

      squared_distance_type sse = clusterOnce(features, k, new_centers, new_membership);

      if (sse < least_sse) {
        least_sse = sse;
        BOOST_FOREACH(Feature f, new_centers)
        {
            centers.push_back(f.clone());
        }
        membership = new_membership;
      }
    }

    return least_sse;
}

CVSimpleKmeans::squared_distance_type CVSimpleKmeans::clusterOnce(const std::vector<Feature>& features, size_t k,
                                  std::vector<Feature>& centers,
                                  std::vector<unsigned int>& membership) const
{

    std::vector<size_t> new_center_counts(k);
    std::vector<Feature> new_centers;

    for (size_t iter = 0; iter < max_iterations_; ++iter)
    {
      // Zero out new centers and counts
      std::fill(new_center_counts.begin(), new_center_counts.end(), 0);
      bool is_stable = true;

      for(size_t t=0;t<k;t++)
      {
          new_center_counts.push_back(0);
          new_centers.push_back(zero_.clone());
      }

      // Assign data objects to current centers
      for (size_t i = 0; i < features.size(); ++i)
      {
        squared_distance_type d_min = std::numeric_limits<squared_distance_type>::max();
        unsigned int nearest = -1;
        // Find the nearest cluster center to feature i
        for (unsigned int j = 0; j < k; ++j)
        {
          squared_distance_type distance = distance_(features[i], centers[j]);
          if (distance < d_min) {
            d_min = distance;
            nearest = j;
          }
        }

        // Assign feature i to the cluster it is nearest to
        if (membership[i] != nearest) {
          is_stable = false;
          membership[i] = nearest;
        }

        // Accumulate the cluster center and its membership count
        new_centers[nearest] += features[i];

        ++new_center_counts[nearest];
      }
      if(is_stable) break;

      // Assign new centers
      for (size_t i = 0; i < k; ++i) {
        if (new_center_counts[i] > 0) {
            new_centers[i] /= new_center_counts[i];
            centers[i] = new_centers[i].clone();
        }
        else {
          // Choose a new center randomly from the input features
          unsigned int index = rand() % features.size();
          centers[i] = features[index].clone();
        }
      }
    }

    // Kahan summation
    squared_distance_type sse = 0.;
    squared_distance_type c = 0.;

    for (size_t i = 0; i < features.size(); ++i) {

        squared_distance_type y = distance_(features[i], centers[membership[i]]) - c;
        squared_distance_type t = sse + y;

        c = (t - sse) - y;
        sse = t;
    }
    return sse;
}

void CVTreeBuilder::build(const FeatureVector &training_features, uint32_t k, uint32_t levels)
{
    // Initial setup and memory allocation for the tree
    tree_.clear();
    tree_.setSize(levels, k);
    tree_.centers().reserve(tree_.nodes());
    tree_.validCenters().reserve(tree_.nodes());

    // We keep a queue of disjoint feature subsets to cluster.
    // Feature* is used to avoid copying features.
    std::deque< std::vector<Feature> > subset_queue(1);

    // At first the queue contains one "subset" containing all the features.
    std::vector<Feature> &feature_ptrs = subset_queue.front();
    feature_ptrs.reserve(training_features.size());
    BOOST_FOREACH(const Feature& f, training_features)
      feature_ptrs.push_back(f); //OpenCV do not copy - shared pointer

    for (uint32_t level = 0; level < levels; ++level) {
      std::vector<unsigned int> membership;

      for (size_t i = 0, ie = subset_queue.size(); i < ie; ++i) {

        FeatureVector centers; // always size k

        std::vector<Feature> &subset = subset_queue.front();

        // If the subset already has k or fewer elements, just use those as the centers.
        if (subset.size() <= k) { //printf("less than k \n");
          for (size_t j = 0; j < subset.size(); ++j) {
            tree_.centers().push_back(subset[j].clone());
            tree_.validCenters().push_back(1);
          }
          // Mark non-existent centers as invalid.
          tree_.centers().insert(tree_.centers().end(), k - subset.size(), zero_.clone());
          tree_.validCenters().insert(tree_.validCenters().end(), k - subset.size(), 0);

          // Push k empty subsets into the queue so all children get marked invalid.
          subset_queue.pop_front();
          subset_queue.insert(subset_queue.end(), k, std::vector<Feature>());
        }
        else {
          // Cluster the current subset into k centers.
          kmeans_.clusterPointers(subset, k, centers, membership);
          // Add the centers and mark them as valid.

          for(uint i=0;i<centers.size();i++)
          {
              tree_.centers().push_back(centers[i].clone());
          }

          tree_.validCenters().insert(tree_.validCenters().end(), k, 1);

          // Partition the current subset into k new subsets based on the cluster assignments.
          std::vector< std::vector<Feature> > new_subsets(k);
          for (size_t j = 0; j < subset.size(); ++j) {

            new_subsets[ membership[j] ].push_back(subset[j]);
          }

          // Update the queue
          subset_queue.pop_front();
          subset_queue.insert(subset_queue.end(), new_subsets.begin(), new_subsets.end());
        }
      }
    }
}

} //namespace vt
