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

#ifndef VOCABULARY_TREE_CV_TREE_H
#define VOCABULARY_TREE_CV_TREE_H

#include <opencv2/core/core.hpp>

#include "vocabulary_tree/generic_tree.h"
#include "vocabulary_tree/simple_kmeans.h"
#include "vocabulary_tree/tree_builder.h"

namespace vt {

namespace distance {

///**
// * \brief L2 distance specialization for cv::Mat
// */
//template<> struct L2<cv::Mat>
//{
//  typedef double result_type;

//  result_type operator()(const cv::Mat& a, const cv::Mat& b) const
//  {
//      return (result_type)cv::norm(a, b);
//  }
//};

/**
 * \brief L1 distance specialization for cv::Mat
 */
template<> struct L1<cv::Mat>
{
  typedef double result_type;

  result_type operator()(const cv::Mat& a, const cv::Mat& b) const
  {
    return (result_type)cv::norm(a, b, cv::NORM_L1);
  }
};

} //namespace distance

/**
 * \brief Vocabulary tree wrapper for easy integration with OpenCV features, or when the
 * (dense) descriptor size and/or type isn't known at compile time.
 *
 * \c cv::Mat is used as the feature type. If the feature type is known at compile time,
 * the VocabularyTree template may be considerably more efficient.
 */
class CVTree : public VocabularyTree<cv::Mat>
{
protected:
    typedef cv::Mat Feature;
    typedef distance::L2<Feature> Distance;

public:

  // Constructor, empty tree.
  CVTree();

  // Constructor, loads vocabulary from file.
  CVTree(const std::string& file);

  CVTree(Distance d)
      : VocabularyTree<Feature>(d) {}

  void setSize(uint32_t levels, uint32_t splits)
  {
    this->levels_ = levels;
    this->k_ = splits;
    this->setNodeCounts();
  }

  uint32_t nodes() const
  {
    return this->word_start_ + this->num_words_;
  }

  std::vector<Feature>& centers() { return this->centers_; }
  const std::vector<Feature>& centers() const { return this->centers_; }

  std::vector<uint8_t>& validCenters() { return this->valid_centers_; }
  const std::vector<uint8_t>& validCenters() const { return this->valid_centers_; }

  /// Save vocabulary to a file.
  void save(const std::string& file) const;
  /// Load vocabulary from a file.
  void load(const std::string& file);

  /// Returns the number of elements in a feature used by this tree.
  size_t dimension() const;
};


class CVSimpleKmeans : public SimpleKmeans<cv::Mat>
{

public:

    typedef cv::Mat Feature;
    typedef distance::L2<Feature> Distance;
    typedef typename Distance::result_type squared_distance_type;
    typedef boost::function<void(const std::vector<Feature>&, size_t, std::vector<Feature>&)> CVInitializer;

  /**
   * \brief Constructor
   *
   * \param zero Object representing zero in the feature space
   * \param d    Functor for calculating squared distance
   */
  CVSimpleKmeans(const Feature& zero = Feature(), Distance d = Distance())
      : SimpleKmeans<Feature>(zero, d),
        choose_centers_(InitCVRandom())
  {}


  squared_distance_type clusterPointers(const std::vector<Feature>& features, size_t k,
                                          std::vector<Feature>& centers,
                                          std::vector<unsigned int>& membership) const;

private:

  squared_distance_type clusterOnce(const std::vector<Feature>& features, size_t k,
                                      std::vector<Feature>& centers,
                                      std::vector<unsigned int>& membership) const;

  CVInitializer choose_centers_;

  struct InitCVRandom
  {
    void operator()(const std::vector<Feature>& features, size_t k, std::vector<Feature>& centers)
    {
      // Construct a random permutation of the features using a Fisher-Yates shuffle
      std::vector<Feature> features_perm = features;
      for (size_t i = features.size(); i > 1; --i) {
        size_t k = rand() % i;
        std::swap(features_perm[i-1], features_perm[k]);
      }
      // Take the first k permuted features as the initial centers
      for (size_t i = 0; i < centers.size(); ++i)
        centers[i] = features_perm[i].clone();
    }
  };

};



/**
 * \brief Class for building a new vocabulary by hierarchically clustering
 * a set of training features.
 */
class CVTreeBuilder : public TreeBuilder<cv::Mat>
{
public:
  typedef cv::Mat Feature;
  typedef std::vector<cv::Mat> FeatureVector;
  typedef CVTree Tree;
  typedef distance::L2<Feature> Distance;
  typedef CVSimpleKmeans Kmeans;

  /**
   * \brief Constructor
   *
   * \param zero Object representing zero in the feature space
   * \param d    Functor for calculating squared distance
   */
  CVTreeBuilder(const Feature& zero = Feature(), Distance d = Distance())
      : tree_(d),
        kmeans_(zero, d),
        zero_(zero)
  {}

  void build(const FeatureVector& training_features, uint32_t k, uint32_t levels);

  /// Get the built vocabulary tree.
  const Tree& tree() const { return tree_; }

protected:
  Tree tree_;
  Kmeans kmeans_;
  Feature zero_;
};

} //namespace vt

#endif
