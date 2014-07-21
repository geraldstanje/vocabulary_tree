#ifndef VOCABULARY_TREE_GENERIC_TREE_H
#define VOCABULARY_TREE_GENERIC_TREE_H

#include <opencv2/core/core.hpp>

#include "vocabulary_tree/vocabulary_tree.h"
#include "vocabulary_tree/simple_kmeans.h"
#include "vocabulary_tree/tree_builder.h"

namespace vt {

namespace distance {
/**
 * \brief L2 distance specialization for cv::Mat
 *
 * Note: Currently returns the real distance, not squared! So this won't work well with kmeans.
 */
template<> struct L2<cv::Mat>
{
  typedef double result_type;

  result_type operator()(const cv::Mat& a, const cv::Mat& b) const
  {
      return (result_type)cv::norm(a, b);
  }
};

/**
 * \brief L1 distance specialization for cv::Mat
 *
 *
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
class GenericTree : public VocabularyTree<cv::Mat>
{
protected:
    typedef cv::Mat Feature;
    typedef distance::L2<Feature> Distance;

public:
  /// Constructor, empty tree.
  GenericTree();
  /// Constructor, loads vocabulary from file.
  GenericTree(const std::string& file);

  GenericTree(Distance d)
      : VocabularyTree<Feature>(d)
  {
  }

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




class GenericSimpleKmeans : public SimpleKmeans<cv::Mat>
{

public:

    typedef cv::Mat Feature;
    typedef distance::L2<Feature> Distance;
    //typedef cv::Allocator<cv::Mat> FeatureAllocator;
    typedef typename Distance::result_type squared_distance_type;
    typedef boost::function<void(const std::vector<Feature>&, size_t, std::vector<Feature>&)> GenericInitializer;

  /**
   * \brief Constructor
   *
   * \param zero Object representing zero in the feature space
   * \param d    Functor for calculating squared distance
   */
  GenericSimpleKmeans(const Feature& zero = Feature(), Distance d = Distance())
      : SimpleKmeans<Feature>(zero, d),
        choose_centers_(InitGenericRandom())
  {}


  squared_distance_type clusterPointers(const std::vector<Feature>& features, size_t k,
                                          std::vector<Feature>& centers,
                                          std::vector<unsigned int>& membership) const;

private:

  squared_distance_type clusterOnce(const std::vector<Feature>& features, size_t k,
                                      std::vector<Feature>& centers,
                                      std::vector<unsigned int>& membership) const;


  //  squared_distance_type clusterOnce(const std::vector<Feature>& features, size_t k,
//                                    std::vector<Feature>& centers,
//                                    std::vector<unsigned int>& membership) const
//  {

//      std::vector<size_t> new_center_counts(k);
//      std::vector<Feature> new_centers;


////      std::cout<< "CLUSTERONCE" << std::endl;

//      for (size_t iter = 0; iter < max_iterations_; ++iter)
//      {
//        // Zero out new centers and counts
//        std::fill(new_center_counts.begin(), new_center_counts.end(), 0);
//        //std::fill(new_centers.begin(), new_centers.end(), zero_.clone());
//        //std::vector<size_t> new_center_counts(k,0);
//        //std::vector<Feature> new_centers(k,zero_);
//        bool is_stable = true;

//        for(size_t t=0;t<k;t++)
//        {
//            new_center_counts.push_back(0);
//            new_centers.push_back(zero_.clone());
//        }

////        std::cout<< "clusteronceLOOP : "<< iter << std::endl;

//        // Assign data objects to current centers
//        for (size_t i = 0; i < features.size(); ++i)
//        {
//          squared_distance_type d_min = std::numeric_limits<squared_distance_type>::max();
//          unsigned int nearest = -1;
//          // Find the nearest cluster center to feature i
//          for (unsigned int j = 0; j < k; ++j)
//          {
//            squared_distance_type distance = distance_(features[i], centers[j]);
//            if (distance < d_min) {
//              d_min = distance;
//              nearest = j;
//            }
//          }

//          //std::cout << nearest << " ";

//          // Assign feature i to the cluster it is nearest to
//          if (membership[i] != nearest) {
//            is_stable = false;
//            membership[i] = nearest;
//          }

//          // Accumulate the cluster center and its membership count
//          //try{

//          //deep copy ?? :s

//          new_centers[nearest] += features[i];

//          //}catch(...){std::cout << i << std::endl;}

//          ++new_center_counts[nearest];
//        }
//        if(is_stable) break;

//        // Assign new centers
//        for (size_t i = 0; i < k; ++i) {
//          if (new_center_counts[i] > 0) {
//              new_centers[i] /= new_center_counts[i];
//              centers[i] = new_centers[i].clone();
//          }
//          else {
//            // Choose a new center randomly from the input features
//            unsigned int index = rand() % features.size();
//            centers[i] = features[index].clone();
//          }
//        }
//      }

//  //    // Return the sum squared error
//  //    /// @todo Kahan summation?
//  //    squared_distance_type sse = squared_distance_type();
//  //    for (size_t i = 0; i < features.size(); ++i) {
//  //      //std::cout << centers[membership[i]] << std::endl;
//  //      sse += distance_(*features[i], centers[membership[i]]);
//  //    }
//  //    return sse;

//      // Return the sum squared error
//      // Kahan summation
//      squared_distance_type sse = 0.;
//      squared_distance_type c = 0.;

//      for (size_t i = 0; i < features.size(); ++i) {

//          squared_distance_type y = distance_(features[i], centers[membership[i]]) - c;
//          squared_distance_type t = sse + y;

//          c = (t - sse) - y;
//          sse = t;
//      }
//      return sse;
//  }

  GenericInitializer choose_centers_;

  struct InitGenericRandom
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
class GenericTreeBuilder : public TreeBuilder<cv::Mat>
{
public:
  typedef cv::Mat Feature;
  typedef std::vector<cv::Mat> FeatureVector;
  typedef GenericTree Tree;
  typedef distance::L2<Feature> Distance;
  typedef GenericSimpleKmeans Kmeans;

  /**
   * \brief Constructor
   *
   * \param zero Object representing zero in the feature space
   * \param d    Functor for calculating squared distance
   */
  GenericTreeBuilder(const Feature& zero = Feature(), Distance d = Distance())
      : tree_(d),
        kmeans_(zero, d),
        zero_(zero)
  {}

  void build(const FeatureVector& training_features, uint32_t k, uint32_t levels)
  {
    // Initial setup and memory allocation for the tree
    tree_.clear();
    tree_.setSize(levels, k);
    tree_.centers().reserve(tree_.nodes());
    tree_.validCenters().reserve(tree_.nodes());

    //std::cout<< "begin : " << tree_.words() << std::endl;

    // We keep a queue of disjoint feature subsets to cluster.
    // Feature* is used to avoid copying features.
    std::deque< std::vector<Feature> > subset_queue(1);

    // At first the queue contains one "subset" containing all the features.
    std::vector<Feature> &feature_ptrs = subset_queue.front();
    feature_ptrs.reserve(training_features.size());
    BOOST_FOREACH(const Feature& f, training_features)
      feature_ptrs.push_back(f); //OpenCV do not copy - shared pointer

    for (uint32_t level = 0; level < levels; ++level) {
//      printf("# Level %u\n", level);
      std::vector<unsigned int> membership;

      //std::cout<< "loop" << level << " : " << tree_.words() << std::endl;

      for (size_t i = 0, ie = subset_queue.size(); i < ie; ++i) {

        FeatureVector centers; // always size k

        std::vector<Feature> &subset = subset_queue.front();
//        printf("#\tClustering subset of size %u\n", subset.size());

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
        else { //printf("more than k , subset size %d\n", subset.size());
          // Cluster the current subset into k centers.
          kmeans_.clusterPointers(subset, k, centers, membership);
          // Add the centers and mark them as valid.

//          std::cout << "CENTERS SIZE : " << centers.size() << std::endl;

          for(uint i=0;i<centers.size();i++)
          {
              tree_.centers().push_back(centers[i].clone());
          }

  //        tree_.centers().insert(tree_.centers().end(), centers.begin(), centers.end());
          tree_.validCenters().insert(tree_.validCenters().end(), k, 1);

          // Partition the current subset into k new subsets based on the cluster assignments.
          std::vector< std::vector<Feature> > new_subsets(k);
          for (size_t j = 0; j < subset.size(); ++j) {
             // std::cout << "J : "<< j << " MEMBERSHIP : "<< membership[j] << std::endl;
            new_subsets[ membership[j] ].push_back(subset[j]);
          }

          // Update the queue
          subset_queue.pop_front();
          subset_queue.insert(subset_queue.end(), new_subsets.begin(), new_subsets.end());
        }
      }
//      printf("# centers so far = %u\n", tree_.centers().size());
    }
  }

  /// Get the built vocabulary tree.
  const Tree& tree() const { return tree_; }

protected:
  Tree tree_;
  Kmeans kmeans_;
  Feature zero_;
};


//struct InitGenericRandom
//{
//  typedef GenericTreeBuilder::Distance Distance;
//  typedef GenericTreeBuilder::Feature Feature;

//  void operator()(const std::vector<Feature>& features, size_t k, std::vector<Feature>& centers)
//  {
//    // Construct a random permutation of the features using a Fisher-Yates shuffle
//    std::vector<Feature> features_perm = features;
//    for (size_t i = features.size(); i > 1; --i) {
//      size_t k = rand() % i;
//      std::swap(features_perm[i-1], features_perm[k]);
//    }
//    // Take the first k permuted features as the initial centers
//    for (size_t i = 0; i < centers.size(); ++i)
//      centers[i] = features_perm[i].clone();
//  }
//};


} //namespace vt

#endif
