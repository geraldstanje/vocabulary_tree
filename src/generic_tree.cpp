#include <vocabulary_tree/generic_tree.h>
#include <stdexcept>
#include <boost/format.hpp>

namespace vt {

GenericTree::GenericTree()
{
}

GenericTree::GenericTree(const std::string& file)
{
  load(file);
}

/// @todo Currently assuming float. Really need more info in the save format.
// Kinda sucks that we need to wrap these functions.
void GenericTree::save(const std::string& file) const
{
  assert( initialized() );

  int toto=0,ind=0;
  std::vector<uint8_t>::const_iterator rit = valid_centers_.begin();
  while(rit != valid_centers_.end())
  {
      rit++;
      toto++;
      if(*rit == 1) ind = toto;
  }

  std::cout << centers_[0] << std::endl;
  std::cout << centers_[2] << std::endl;
  std::cout << centers_[4] << std::endl;
  std::cout << centers_[6] << std::endl;
  std::cout << centers_[8] << std::endl;
  std::cout << centers_[9] << std::endl;
  std::cout << ind << std::endl;
  std::cout << centers_[ind] << std::endl;

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

void GenericTree::load(const std::string& file)
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

    // Use arithmetic on file size to get the descriptor length, ugh. Assuming float.
//    in.seekg(0, std::ios::end);
//    int length = in.tellg();
//    int dimension = ((length - 12)/size - sizeof(uint8_t)) / sizeof(float);
//    in.seekg(12, std::ios::beg);

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
    assert(in.tellg() == length);
  }
  catch (std::ifstream::failure& e) {
    throw std::runtime_error( (boost::format("Failed to load vocabulary tree file '%s'") % file).str() );
  }

  int toto=0,ind=0;
  std::vector<uint8_t>::const_iterator rit = valid_centers_.begin();
  while(rit != valid_centers_.end())
  {
      rit++;
      toto++;
      if(*rit == 1) ind = toto;
  }

  std::cout << centers_[0] << std::endl;
  std::cout << centers_[2] << std::endl;
  std::cout << centers_[4] << std::endl;
  std::cout << centers_[6] << std::endl;
  std::cout << centers_[8] << std::endl;
  std::cout << centers_[9] << std::endl;
  std::cout << ind << std::endl;
  std::cout << centers_[ind] << std::endl;

  setNodeCounts();
  assert(size == num_words_ + word_start_);
}

size_t GenericTree::dimension() const
{
  assert( initialized() );
  return centers_[0].cols;
}



GenericSimpleKmeans::squared_distance_type GenericSimpleKmeans::clusterPointers(const std::vector<Feature>& features, size_t k,
                                      std::vector<Feature>& centers,
                                      std::vector<unsigned int>& membership) const
{

//      std::cout<< "CLUSTERPOINTS" << std::endl;
    std::vector<Feature> new_centers;

    for(size_t t=0;t<k;t++)
    {
        new_centers.push_back(zero_.clone());
    }

//    BOOST_FOREACH(Feature f, new_centers)
//            new_centers.push_back(f.clone());
//    new_centers.resize(k);

    std::vector<unsigned int> new_membership(features.size(),k+2);

    squared_distance_type least_sse = std::numeric_limits<squared_distance_type>::max();
    for (size_t starts = 0; starts < restarts_; ++starts) {
      choose_centers_(features, k, new_centers);

//        std::cout << "Chosen centers : " << std::endl;
//        BOOST_FOREACH(Feature f, new_centers)
//                std::cout << f << std::endl<< std::endl;

      squared_distance_type sse = clusterOnce(features, k, new_centers, new_membership);

      if (sse < least_sse) {
        least_sse = sse;
        BOOST_FOREACH(Feature f, new_centers)
        {
            centers.push_back(f.clone());
//              std::cout << "f : " << f << std::endl;
        }

//          for(uint i=0;i<new_centers.size();i++)
//          {
//              std::cout << "centers[i] : " << centers[i] << std::endl;
//              std::cout << "new_centers[i] : " << new_centers[i] << std::endl;
//              centers[i] = new_centers[i].clone();
//          }
//        centers = new_centers;

        membership = new_membership;
      }
    }

    return least_sse;
}

GenericSimpleKmeans::squared_distance_type GenericSimpleKmeans::clusterOnce(const std::vector<Feature>& features, size_t k,
                                  std::vector<Feature>& centers,
                                  std::vector<unsigned int>& membership) const
{

    std::vector<size_t> new_center_counts(k);
    std::vector<Feature> new_centers;


//      std::cout<< "CLUSTERONCE" << std::endl;

    for (size_t iter = 0; iter < max_iterations_; ++iter)
    {
      // Zero out new centers and counts
      std::fill(new_center_counts.begin(), new_center_counts.end(), 0);
      //std::fill(new_centers.begin(), new_centers.end(), zero_.clone());
      //std::vector<size_t> new_center_counts(k,0);
      //std::vector<Feature> new_centers(k,zero_);
      bool is_stable = true;

      for(size_t t=0;t<k;t++)
      {
          new_center_counts.push_back(0);
          new_centers.push_back(zero_.clone());
      }

//        std::cout<< "clusteronceLOOP : "<< iter << std::endl;

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

        //std::cout << nearest << " ";

        // Assign feature i to the cluster it is nearest to
        if (membership[i] != nearest) {
          is_stable = false;
          membership[i] = nearest;
        }

        // Accumulate the cluster center and its membership count
        //try{

        //deep copy ?? :s

        new_centers[nearest] += features[i];

        //}catch(...){std::cout << i << std::endl;}

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

    // Return the sum squared error
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

} //namespace vt
