#include <string>
#include <pcl/point_types.h>
#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/ModelCoefficients.h>
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/common/common_headers.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <map>
#include <Eigen/Dense>
#include <vector>

 pcl::visualization::PCLVisualizer visualize (pcl::visualization::PCLVisualizer viewer, 
                                                            std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr > &newClouds,
                                                            std::string cloud_initial_name,
                                                            bool OPT_RED,
                                                            bool OPT_BIG_POINTS);
using namespace pcl;
using namespace std;

// put true when you want to debug the code (pausing after each time step)
bool debug=false;


struct thresholds
{
	float densityThreshold, 
	  	  slindernessThreshold, 
	      sdXYThreshold;
};

struct parameters
{
	float 	leafSize,
	  		minClusterSize,
	  		clusterTolerance,
	  		limitsXY,
	  		limitZlow,
	  		limitZhigh;
};




// *********************  MAIN  ********************* //
int main (int argc, char *argv[])
{
// Pass parameters
int num_frames;
int initial_frame;
istringstream (argv[1]) >> initial_frame;
istringstream (argv[2]) >> num_frames;


// Set thresholds
thresholds T = {
	200,  // densityThreshold
	1.5,  // slendernessThreshold
	0.25,   // sdXYThreshold
};

// Set parameters
parameters P = {
	  0.05,  // leafSize
	  20,   // minClusterSize
	  0.6,  // clusterTolerance
	  25,   // limitsXY
	  -0.7, // limitZlow
	  1    // limitZhigh
};

// Declare variables
double area, density, meanX, meanY, meanZ, varX, varY, varZ, sdX, sdY, sdZ, slendernessX, slendernessY; 

// configuring point cloud viewer
pcl::visualization::PCLVisualizer viewer("PCL Viewer");
viewer.setBackgroundColor( 0.0, 0.0, 0.0 );

//viewer.setFullScreen(true); 
viewer.setCameraPosition(-21.1433, -23.4669, 12.7822,0.137915, -0.429331, -1.9301,0.316165, 0.28568, 0.904669);
viewer.setCameraClipDistances(0.0792402, 79.2402); 

for (int a= initial_frame; a <= num_frames; a++)
{
/*
		//-------------please comment this part if you do't want to debug the code-----------//
		// configuring point cloud viewer
		pcl::visualization::PCLVisualizer viewer("PCL Viewer");
		viewer.setBackgroundColor( 0.0, 0.0, 0.0 );

		//viewer.setFullScreen(true); 
		viewer.setCameraPosition(-21.1433, -23.4669, 12.7822,0.137915, -0.429331, -1.9301,0.316165, 0.28568, 0.904669);
		viewer.setCameraClipDistances(0.0792402, 79.2402); 
		//-----------------------------------------------------------------------------------//
*/
  	// Create vaiables
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud          (new pcl::PointCloud<pcl::PointXYZ>), 
										cloud_original (new pcl::PointCloud<pcl::PointXYZ>);

    // Read current file
	pcl::PCDReader reader;
	std::string filename2;
	std::stringstream sa;
	sa << setw(6) << setfill('0') << a;
	filename2= sa.str();
	reader.read ("../Data/pcd-files/KITTI/" + filename2 + ".pcd", *cloud_original);

	// Voxels
	pcl::VoxelGrid<pcl::PointXYZ> vg;
	vg.setInputCloud (cloud_original);
	vg.setLeafSize (P.leafSize, P.leafSize, P.leafSize);
	// vg.setMinimumPointsNumberPerVoxel(2);
	vg.filter (*cloud);

	// Eliminate far points
	pcl::PassThrough<pcl::PointXYZ> pass;

	pass.setInputCloud (cloud_original);
	pass.setFilterFieldName ("x");
	pass.setFilterLimits (-P.limitsXY, P.limitsXY);
	pass.filter (*cloud);

	pass.setInputCloud (cloud);
	pass.setFilterFieldName ("y");
	pass.setFilterLimits (-P.limitsXY, P.limitsXY);
	pass.filter (*cloud);

	pass.setInputCloud (cloud);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (P.limitZlow, P.limitZhigh);
	pass.filter (*cloud);

	// Remove outliers
/*	pcl::RadiusOutlierRemoval<pcl::PointXYZ> rorfilter (false);
	rorfilter.setInputCloud (cloud);
	rorfilter.setRadiusSearch (3*P.leafSize);
	rorfilter.setMinNeighborsInRadius (6);
	rorfilter.filter (*cloud);*/

	// Extraction of clusters
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloud);
	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (P.clusterTolerance);
	ec.setMinClusterSize (P.minClusterSize);
	ec.setSearchMethod (tree);
	ec.setInputCloud (cloud);
	ec.extract (cluster_indices);

	// Visualize all cloud - White cloud
	viewer.removeAllPointClouds();
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> white_color (cloud_original, 255, 255, 255);
	viewer.addPointCloud <pcl::PointXYZ> (cloud_original, white_color, "cloud_original");

	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters; // Vector of pointclouds pointers
	std::stringstream cluterInd;
	int j=0, numClusters= 0, i;

	// Loop over clusters
	for (std::vector<pcl::PointIndices>::const_iterator it= cluster_indices.begin (); it != cluster_indices.end (); ++it,++j)
	{
	    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);

	    // Add the points for the current cluster
	    for (std::vector<int>::const_iterator pit= it->indices.begin (); pit != it->indices.end (); pit++)
	    {
	      cloud_cluster->points.push_back (cloud->points[*pit]);
	    }
	    cloud_cluster->width = int (cloud_cluster->points.size ());
	    cloud_cluster->height = 1;
	    cloud_cluster->is_dense = true;
	    clusters.push_back(cloud_cluster);

	    // Compute parameter for the cluster
	    cluterInd << j;
	    meanX= meanY= meanZ= 0;
	    for (i=0; i < (clusters[j]->points.size()); i++) // find type of object and find inbuilt function!!
	    {
			meanX= meanX + double(clusters[j]->points[i].x);
			meanY= meanY + double(clusters[j]->points[i].y);
			meanZ= meanZ + double(clusters[j]->points[i].z);
	    }

	    meanX= meanX / double(clusters[j]->points.size());
	    meanY= meanY / double(clusters[j]->points.size());
	    meanZ= meanZ / double(clusters[j]->points.size());

	    varX= varY= varZ=0;
	    for (i=0; i < (clusters[j]->points.size()); i++)
	    {
			varX= varX + pow( double(clusters[j]->points[i].x) - meanX, 2 );
			varY= varY + pow( double(clusters[j]->points[i].y) - meanY, 2 );
			varZ= varZ + pow( double(clusters[j]->points[i].z) - meanZ, 2 );
	  	}
	    varX= varX / double( clusters[j]->points.size() - 1 ); sdX= sqrt(varX);
	    varY= varY / double( clusters[j]->points.size() - 1 ); sdY= sqrt(varY);
	    varZ= varZ / double( clusters[j]->points.size() - 1 ); sdZ= sqrt(varZ);

	    // Parameters of the cluster
	    slendernessX= sdZ / sdX;
	    slendernessY= sdZ / sdY;
	    density= double( clusters[j]->points.size() ) / (27*sdX*sdY*sdZ);

	    if ( density > T.densityThreshold  		   &&  
	    	 sdX < T.sdXYThreshold				   &&
	    	 sdY < T.sdXYThreshold				   &&
	    	 slendernessX > T.slindernessThreshold &&  
	    	 slendernessY > T.slindernessThreshold )
	    {
		    numClusters++;

		    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red_color (clusters[j], 255, 0, 0); 
		    viewer.addPointCloud <pcl::PointXYZ> (clusters[j], red_color, "clusters["+ cluterInd.str()+"]");
		    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "clusters["+ cluterInd.str()+"]");

		    //Save cluster
		    cout << "Cluster " << j << " ------> " << clusters[j]->points.size () << " points." << std::endl;
		    cout << "SD volume = " << sdX*sdY*sdZ << endl;
		    cout << "density = " << density << endl;
		    cout << "SD in X = " << sdX << endl;
		    cout << "SD in Y = " << sdY << endl;
		    cout << "SD in Z = " << sdZ << endl;
		    cout << "slendeness in X = " << slendernessX << endl;
		    cout << "slendeness in Y = " << slendernessY << endl << endl;

	    }
	} // End of loop over clusters
	cout<< "-----------------------------------"<< endl;
	cout << "# clusters = " << numClusters << endl;
	cout<< "-----------------------------------"<< endl;

	// debugging point for completeness and correctness evaluation
	if (debug==false){
		viewer.spinOnce ();
	}
	else{
		while (!viewer.wasStopped ())
    		{
    			viewer.spinOnce ();
    		}
	}
}

} // End of Main