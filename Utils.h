#include <iostream>
#include <fstream>
#include <dirent.h>
#include <random>
#include <unistd.h>
#include <stdlib.h>
#include <iterator>

# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/nonfree/features2d.hpp"
# include "opencv2/opencv_modules.hpp"
# include <ml.h>

// boost
#include <boost/filesystem.hpp>
using namespace std;
using namespace cv;

string root_path = "/Volumes/SOLEDAD/";



void listDir(string path_, vector<vector<string> > &files, vector<string> &label_name_vector, vector<int > &label_vector, int initial_label)
{

	namespace fs = boost::filesystem;

	fs::path p(path_);
	if( fs::is_directory(p ) )
	{
		vector<string> v;
		for (auto i = fs::directory_iterator(p); i != fs::directory_iterator(); i++)
		{
			if (!is_directory(i->path()))
			{
				string file_path = i->path().string();

				if((i->path().filename().string().at(0)) != '.')
				{
					v.push_back(file_path);
					label_vector.push_back(initial_label);
				}
			}
			else
			{
				initial_label++;
				label_name_vector.push_back(i->path().filename().string());
				listDir(i->path().string(), files, label_name_vector, label_vector, initial_label);
			}
		}
		if(!v.empty())
			files.push_back(v);
	}
	else
	{
		vector<string> v;
		v.push_back(p.string());
		files.push_back(v);
		label_vector.push_back(initial_label);
		return;
	}
}


void splitDataSet(vector<vector<string> > &files, vector<string> &processed_files, vector<int> &processed_files_class, int mode)
{

	for (int i = 0; i < files.size(); ++i)
	{
		if(files.at(i).size() > 1){

			if (mode == 1 || mode == 0)
			{
				for (int jj = 0; jj < files.at(i).size()/2; ++jj)
				{
					processed_files.push_back(files.at(i).at(jj));
					processed_files_class.push_back(i+1);
				}
			}
			else
			{
				for (int ii = (files.at(i).size()/2); ii < files.at(i).size(); ++ii)
				{
					processed_files.push_back(files.at(i).at(ii));
					processed_files_class.push_back(i+1);
				}
			}
		}
		else
		{
			processed_files.push_back(files.at(i).at(0));
			processed_files_class.push_back(i+1);
		}
	}
}


void initFile(std::string &fileName){
	std::ofstream f (fileName.c_str(), std::ofstream::out);
	f.close();
}





void matToFile(const char* fileName, cv::Mat &v){

	std::ofstream output_file;

	// output_file.open (fileName, std::ios::out | std::ios::app);
	output_file.open (fileName, std::ios::out);

	if(output_file.is_open()){
		for (int j = 0; j < v.rows; ++j)
		{
			output_file << v.row(j);
			output_file << "\n";
		}

		output_file.close();
	}
	else
	{
		cout << "file could not be open" << endl;
	}
}

bool fileToMat(const char* fileName, cv::Mat &v){

	std::ifstream input_file;  
	input_file.open (fileName, std::ios::in);

	if(!input_file.is_open()){
		printf("%s file not open\n", fileName);
		return false;
	}

	std::string line;
	while (std::getline(input_file, line)) {
		std::istringstream stream(line);

		std::string x;
		std::vector<float> row;
		while (stream >> x) 
		{  
			int pos = x.find ("[", 0);
			if( pos != std::string::npos){ 

				x.erase (x.begin() + pos, x.begin() + pos + 1);
			}

			x.erase (x.end()-1, x.end());

			row.push_back(std::stof(x));
		}
		cv::Mat row_(row);
		row_ = row_.t();
		v.push_back(row_);
	}

	input_file.close();
	return true;
}




void randomMatrixSplit( cv::Mat &data, cv::Mat &mat1, cv::Mat &mat2, std::vector<int> &mat1_idx, std::vector<int> &mat2_idx, int split_size)
{

	mat1.reserve(split_size);
	mat1.reserve(data.rows - split_size);

	std::vector<int> data_idx;
	data_idx.reserve(data.rows);
	int n(0);
	std::generate_n(std::back_inserter(data_idx), data.rows, [n]()mutable { return n++; });

	std::map<int, int> random_indexes;

	// create map with random and unique values
	while(random_indexes.size() < split_size)
	{
		int rand_idx = std::rand() % data.rows;
		random_indexes[rand_idx] = rand_idx;
	}

	mat1_idx.clear();

	// fill mat1 with the data at the random indexes
	for( std::map<int, int>::iterator it = random_indexes.begin(); it != random_indexes.end(); ++it ) 
	{
		mat1_idx.push_back( int(it->second) );
		mat1.push_back( data.row(int(it->second) ));
	}

	// fill mat2 with the remaining data
	int j = 0;
	for (int i = 0; i < data_idx.size(); ++i)
	{ 
		if(j < mat1_idx.size())
		{
			if(mat1_idx.at(j) == data_idx.at(i))
			{
				j++;
			}
			else
			{
				mat2_idx.push_back(data_idx.at(i));
				mat2.push_back(data.row(i));
			}
		}
		else
		{
			mat2_idx.push_back(data_idx.at(i));
			mat2.push_back(data.row(i));
		}
	}
}


void findNearestNeighbor(cv::Mat &training_data, cv::Mat &training_labels, cv::Mat &testing_data, cv::Mat &testing_labels, int k)
{

	CvKNearest* knn = new CvKNearest();

	knn->train(training_data, training_labels, cv::Mat(), false, 1);

	cv::Mat neighborResponses, dists;

	float resultNode = knn->find_nearest(testing_data, k, testing_labels, neighborResponses, dists);

	neighborResponses.release();
	dists.release();

	delete knn;
}


void buildSVMFile(std::vector<float> &data, int label, const char *filename)
{

	std::ofstream output(filename, std::ofstream::out | std::ofstream::app);

	if(!output)
	{
		printf("File not opened \n");
		return;
	}

	output << label << " "; 
	for(int j = 0; j < data.size(); ++j)
	{
		output << j+1 << ":" << data.at(j) << " ";
	}
	output << std::endl;


	output.close();
}


void svm_train(std::vector<std::vector<float> > &histogram, cv::Mat &sample_label)
{

	printf("Building SVM Model...\n");
	string svm_train_fn = "svm_model";
	initFile(svm_train_fn);
	for(int i = 0; i < histogram.size(); ++i)
	{ 
		int v_label = sample_label.at<float>(0, i);
		cout << v_label << endl;
		buildSVMFile(histogram.at(i), int(v_label), svm_train_fn.c_str());
	}

	// std::string scaling_command = "libsvm/./svm-scale -l -1 -u 1 -s ./train_data/svm_range " + svm_train_fn + " > " + svm_train_scale_fn;
	// std::system(scaling_command.c_str());

	// std::string svm_command = "libsvm/./svm-train -s 0 -c 1000 -t 2 -g 1 -e 0.1 " + svm_train_scale_fn;
	std::string svm_command = "libsvm/./svm-train -s 0 -c 100 -t 2 -g 10 -e 0.1 " + svm_train_fn;
	std::system(svm_command.c_str());

}


void bow_build(cv::Mat &train_data_m, cv::Mat &feature_class_id_m, cv::Mat &feature_video_id_m, cv::Mat &video_class, int num_of_files)
{
	printf("Building BOW codebook from %d features...\n", train_data_m.rows);

	int cluster_n = 400;

	cv::Mat kmeans_data_m, kmeans_labels_m, kmeans_centers_m, kmeans_data_diff_m;
	std::vector<std::vector<float> > histograms_v;
	std::vector<float> kmeans_labels_v, feature_class_id_v, kmeans_label_diff_v;
	std::vector<int> kmeans_data_idx_v, kmeans_data_diff_idx_v, feature_video_id_v;

	int max               = 0;
	float data_percentage = 0.2;
	int kmeans_data_size  = std::ceil( train_data_m.rows * data_percentage );

	feature_video_id_m.copyTo(feature_video_id_v);


  	// Randomly select kmeans_data_size rows from train_data_m to run kmeans
	randomMatrixSplit( train_data_m, kmeans_data_m, kmeans_data_diff_m, kmeans_data_idx_v, kmeans_data_diff_idx_v, kmeans_data_size);

  	// Run Kmeans with a reduced matrix
	printf("Running Kmeans for %d samples...\n",kmeans_data_m.rows);
	if(kmeans_data_m.rows < cluster_n)
		cluster_n = kmeans_data_m.rows;
	cv::kmeans(kmeans_data_m, cluster_n, kmeans_labels_m, cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, kmeans_centers_m);   

	int n(0);
	std::vector<int> kmeans_centers_labels_v;
	kmeans_centers_labels_v.reserve(kmeans_centers_m.rows);
	std::generate_n(std::back_inserter(kmeans_centers_labels_v), kmeans_centers_m.rows, [n]()mutable { return n++; });
	cv::Mat kmeans_centers_labels_m(kmeans_centers_labels_v);

  	// Save training data
	string data_samples_fn = "train_samples";
	initFile(data_samples_fn);
	matToFile(data_samples_fn.c_str(), kmeans_data_m);

	string data_samples_labels_fn = "train_samples_labels";
	initFile(data_samples_labels_fn);
	matToFile(data_samples_labels_fn.c_str(), kmeans_labels_m);

	string visual_words_fn = "visual_words";
	initFile(visual_words_fn);
	matToFile(visual_words_fn.c_str(), kmeans_centers_m);

	string visual_words_labels_fn = "visual_words_labels";
	initFile(visual_words_labels_fn);
	matToFile(visual_words_labels_fn.c_str(), kmeans_centers_labels_m);


	printf("Finding nearest neighboor...\n");
  	// classify the rest of the data based on the kmeans centroids
	cv::Mat kmeans_labels_diff_m;
	findNearestNeighbor(kmeans_centers_m, kmeans_centers_labels_m, kmeans_data_diff_m, kmeans_labels_diff_m, 1);

	feature_class_id_m.copyTo(feature_class_id_v);
	kmeans_labels_m.copyTo(kmeans_labels_v);

	std::vector<int> kmeans_labels_diff_v;
	kmeans_labels_diff_m.copyTo(kmeans_labels_diff_v);

  	// release memory
	kmeans_data_m.release();
	kmeans_labels_m.release();
	kmeans_centers_m.release();
	kmeans_labels_diff_m.release();
	kmeans_data_diff_m.release();

	printf("Initializing histograms...\n");
  	// Initialize histogram vector
	for (int i = 0; i < num_of_files; ++i)
	{
		std::vector<float> hist_i(cluster_n, 0);
		histograms_v.push_back(hist_i);
	}

	printf("histogram size %lu %lu\n", histograms_v.size(), histograms_v.at(0).size());

	printf("Building histogram...\n");
  	// Fill histograms
	for( int ii = 0; ii < kmeans_labels_v.size(); ++ii)
	{     
		int idx       = kmeans_data_idx_v.at(ii);
		int video_id  = feature_video_id_v.at(idx);
		int label     = kmeans_labels_v.at(ii);

		histograms_v.at(video_id).at(label) += 1;
	}

	for( int jj = 0; jj < kmeans_labels_diff_v.size(); ++jj)
	{     
		int idx       = kmeans_data_diff_idx_v.at(jj); 
		int video_id  = feature_video_id_v.at(idx);
		int label     = kmeans_labels_diff_v.at(jj);

		histograms_v.at(video_id).at(label) += 1;
	}  


	std::ofstream data_file;      // pay attention here! ofstream
	data_file.open("train_histograms");

	for (int i = 0; i < histograms_v.size(); ++i)
	{
		copy(histograms_v.at(i).begin(), histograms_v.at(i).end(), std::ostream_iterator<float>(data_file, " ")); 
		data_file << "\n";
	}
	data_file.close();

	// recognizeSVMEfficient_train(histograms_v, video_class);
	svm_train(histograms_v, video_class);

	histograms_v.clear();	
	return;
}


bool extract_features_from_file(const char* fileName, Mat &trajectory_feat, Mat &hog_feat, Mat &hof_feat, Mat &mbh_feat)
{

	ifstream input_file;  
	input_file.open (fileName, ios::in);

	if(!input_file.is_open())
	{
		printf("%s file not open\n", fileName);
		return false;
	}

	int traj_beg 	= 10;
	int traj_end 	= 40;
	int hog_end 	= traj_end + 96;
	int hof_end 	= hog_end + 108;
	int mbh_end 	= hof_end + 192;

	string line;

	while (getline(input_file, line)) 
	{
		istringstream stream(line);

		string x;
		int index = 0;
		vector<float> traj_row;
		vector<float> hog_row;
		vector<float> hof_row;
		vector<float> mbh_row;

		while (stream >> x) 
		{  
			if(index < traj_beg)
				++index;
			else if(index >= traj_beg && index < traj_end)
			{
				traj_row.push_back(stof(x));
				++index;
			}
			else if(index >= traj_end && index < hog_end)
			{
				hog_row.push_back(stof(x));
				++index;
			}
			else if(index >= hog_end && index < hof_end)
			{
				hof_row.push_back(stof(x));
				++index;
			}
			else if (index >= hof_end && index < mbh_end)
			{
				mbh_row.push_back(stof(x));
				++index;
			}

		}
		
		cv::Mat traj_row_(traj_row);
		traj_row_ = traj_row_.t();
		trajectory_feat.push_back(traj_row_);

		cv::Mat hog_row_(hog_row);
		hog_row_ = hog_row_.t();
		hog_feat.push_back(hog_row_);

		cv::Mat hof_row_(hof_row);
		hof_row_ = hof_row_.t();
		hof_feat.push_back(hof_row_);

		cv::Mat mbh_row_(mbh_row);
		mbh_row_ = mbh_row_.t();
		mbh_feat.push_back(mbh_row_);
	}

	input_file.close();
	return true;
}


void load_features_from_file(const char *path, vector<string> &files_to_process, vector<int> &files_to_process_label, Mat &trajectory_feat, 
	Mat &hog_feat, Mat &hof_feat, Mat &mbh_feat, Mat &samples_label, Mat &samples_file_id, int &num_of_files)
{


	for (int i = 0; i < files_to_process.size(); ++i)
	{
		cout << "loading file: " << files_to_process[i] << endl;

		extract_features_from_file(files_to_process[i].c_str(), trajectory_feat, hog_feat, hof_feat, mbh_feat);

		vector<float> samples_label_v(trajectory_feat.rows, files_to_process_label[i]);
		Mat samples_label_m( samples_label_v );
		samples_label.push_back(samples_label_m);

		vector<float> sample_file_id_v(trajectory_feat.rows, num_of_files);
		Mat sample_file_id_m( sample_file_id_v );
		samples_file_id.push_back(sample_file_id_m);
		++num_of_files;
	}

	Mat files_label(files_to_process_label);
	string file_path = string(path) + "/files_label";
	matToFile(file_path.c_str(), files_label);
	cout << "writing " << file_path << endl;

	file_path = string(path) + "/samples_file_id";
	matToFile(file_path.c_str(), samples_file_id);
	cout << "writing " << file_path << endl;

	file_path = string(path) + "/samples_label";
	matToFile(file_path.c_str(), samples_label);
	cout << "writing " << file_path << endl;
	
	file_path = string(path) + "/trajectory_features";
	matToFile(file_path.c_str(), trajectory_feat);
	cout << "writing " << file_path << endl;
	
	file_path = string(path) + "/hog_features";
	matToFile(file_path.c_str(), hog_feat);
	cout << "writing " << file_path << endl;
	
	file_path = string(path) + "/hof_features";
	matToFile(file_path.c_str(), hof_feat);
	cout << "writing " << file_path << endl;

	file_path = string(path) + "/mbh_features";
	matToFile(file_path.c_str(), mbh_feat);
	cout << "writing " << file_path << endl;

	return;

}


