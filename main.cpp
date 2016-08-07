 // g++ -O3 -std=c++11 train.cpp -o main `pkg-config --libs opencv` -lboost_system -lboost_filesystem -Ilibsvm -O3 libsvm/svm.cpp `pkg-config --cflags opencv`


# include <stdio.h>
# include <iostream>
# include <fstream>
# include "opencv2/opencv_modules.hpp"
# include "libsvm/svm.h"
# include "Utils.h"

using namespace std;
using namespace cv;

int load_and_split_features = 1;
int load_features 			= 0;
int build_bow 		 		= 0;
int train 	 		 		= 0;
int test 	 		 		= 0;

extern string root_path;


int main( int argc, char** argv )
{

	Mat trajectory_feat;
	Mat hog_feat;
	Mat hof_feat;
	Mat mbh_feat;
	Mat samples_label;
	Mat samples_file_id;
	Mat files_label;
	int num_of_files = 0;
	int mode = 2;

	if (load_and_split_features)
	{
		string feat_path;
		string path = "/Volumes/SOLEDAD/decompressed_features/";

		if(mode == 2){
			feat_path = root_path + "test_data";
			string cmd = "mkdir " + root_path + "test_data";
			system(cmd.c_str());
		}
		else
		{
			feat_path = root_path + "train_data";
			string cmd = "mkdir " + root_path + "train_data";
			system(cmd.c_str());
		}

		int initial_label = 1;
		vector<vector<string> > files;
		vector<string> labels_name;
		vector<int> files_labeles;

		listDir(path.c_str(), files, labels_name, files_labeles, initial_label);

		
		vector<string> files_to_process;
		vector<int> files_to_process_label;
		splitDataSet(files, files_to_process, files_to_process_label, mode);

		// Mat files_label(files_to_process_label);

		cout << endl << "Loading and splitting files..." << endl;
	// load feature files
		load_features_from_file(feat_path.c_str(), files_to_process, files_to_process_label, trajectory_feat, hog_feat, hof_feat, mbh_feat, 
			samples_label, samples_file_id, num_of_files);
	// for (int i = 0; i < files_to_process.size(); ++i)
	// {
	// 	cout << "loading file: " << files_to_process[i] << endl;

	// 	extract_features_from_file(files_to_process[i].c_str(), trajectory_feat, hog_feat, hof_feat, mbh_feat);

	// 	vector<float> samples_label_v(trajectory_feat.rows, files_to_process_label[i]);
	// 	Mat samples_label_m( samples_label_v );
	// 	samples_label.push_back(samples_label_m);

	// 	vector<float> sample_file_id_v(trajectory_feat.rows, num_of_files);
	// 	Mat sample_file_id_m( sample_file_id_v );
	// 	samples_file_id.push_back(sample_file_id_m);
	// 	++num_of_files;
	// }

	// matToFile("files_label", files_label);
	// matToFile("trajectory_features", trajectory_feat);
	// matToFile("hog_features", hog_feat);
	// matToFile("hof_features", hof_feat);
	// matToFile("mbh_features", mbh_feat);

	// fileToMat("trajectory_features", trajectory_feat);
	// fileToMat("hog_features", hog_feat);
	// fileToMat("hof_features", hof_feat);
	// fileToMat("mbh_features", mbh_feat);
	}

	if (load_features && build_bow)
	{
		cout << endl << "Loading files and building BOW..." << endl;

		string file_name = root_path + "train_data/trajectory_features";
		fileToMat(file_name.c_str(), trajectory_feat);

		file_name = root_path + "train_data/hog_features";
		fileToMat(file_name.c_str(), hog_feat);

		file_name = root_path + "train_data/hof_features";
		fileToMat(file_name.c_str(), hof_feat);
		
		file_name = root_path + "train_data/mbh_features";
		fileToMat(file_name.c_str(), mbh_feat); 
		
		file_name = root_path + "train_data/files_label";
		fileToMat(file_name.c_str(), files_label); 
		
		file_name = root_path + "train_data/samples_label";
		fileToMat(file_name.c_str(), samples_label);

		file_name = root_path + "train_data/samples_file_id";
		fileToMat(file_name.c_str(), samples_file_id); 

		bow_build(trajectory_feat, samples_label, samples_file_id, files_label, files_label.rows);
	}
	else if(build_bow)
	{
		cout << endl << "Building BOW..." << endl;
		bow_build(trajectory_feat, samples_label, samples_file_id, files_label, files_label.rows);
	}


	if (test)
	{
		system("mkdir test_data");

		Mat visual_words_m;
		Mat visual_words_labels_m;
		
		string visual_words_fn 			= root_path + "/train_data/visual_words";
		string visual_words_labels_fn 	= root_path + "/train_data/visual_words_labels";

		bool st;

		st = fileToMat(visual_words_fn.c_str(), visual_words_m);

		if(st == false || visual_words_m.rows == 0){
			printf("File %s was not found or is empty\n", visual_words_fn.c_str());
			return 0;
		}

		st = fileToMat(visual_words_labels_fn.c_str(), visual_words_labels_m);

		if(st == false || visual_words_labels_m.rows == 0){
			printf("File %s was not found or is empty\n", visual_words_labels_fn.c_str());
			return 0;
		}
	}

	return 0;
}