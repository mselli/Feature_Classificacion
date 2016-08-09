 // g++ -O3 -std=c++11 train.cpp -o main `pkg-config --libs opencv` -lboost_system -lboost_filesystem -Ilibsvm -O3 libsvm/svm.cpp `pkg-config --cflags opencv`

# include <stdio.h>
# include <iostream>
# include <fstream>
# include "opencv2/opencv_modules.hpp"
# include "libsvm/svm.h"
# include "Utils.h"

using namespace std;
using namespace cv;

int load_and_split_features = 0;
int load_train_features 	= 0;
int build_bow_and_svm 		= 0;
int test 	 	 		 	= 1;
int nn 	 		 			= 1;
int svm 	 		 		= 0;

extern string root_path;
extern 	int cluster_n;


int main( int argc, char** argv )
{

	Mat trajectory_feat;
	Mat hog_feat;
	Mat hof_feat;
	Mat mbh_feat;
	Mat samples_label;
	Mat samples_file_id;
	Mat files_label;

	int num_of_files 	= 0;
	int mode 			= 2;

	string path = "/Volumes/SOLEDAD/decompressed_features/";
	string feat_path;

	string cmd = "mkdir " + root_path + "bow";
	system(cmd.c_str());
	cmd = "mkdir " + root_path + "svm";
	system(cmd.c_str());

	if (load_and_split_features)
	{

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
		vector<int> files_labels;

		listDir(path.c_str(), files, labels_name, files_labels, initial_label);

		
		vector<string> files_to_process;
		vector<int> files_to_process_label;
		splitDataSet(files, files_to_process, files_to_process_label, mode);

		cout << endl << "Loading and splitting files..." << endl;
		// load feature files
		load_features_from_file(feat_path.c_str(), files_to_process, files_to_process_label, trajectory_feat, hog_feat, hof_feat, mbh_feat, 
			samples_label, samples_file_id, num_of_files);

		Mat files_label(files_to_process_label);
	}

	if (load_train_features)
	{
		string file_name;
		// trajectory_feat.release();
		// file_name = root_path + "train_data/trajectory_features";
		// cout << "Loading "<< file_name << "..." << endl;
		// fileToMat(file_name.c_str(), trajectory_feat);

		hog_feat.release();
		file_name = root_path + "train_data/hog_features";
		cout << "Loading "<< file_name << "..." << endl;
		fileToMat(file_name.c_str(), hog_feat);

		// hof_feat.release();
		// file_name = root_path + "train_data/hof_features";
		// cout << "Loading "<< file_name << "..." << endl;
		// fileToMat(file_name.c_str(), hof_feat);
		
		// file_name = root_path + "train_data/mbh_features";
		// cout << "Loading "<< file_name << "..." << endl;
		// fileToMat(file_name.c_str(), mbh_feat); 
		
		files_label.release();
		file_name = root_path + "train_data/files_label";
		cout << "Loading "<< file_name << "..." << endl;
		fileToMat(file_name.c_str(), files_label); 
		
		samples_label.release();
		file_name = root_path + "train_data/samples_label";
		cout << "Loading "<< file_name << "..." << endl;
		fileToMat(file_name.c_str(), samples_label);

		samples_file_id.release();
		file_name = root_path + "train_data/samples_file_id";
		cout << "Loading "<< file_name << "..." << endl;
		fileToMat(file_name.c_str(), samples_file_id); 

	}

	if(build_bow_and_svm)
	{
		// Mat files_label;
		// string file_name = root_path + "train_data/files_label";
		// cout << "Loading "<< file_name << "..." << endl;
		// fileToMat(file_name.c_str(), files_label); 

		cout << endl << "Building BOW and SVM model from " << files_label.rows << " files ..." << endl;
		bow_build(hog_feat, samples_label, samples_file_id, files_label, files_label.rows);
	}

	if (test)
	{

		int initial_label = 1;
		vector<vector<string> > files;
		vector<string> labels_name;
		vector<int> files_labels;
		listDir(path.c_str(), files, labels_name, files_labels, initial_label);

		vector<string> files_to_process;
		vector<int> files_to_process_label;
		splitDataSet(files, files_to_process, files_to_process_label, mode);

		Mat visual_words_m;
		Mat visual_words_labels_m;


		string visual_words_fn 			= root_path + "bow/visual_words";
		string visual_words_labels_fn 	= root_path + "bow/visual_words_labels";

		bool st;

		st = fileToMat(visual_words_fn.c_str(), visual_words_m);
		cout << "Loading "<< visual_words_fn << "..." << endl;

		if(st == false || visual_words_m.rows == 0)
		{
			printf("File %s was not found or is empty\n", visual_words_fn.c_str());
			return 0;
		}

		st = fileToMat(visual_words_labels_fn.c_str(), visual_words_labels_m);
		cout << "Loading "<< visual_words_labels_fn << "..." << endl;

		if(st == false || visual_words_labels_m.rows == 0)
		{
			printf("File %s was not found or is empty\n", visual_words_labels_fn.c_str());
			return 0;
		}

		Mat train_histograms;
		Mat train_histograms_labels;

		if(nn)
		{
			string train_hist_fn = root_path + "svm/train_histograms";
			cout << "Loading " << train_hist_fn << "..." << endl;
			ifstream input_file(train_hist_fn.c_str());

			if (input_file) 
			{        
				string line;
				while (getline(input_file, line)) 
				{
					istringstream stream(line);
					string x;
					vector<float> row;

					while (stream >> x) 
					{
						row.push_back(stof(x));
					}
					Mat row_i(row);
					row_i = row_i.t();
					train_histograms.push_back(row_i);
				}
			}
			input_file.close();

			string train_histograms_labels_fn = root_path + "train_data/files_label";
			st = fileToMat(train_histograms_labels_fn.c_str(), train_histograms_labels);
			cout << "train histograms: " << train_histograms.size() << endl;
			cout << "train histograms labels: " << train_histograms_labels.size() << endl;
		}

		int correct_label = 0;

		// load feature files

		Mat files_to_process_label_m(files_to_process_label);
		cout << "Processing " << files_to_process_label_m.size() << endl;

		for (int i = 0; i < files_to_process.size(); ++i)
		{
			int label = files_to_process_label[i];

			cout << "Splitting features from "<< files_to_process[i] << endl;
			extract_features_from_file(files_to_process[i].c_str(), trajectory_feat, hog_feat, hof_feat, mbh_feat);

			vector<float> bow_histogram(cluster_n, 0.f);

			cout << "building BOW..." << endl;
			get_bow_histogram(hog_feat, label, bow_histogram, visual_words_m, visual_words_labels_m);
			

			if(nn)
			{
				cout << "running NN..." << endl;

				Mat histogram_i(bow_histogram);
				histogram_i = histogram_i.t();
				// cout << histogram_i.size() << endl;

				Mat histogram_label_i;
				// CvKNearest* knn = new CvKNearest();
				// knn->train(train_histograms, files_to_process_label_m);

				// cout << knn->get_sample_count() << endl;

				// cout << "finding NN..." << endl;
				// Mat neighborResponses, dists;
				// float resultNode = knn->find_nearest(histogram_i, 1);

				cout << train_histograms.size() << " " << train_histograms_labels.size() << " " << histogram_i.size() << " " << histogram_label_i.size() << endl;

				findNearestNeighbor(train_histograms, train_histograms_labels, histogram_i, histogram_label_i, 1);

				cout << "label: " << histogram_label_i.at<float>(0, 0) << " ground truth: " <<  label << endl << endl;

				if(int( histogram_label_i.at<float>(0, 0) ) == label )
					++correct_label;

				cout << "correct labels: " << correct_label << " num of files " << files_to_process.size() << endl;

				histogram_i.release();
				histogram_label_i.release();
			}

			if(svm)
			{
				cout << endl << "svm predict..." << endl;
				svm_predict(bow_histogram, label);
			}

			
			bow_histogram.clear();
			trajectory_feat.release();
			hog_feat.release();
			hof_feat.release();
			mbh_feat.release();

		}

		cout << "accuracy: " << correct_label / float(files_to_process.size()) << endl;
	}

	return 0;
}