#include <stdio.h>
#include <stdio.h>
#include<iostream> 
#include<future>
#include<functional>
#include<thread>
#include<optional>
#include <queue>
#include <atomic> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <unordered_map>


using namespace cv;
using namespace std;
template <class T> class ThreadSafeQueue;
template <typename T> class FunctionWrapper;
template <typename T> class Pool;



template<class T>
class ThreadSafeQueue {

private:
    queue<T> queue_;
    mutex mu;
public:

    ThreadSafeQueue() {}
    ThreadSafeQueue(const ThreadSafeQueue&& other) {
        lock_guard<mutex> lg(mu);
        queue_ = move(other->queue_);
    }

    optional<T> pop() {
        lock_guard<mutex> lg(mu);
        if (queue_.empty()) {
            return nullopt;
        }
        auto tmp = move(queue_.front());
        queue_.pop();
        return move(tmp);
    }

    void push(T&& val) {
        lock_guard<mutex> lg(mu);
        queue_.push(move(val));
    }

    T front() {
        lock_guard<mutex> lg(mu);
        return queue_.front();
    }

    unsigned int size() {
        lock_guard<mutex> lg(mu);
        return queue_.size();
    }

    bool empty() {
        lock_guard<mutex> lg(mu);
        return queue_.empty();
    }
};


template <typename T>
class FunctionWrapper {
private:
    packaged_task<T()> mTask;
public:
    FunctionWrapper(packaged_task<T()> task) {
        mTask = move(task);
    }
    packaged_task<T()> getTask() {
        return move(mTask);
    }
};


template <typename T>
class Pool {
public:
    atomic_bool done;
    vector<thread> threads;
    ThreadSafeQueue<FunctionWrapper<T>> mQueue;
    void worker() {
        while (not done) {
            auto mWarpper = mQueue.pop();
            if (mWarpper) {
                cout<<" Processing task using thread :- "<<hash<thread::id>{}(this_thread::get_id())<<"\n";
                auto curr_task = move((*mWarpper).getTask());
                curr_task();
            }
            this_thread::sleep_for(chrono::seconds(3));
        }
        cout << "Thread done = " << done << "\n";
    }

    Pool(int numThreads=4) {
        for (int i = 0; i < numThreads; i++) {
            threads.push_back(move(thread(&Pool::worker, this)));
        }
    }

    ~Pool() {
        cout << "\nThreadPool Destructor called, joining threads ...\n";
        for (auto& th : threads) th.join();
    }

    future<T> submit(function<T()> func) {
        packaged_task<T()> task(func);
        future<T> fu = task.get_future();
        FunctionWrapper<T> mTask(move(task));
        //this_thread::sleep_for(chrono::seconds(1));
        mQueue.push(move(mTask));
        return fu;
    }
};


class Processor {


private:
    int penTipSize = 3; 

public:
    Processor (){}
    Processor(int penTipSize){
        this->penTipSize = penTipSize; 
    }
    Mat add(Mat& a, Mat& b) {
        auto c = a + b;
        //cout << "hi from task, added_value = " << c << "\n";
        return c;
    }

    pair<int, int> find_centroid(Mat& img) {
        Mat tmp;
        int n = connectedComponents(img, tmp);

        //cout << "tmp = " << tmp << "\n";
        unordered_map<int, int> counts;
        unordered_map<int, pair<int, int>> centroidX;
        unordered_map<int, pair<int, int>> centroidY;

        for (int i = 0; i < tmp.rows; i++) {
            for (int j = 0; j < tmp.cols; j++) {
                int label = tmp.at<int>(i, j);
                if (counts.find(label) != counts.end()) {
                    counts[label] += 1;
                    centroidX[label].first += i;
                    centroidX[label].second += 1;
                    centroidY[label].first += j;
                    centroidY[label].second += 1;
                }
                else {
                    counts[label] = 1;
                    centroidX[label] = make_pair(i, 1);
                    centroidY[label] = make_pair(j, 1);
                }
            }
        }

        int max_label = -1;
        int max_count = -1;

        auto it = counts.begin();
        while (it != counts.end()) {
            if (it->first != 0 and it->second > max_count) {
                max_count = it->second;
                max_label = it->first;
            }
            it++;
        }

        //cout << "\centroid = " << centroidX[max_label].first << "\n";
        //cout << "\centroid = " << centroidX[max_label].second << "\n";

        int cent_x = (centroidX[max_label].second > 0) ? centroidX[max_label].first / centroidX[max_label].second : -1;
        int cent_y = (centroidY[max_label].second > 0) ? centroidY[max_label].first / centroidY[max_label].second : -1;

        //cout << "\nmax_label = " << max_label << "\n";

        //int cent_x = 0;
        //int cent_y = 0;
        return make_pair(cent_x, cent_y);
    }

    void floodfill(int i, int j, Mat& img, int levels) {
        int level = levels;
        img.at<uchar>(i, j) = 255;
        for (int level = 1; level <= levels; level++) {
            for (int k = max(0, i - level); k <= min(i + level, img.rows); k++) {
                img.at<uchar>(k, max(j - level, 0)) = 255;
                img.at<uchar>(k, min(j + level, img.cols)) = 255;
            }
            for (int k = max(0, j - level); k <= min(j + level, img.cols); k++) {
                img.at<uchar>(max(i - level, 0), k) = 255;
                img.at<uchar>(min(i + level, img.rows), k) = 255;
            }

        }
    }

    bool inBound(pair<int, int> centroid, int rows, int cols) {
        return ((0 <= centroid.first) and
                (centroid.first < rows) and
                (0 <= centroid.second) and
                (centroid.second < cols)
                );
    }

    bool detectPenTip(Mat& img, Mat& board, Size& size) {
        filterBlueColorTip(img, size);
        auto centroid = this->find_centroid(img); 
        if (this->inBound(centroid, board.rows, board.cols)) {
            this->floodfill(centroid.first, centroid.second, board, this->penTipSize);
            cout << "centroid = " << centroid.first << "  " << centroid.second << "\n";
            return true;
        }
        cout << "returning False ";
        return false;
    }

    void filterBlueColorTip(Mat& img, Size& size) {
        resize(img, img, size);
        Mat rgbChannel[3];
        split(img, rgbChannel);
        threshold(rgbChannel[0] - rgbChannel[2], img, 50, 255, 0);
    }


};



class BoardDisplayer {

public:
    BoardDisplayer (){}
    void displayContentFromFutureQueue(ThreadSafeQueue<future<bool>>& procQueue, Mat& board) {

        while (not procQueue.empty()) {
            auto fu_ = move(procQueue.pop());
            if (not fu_) continue;
            auto status = (*fu_).wait_for(chrono::milliseconds(20));
            if (status != future_status::ready) {
                procQueue.push(move((*fu_)));
            }
            else {
                auto doneProcessing = (*fu_).get();
                if (doneProcessing)
                {
                    imshow("test", board);
                    waitKey(0);
                }
            }
        }


    }

};



int main()
{   //ThreadPool 
    Pool<bool>* pool = new Pool<bool>();
    Processor* proc = new Processor();
    BoardDisplayer* boardDisplayer = new BoardDisplayer();
    Size size(255, 255);

    Mat board(size, CV_8UC1, Scalar(0));
    Mat img1 = imread("D:/projects/opencv/opencv_1/pics/left_up.jpg");
    Mat img2 = imread("D:/projects/opencv/opencv_1/pics/centre.jpg");
    Mat img3 = imread("D:/projects/opencv/opencv_1/pics/right_up.jpg");
    Mat img4 = imread("D:/projects/opencv/opencv_1/pics/left_middle.jpg");
    Mat img5 = imread("D:/projects/opencv/opencv_1/pics/right_middle.jpg");

    vector<Mat> inputImages;
    inputImages.push_back(img1);
    inputImages.push_back(img2);
    inputImages.push_back(img3);
    inputImages.push_back(img4);
    inputImages.push_back(img5);

    //futures
    ThreadSafeQueue<future<bool>> procQueue;
    for (auto& img : inputImages) {
        auto penTipFun = bind(&Processor::detectPenTip, proc, img, board, size);
        function<bool()> func(penTipFun);
        future<bool> fu = pool->submit(func);
        procQueue.push(move(fu));
    }
    boardDisplayer->displayContentFromFutureQueue(procQueue, board);
    pool->done = true;
    delete pool;
    return 0;
}