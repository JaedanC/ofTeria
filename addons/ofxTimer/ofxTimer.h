#pragma once
#ifndef OFX_TIMER_H
#define PFX_TIMER_H

#include "ofMain.h"
#include "ofxDebugger/ofxDebugger.h"

class ofxTimer {
public:
	ofxTimer(string timerName) : timerName(timerName) {
		startPoint = chrono::high_resolution_clock::now();
	}

	~ofxTimer() {
		endPoint = chrono::high_resolution_clock::now();

		auto start = chrono::time_point_cast<chrono::microseconds>(startPoint).time_since_epoch().count();
		auto end = chrono::time_point_cast<chrono::microseconds>(endPoint).time_since_epoch().count();
		duration = (end - start) * 0.001;

		debugPush(timerName + " timer took " + ofToString(duration));
	}

private:
	string timerName;
	chrono::time_point<chrono::high_resolution_clock> startPoint;
	chrono::time_point<chrono::high_resolution_clock> endPoint;
	double duration;
};

#endif