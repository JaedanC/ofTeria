#pragma once
#ifndef OFXDEBUGGER_H
#define OFXDEBUGGER_H

#include "ofMain.h"

class ofxDebugger {
public:
	vector<string> debugStrings;

	void push(const string& message);
	void draw();
};
#endif /* OFXDEBUGGER_H */