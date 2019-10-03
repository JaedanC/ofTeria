#include "ofxDebugger.h"

void ofxDebugger::push(const string& message)
{
	debugStrings.push_back(message);
}

void ofxDebugger::draw() {
  for (unsigned int i = 0; i < debugStrings.size(); i++) {
    ofDrawBitmapString(debugStrings[i], 5, (i + 1) * 15);
  }
  debugStrings.clear();
}
