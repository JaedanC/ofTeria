#include "ofApp.h"
#include "ofxDebugger/ofxDebugger.h"
#include "ofxMemoryMapping/ofxMemoryMapping.h"
#include "ofxGameStates/ofxGameEngine.h"
#include "States/MenuState.h"

ofxDebugger debug;
ofxGameEngine* engine;

//--------------------------------------------------------------
void ofApp::setup(){
	engine = ofxGameEngine::Instance();
	engine->setup();
	engine->PushState(MenuState::Instance());
}

//--------------------------------------------------------------
void ofApp::update(){
	engine->update();
}

//--------------------------------------------------------------
void ofApp::draw(){
	// Background
	ofBackground(ofColor::dimGrey);

	// GameState draw
	engine->draw();

	// Debug
	ofSetColor(ofColor::white);
	debug.push("FrameRate: " + ofToString(ofGetFrameRate()));
	debug.draw();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
