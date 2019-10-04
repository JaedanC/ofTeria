#include "ofApp.h"
#include "ofxDebugger/ofxDebugger.h"
#include "ofxIni/ofxIniFile.h"
#include "ofxGameStates/ofxGameEngine.h"
#include "States/MenuState.h"
#include "Settings/Settings.h"

ofxDebugger debug;
ofxGameEngine* engine;

//--------------------------------------------------------------
void ofApp::setup(){
	ofxIniFile settings("settings.ini");

	// ------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------
	/* Load the games default settings and apply them to the game. If the settings
	do not exist, write the default values to the settings file. */
	// ------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------
	bool fullscreen		= Settings::loadSetBool(settings, "game", "fullscreen", false);
	bool vsync			= Settings::loadSetBool(settings, "game", "vsync", false);
	int target_fps		= Settings::loadSetInt(settings, "game", "target_fps", 2000);
	if (!fullscreen) {
		// Only set the window size if the game is not fullscreen
		int window_width  = Settings::loadSetInt(settings, "game", "window_width", 800);
		int window_height = Settings::loadSetInt(settings, "game", "window_height", 600);
		ofSetWindowShape(window_width, window_height);
	}

	// Will still set the target fps if we are in vsync since it might be turned off later.
	ofSetFrameRate(target_fps);
	ofSetFullscreen(fullscreen);
	ofSetVerticalSync(vsync);
	// ------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------

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
	debug.push("WindowPosition: " + ofToString(ofVec2f(ofGetWindowPositionX(), ofGetWindowPositionY())));
	debug.push("WindowSize: " + ofToString(ofGetWindowSize()));
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
